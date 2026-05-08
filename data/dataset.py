from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

@dataclass
class ParamNormalizer:
    '''
    Doing this because each parameter for Heston model has very different ranges,
    using them raw will have larger terms dominating smaller ones
    '''
    mean: np.ndarray
    std: np.ndarray

    def encode(self, p: np.ndarray) -> np.ndarray:
        return (p - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

class HestonSurfaceDataset(Dataset):
      def __init__(                                                                                                      
        self,           
        surfaces: np.ndarray,   # (N, n_k, n_tau)
        params:   np.ndarray,   # (N, 5) raw, un-normalized                                                            
        normalizer: ParamNormalizer,                                                                                   
        mask_fraction: float = 0.0 
      ):                                                                                                                 
          # Eager-load: with N≈1000 and 16x20 float32 the whole thing is ~2.5MB.                                         
          # Cheaper and safer than holding an h5py handle (see split_datasets).                                          
          self.surfaces = surfaces.astype(np.float32, copy=False)                                                        
          self.params_raw = params.astype(np.float32, copy=False)                                                        
          self.normalizer = normalizer                                                                                   
          self.params_z = normalizer.encode(self.params_raw).astype(np.float32)                                          
          self.mask_fraction = float(mask_fraction)

      def __len__(self) -> int:                                                                                          
          return self.surfaces.shape[0]                                                                                  

      def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:                                              
          # Clone is used to ensure that we don't affect stored data, each instance is a copy of the underlying data
          surface = torch.from_numpy(self.surfaces[idx]).unsqueeze(0).clone()
          if self.mask_fraction > 0.0:
              keep = np.random.random(surface.shape[1:]) > self.mask_fraction
              surface = surface * torch.from_numpy(keep.astype(np.float32))
          # Add a channel dimension: FNO expects (C, H, W). Here C=1 because
          # the IV surface is a single scalar field over (k, tau).                                                       
          target  = torch.from_numpy(self.params_z[idx])                                                                 
          return surface, target

def _load_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        return f["surfaces"][:], f["parameters"][:]

def _fit_or_load_normalizer(train_params: np.ndarray, stats_path: str) -> ParamNormalizer:
    '''
    If stats_path exists, load it, otherwise build from scratch with sidecar
    '''
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        return ParamNormalizer(
                mean=np.array(stats["mean"], dtype=np.float32),
                std=np.array(stats["std"], dtype=np.float32)
                )
    mean = train_params.mean(axis=0)
    std = train_params.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard against a degenerate constant column
    normalizer = ParamNormalizer(mean=mean, std=std)
    with open(stats_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    return normalizer

def load_datasets(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    ood_path: str | None = None,
    norm_stats_path: str | None = None,
    mask_fraction: float = 0.0,
) -> tuple[
    HestonSurfaceDataset,
    HestonSurfaceDataset | None,
    HestonSurfaceDataset | None,
    HestonSurfaceDataset | None,
    ParamNormalizer,
]:
    """
    Phase 1 multi-file loader. Train file is required; others are optional.
    Normalizer is fit on the train set's params (or loaded from the JSON sidecar
    at norm_stats_path if it already exists) and shared across all splits.
    mask_fraction applies only to the train set — eval splits stay clean so
    val/test loss measures model accuracy, not augmentation noise.
    """
    train_surfaces, train_params = _load_h5(train_path)

    # Default sidecar path lives next to the train file: foo.h5 -> foo.norm.json
    if norm_stats_path is None:
        norm_stats_path = train_path.replace(".h5", ".norm.json")
    normalizer = _fit_or_load_normalizer(train_params, norm_stats_path)

    train_ds = HestonSurfaceDataset(
        train_surfaces, train_params, normalizer, mask_fraction=mask_fraction
    )
    val_ds = test_ds = ood_ds = None
    if val_path:
        s, p = _load_h5(val_path)
        val_ds = HestonSurfaceDataset(s, p, normalizer)
    if test_path:
        s, p = _load_h5(test_path)
        test_ds = HestonSurfaceDataset(s, p, normalizer)
    if ood_path:
        s, p = _load_h5(ood_path)
        ood_ds = HestonSurfaceDataset(s, p, normalizer)

    return train_ds, val_ds, test_ds, ood_ds, normalizer


def split_datasets(
  h5_path: str,                                                                                                      
  train_frac: float,  
  val_frac: float,
  seed: int,                                                                                                         
) -> tuple[HestonSurfaceDataset, HestonSurfaceDataset, HestonSurfaceDataset, ParamNormalizer]:
    """                                                                                                                
    Load the HDF5 once, deterministically permute, split into train/val/test,
    fit the normalizer on TRAIN ONLY, and return three datasets that share it.                                         
    """                                                                                                                
    with h5py.File(h5_path, "r") as f:                                                                                 
        surfaces = f["surfaces"][:]      # (N, n_k, n_tau)                                                             
        params   = f["parameters"][:]    # (N, 5)                                                                      

    n = surfaces.shape[0]                                                                                              
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                                                                                          
                      
    n_train = int(round(train_frac * n))                                                                               
    n_val   = int(round(val_frac   * n))
    idx_train = perm[:n_train]                                                                                         
    idx_val   = perm[n_train : n_train + n_val]
    idx_test  = perm[n_train + n_val :]                                                                                

    # Fit normalizer on TRAIN ONLY. Using the full dataset's mean/std would                                            
    # leak test-set information into training (the mean is a function of every
    # sample). Tiny effect here, but it's the correct contract.                                                        
    train_params = params[idx_train]                                                                                   
    mean = train_params.mean(axis=0)                                                                                   
    std  = train_params.std(axis=0)                                                                                    
    std  = np.where(std < 1e-8, 1.0, std)   # guard against a degenerate constant column                               
    normalizer = ParamNormalizer(mean=mean, std=std)                                                                   
                                                                                                                     
    train = HestonSurfaceDataset(surfaces[idx_train], params[idx_train], normalizer)                                   
    val   = HestonSurfaceDataset(surfaces[idx_val],   params[idx_val],   normalizer)
    test  = HestonSurfaceDataset(surfaces[idx_test],  params[idx_test],  normalizer)                                   
    return train, val, test, normalizer
