from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
      ):                                                                                                                 
          # Eager-load: with N≈1000 and 16x20 float32 the whole thing is ~2.5MB.                                         
          # Cheaper and safer than holding an h5py handle (see split_datasets).                                          
          self.surfaces = surfaces.astype(np.float32, copy=False)                                                        
          self.params_raw = params.astype(np.float32, copy=False)                                                        
          self.normalizer = normalizer                                                                                   
          self.params_z = normalizer.encode(self.params_raw).astype(np.float32)                                          
                                                                                                                         
      def __len__(self) -> int:                                                                                          
          return self.surfaces.shape[0]                                                                                  
                          
      def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:                                              
          # Add a channel dimension: FNO expects (C, H, W). Here C=1 because
          # the IV surface is a single scalar field over (k, tau).                                                       
          surface = torch.from_numpy(self.surfaces[idx]).unsqueeze(0)                                                    
          target  = torch.from_numpy(self.params_z[idx])                                                                 
          return surface, target

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
