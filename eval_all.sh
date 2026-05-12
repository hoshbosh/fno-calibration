for ckpt in checkpoints/{fno,mlp}_seed*_best.pt; do                                                                                                                                
    python eval.py --checkpoint "$ckpt" --split test
    python eval.py --checkpoint "$ckpt" --split ood                                                                                                                                
done
