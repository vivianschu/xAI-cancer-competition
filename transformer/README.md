Set env to optimize memory fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

To run the transformer model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py
```

If you would like to log your runs on W&B:
```bash
wandb login
```