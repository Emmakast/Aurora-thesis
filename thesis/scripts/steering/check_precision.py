import os
import torch
from pathlib import Path

OUT_DIR = Path("/scratch-shared/ekasteleyn/aurora_latents/")
files = list(OUT_DIR.glob("*.pt"))

total_fp32_size = 0
total_fp16_size = 0

print(f"{'File Name':<35} | {'Max Diff':<12} | {'Mean Diff':<12} | {'S FP32':<9} | {'S FP16':<9}")
print("-" * 88)

for filepath in files:
    if "decoder" in filepath.name:
        continue # Ignore old decoder latents if they exist
        
    fp32_size = os.path.getsize(filepath)
    fp16_size = fp32_size / 2 # Float16 takes exactly half the memory
    
    total_fp32_size += fp32_size
    total_fp16_size += fp16_size
    
    # Load float32 tensor
    tensor_fp32 = torch.load(filepath, map_location="cpu")
    
    # Cast to float16 and back to float32
    tensor_fp16 = tensor_fp32.half()
    tensor_reconstructed = tensor_fp16.float()
    
    # Calculate difference
    diff = torch.abs(tensor_fp32 - tensor_reconstructed)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"{filepath.name:<35} | {max_diff:<12.3e} | {mean_diff:<12.3e} | {fp32_size/1e6:<7.1f}MB | {fp16_size/1e6:<7.1f}MB")

print("-" * 88)
print(f"Total size in float32: {total_fp32_size / 1e9:.2f} GB")
print(f"Total size in float16: {total_fp16_size / 1e9:.2f} GB")

