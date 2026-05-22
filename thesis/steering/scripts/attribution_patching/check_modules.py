import torch
from aurora import Aurora

model = Aurora()
print([name for name in dict([*model.named_modules()]).keys() if 'encoder' in name])
