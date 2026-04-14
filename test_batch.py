import torch
import aurora
import sys

b = torch.load('/scratch-shared/ekasteleyn/aurora_data/batch_2016-01-13.pt', map_location='cpu', weights_only=False)
print("Type of b:", type(b))

def process(obj, name):
    if hasattr(obj, 'shape'):
        print(f"{name}: shape {obj.shape}, type {type(obj)}")
    elif isinstance(obj, tuple) or isinstance(obj, list):
        print(f"{name}: tuple/list of len {len(obj)}")
        for i, item in enumerate(obj):
            process(item, f"{name}[{i}]")
    elif isinstance(obj, dict):
        print(f"{name}: dict of len {len(obj)}")
        for k, v in obj.items():
            process(v, f"{name}.{k}")
    elif hasattr(obj, '__dict__'):
        print(f"{name}: object with __dict__")
        for k, v in obj.__dict__.items():
            process(v, f"{name}.{k}")
    elif hasattr(obj, '__dataclass_fields__'):
        print(f"{name}: dataclass")
        for k in obj.__dataclass_fields__:
            process(getattr(obj, k), f"{name}.{k}")
    else:
        print(f"{name}: {type(obj)}")

process(b, "b")
