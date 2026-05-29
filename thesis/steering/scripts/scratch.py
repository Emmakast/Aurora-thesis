import torch
from aurora import Aurora
model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
model.eval()
print(dict(model.named_modules()).keys())

target = dict(model.named_modules())["backbone.encoder_layers.2"]
def hook(module, args, output):
    if isinstance(output, tuple):
        print("Output shape:", [o.shape for o in output if isinstance(o, torch.Tensor)])
    else:
        print("Output shape:", output.shape)
target.register_forward_hook(hook)

# dummy input
# wait we need to prepare a batch to see the actual shape
