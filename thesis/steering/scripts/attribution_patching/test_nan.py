import torch
from aurora import Aurora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
model.eval()
model = model.to(device)

import sys
from pathlib import Path
sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/data_loader")))
from extract_latents_hres import prepare_batch, download_data, download_static
import xarray as xr

download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
batch = prepare_batch("2017-03-08", download_path, init_hour=12, static_vars_ds=static_vars_ds)
batch = batch.to(device)

saved_tensors = {}
def hook_fn(module, input, output):
    tensor = output[0] if isinstance(output, tuple) else output
    saved_tensors['latent'] = tensor.detach().cpu()

target_layer = dict([*model.named_modules()])["backbone.encoder_layers.2"]
target_layer.register_forward_hook(hook_fn)

from aurora import rollout
import sys
print("Running aurora.rollout...")
for pred in rollout(model, batch, steps=1):
    pass

l1 = saved_tensors['latent']
print("rollout nans:", torch.isnan(l1).sum().item())

# Now run the manual way
batch2 = prepare_batch("2017-03-08", download_path, init_hour=12, static_vars_ds=static_vars_ds)
batch2 = batch2.to(device)

p = next(model.parameters())
batch2 = model.batch_transform_hook(batch2)
batch2 = batch2.type(p.dtype)
batch2 = batch2.crop(model.patch_size)

print("Running manual forward pass...")
pred2 = model.forward(batch2)
l2 = saved_tensors['latent']
print("manual nans:", torch.isnan(l2).sum().item())
