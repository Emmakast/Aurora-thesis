import sys
from pathlib import Path
import torch
import xarray as xr
sys.path.append(str(Path("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/data_loader")))
from extract_latents_hres import prepare_batch, download_data, download_static
from aurora import Aurora, rollout

device = torch.device("cpu")
model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
model.eval()

target_layer_name = "backbone.encoder_layers.2"
target_layer = dict([*model.named_modules()])[target_layer_name]

def hook(module, args, output):
    print("args[2] is:", args[2])
    x = output[0] if isinstance(output, tuple) else output
    print("x shape:", x.shape)
    sys.exit(0)

target_layer.register_forward_hook(hook)

download_path = Path("/scratch-shared/ekasteleyn/downloads/hres_t0")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4").load()
batch = prepare_batch("2017-03-08", download_path, init_hour=0, static_vars_ds=static_vars_ds)
for pred in rollout(model, batch, steps=1):
    pass
