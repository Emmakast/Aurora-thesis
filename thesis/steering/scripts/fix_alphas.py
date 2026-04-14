import re

with open('thesis/steering/scripts/steer_aurora.py', 'r') as f:
    code = f.read()

# Change --alpha to --alphas
code = code.replace(
    'parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")',
    'parser.add_argument("--alphas", type=float, nargs="+", default=[1.0], help="List of steering strengths")'
)

# Fix print statement
code = code.replace(
    'print(f"Starting Contrastive Activation Addition (CAA) Steering Pipeline ({args.phenomenon}, alpha={args.alpha})...")',
    'print(f"Starting Contrastive Activation Addition (CAA) Steering Pipeline ({args.phenomenon}, alphas={args.alphas})...")'
)

# Extract and rewrite the rollout / saving loop
start_marker = "    # Move to device (handle tuples or direct tensors)"

end_idx = code.find("if __name__ == \"__main__\":")
start_idx = code.find(start_marker)

if start_idx != -1 and end_idx != -1:
    new_tail = """    # Move to device (handle tuples or direct tensors)
    if isinstance(batch, tuple):
        batch = tuple(t.to(device) if hasattr(t, 'to') else t for t in batch)
    else:
        batch = batch.to(device)
    
    base_output_filename = f"base_{args.phenomenon.lower()}{suffix_str}_{date_tag}_alpha_0.0.nc"
    
    if not os.path.exists(base_output_filename):
        print("Running base inference (alpha=0.0) without hook...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=1):
                base_pred_batch = pred
                break
                
        base_pred_batch = base_pred_batch.to("cpu")
        base_ds = batch_to_dataset(base_pred_batch, step=1)
        
        tmp_base_filename = f"{base_output_filename}.tmp_base"
        base_ds.to_netcdf(tmp_base_filename)
        import os
        os.rename(tmp_base_filename, base_output_filename)
        print(f"Saved base output to {base_output_filename}")
    else:
        print(f"Base output {base_output_filename} already exists, skipping base inference.")
        
    for alpha_val in args.alphas:
        print(f"Applying hook with alpha={alpha_val}...")
        
        # NOTE: aurora batch tuples might have issues.
        # But base inference logic is identical to steer inference logic inside the hook
        hook_handle = model.backbone.encoder_layers[2].register_forward_hook(
            make_intervention_hook(masked_delta_v, alpha=alpha_val)
        )
        
        print(f"Running steered inference (alpha={alpha_val})...")
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=1):
                pred_batch = pred
                break
                
        pred_batch = pred_batch.to("cpu")
            
        # Remove hook when done
        hook_handle.remove()
        
        print(f"Converting prediction to xarray for alpha={alpha_val}...")
        ds = batch_to_dataset(pred_batch, step=1)

        output_filename = f"steered_{args.phenomenon.lower()}{suffix_str}_{date_tag}_alpha_{alpha_val}.nc"
        ds.to_netcdf(output_filename)
        print(f"Saved steered output to {output_filename}")

    print("Done!")

"""
    code = code[:start_idx] + new_tail + code[end_idx:]

with open('thesis/steering/scripts/steer_aurora.py', 'w') as f:
    f.write(code)

