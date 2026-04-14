import sys
import gc

file_path = 'thesis/steering/scripts/steer_aurora.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_mode = False
for line in lines:
    if "batch_list = []" in line:
        indent = line[:line.find("batch_list")]
        new_lines.append(indent + "running_sum = None\n")
        new_lines.append(indent + "valid_count = 0\n")
        new_lines.append(indent + "import gc\n")
        continue

    if "batch_list.append((b))" in line or "batch_list.append(b)" in line:
        indent = line[:line.find("batch_list.append")]
        new_lines.append(indent + "if running_sum is None:\n")
        new_lines.append(indent + "    running_sum = tuple(t.clone() if hasattr(t, 'clone') else t for t in b) if isinstance(b, tuple) else b.clone()\n")
        new_lines.append(indent + "else:\n")
        new_lines.append(indent + "    running_sum = tuple(rs + t if hasattr(t, 'clone') else rs for rs, t in zip(running_sum, b)) if isinstance(b, tuple) else running_sum + b\n")
        new_lines.append(indent + "valid_count += 1\n")
        new_lines.append(indent + "try:\n")
        new_lines.append(indent + "    del b\n")
        new_lines.append(indent + "except NameError:\n")
        new_lines.append(indent + "    pass\n")
        new_lines.append(indent + "gc.collect()\n")
        continue

    if 'def average_tuple_of_tensors(' in line:
        skip_mode = True
        # add replacement before skipping everything
        indent = line[:line.find("def")]
        new_lines.append(indent + "if valid_count == 0:\n")
        new_lines.append(indent + "    print('Error: No valid batches loaded.')\n")
        new_lines.append(indent + "    sys.exit(1)\n")
        new_lines.append(indent + "if isinstance(running_sum, tuple):\n")
        new_lines.append(indent + "    batch = tuple(rs / valid_count if hasattr(rs, 'clone') else rs for rs in running_sum)\n")
        new_lines.append(indent + "else:\n")
        new_lines.append(indent + "    batch = running_sum / valid_count\n")
        continue

    if skip_mode:
        if 'base_day_str = "climatology"' in line:
            skip_mode = False
            new_lines.append(line)
        continue

    new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)
