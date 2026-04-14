with open('steer_aurora.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_def = False
defs = []

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("def recursive_clone") or line.startswith("def recursive_add") or line.startswith("def recursive_div"):
        in_def = True
        defs.append(line)
        i += 1
        while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t") or lines[i].strip() == ""):
            defs.append(lines[i])
            i += 1
        continue
    new_lines.append(line)
    i += 1

# Find the last import
last_import_idx = 0
for j, line in enumerate(new_lines):
    if line.startswith("import ") or line.startswith("from "):
        last_import_idx = j

final_lines = new_lines[:last_import_idx+1] + ["\n"] + defs + ["\n"] + new_lines[last_import_idx+1:]

# put import gc back
import_gc_idx = -1
for j, line in enumerate(final_lines):
    if "if len(neutral_dates) > 1:" in line:
        final_lines.insert(j+1, "        import gc\n")
        break

with open('steer_aurora.py', 'w') as f:
    f.writelines(final_lines)
