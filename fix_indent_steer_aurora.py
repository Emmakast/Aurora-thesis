import re

with open('thesis/steering/scripts/steer_aurora.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "if running_sum is None:" in line:
        base_indent = len(line) - len(line.lstrip())
        indent_str = " " * base_indent
        
        # fix the next lines
        if "running_sum = recursive_clone(b)\n" in lines[i+1]:
            lines[i+1] = indent_str + "    running_sum = recursive_clone(b)\n"
        if "else:\n" in lines[i+2]:
            lines[i+2] = indent_str + "else:\n"
        if "running_sum = recursive_add(running_sum, b)\n" in lines[i+3]:
            lines[i+3] = indent_str + "    running_sum = recursive_add(running_sum, b)\n"

with open('thesis/steering/scripts/steer_aurora.py', 'w') as f:
    f.writelines(lines)
