import re

with open('thesis/steering/scripts/steer_aurora.py', 'r') as f:
    code = f.read()

from_code = """import gc"""

to_code = """import gc

def recursive_clone(obj):
    if hasattr(obj, 'clone'):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: recursive_clone(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(recursive_clone(x) for x in obj)
    elif isinstance(obj, list):
        return [recursive_clone(x) for x in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        from copy import copy
        new_obj = copy(obj)
        for field_name in obj.__dataclass_fields__:
            setattr(new_obj, field_name, recursive_clone(getattr(obj, field_name)))
        return new_obj
    else:
        return obj

def recursive_add(obj1, obj2):
    if hasattr(obj1, 'clone'):
        return obj1 + obj2
    elif isinstance(obj1, dict):
        return {k: recursive_add(obj1[k], obj2[k]) for k in obj1.keys()}
    elif isinstance(obj1, tuple):
        return tuple(recursive_add(x, y) for x, y in zip(obj1, obj2))
    elif isinstance(obj1, list):
        return [recursive_add(x, y) for x, y in zip(obj1, obj2)]
    elif hasattr(obj1, '__dataclass_fields__'):
        from copy import copy
        new_obj = copy(obj1)
        for field_name in obj1.__dataclass_fields__:
            setattr(new_obj, field_name, recursive_add(getattr(obj1, field_name), getattr(obj2, field_name)))
        return new_obj
    else:
        return obj1

def recursive_div(obj, div):
    if hasattr(obj, 'clone'):
        return obj / div
    elif isinstance(obj, dict):
        return {k: recursive_div(v, div) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(recursive_div(x, div) for x in obj)
    elif isinstance(obj, list):
        return [recursive_div(x, div) for x in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        from copy import copy
        new_obj = copy(obj)
        for field_name in obj.__dataclass_fields__:
            setattr(new_obj, field_name, recursive_div(getattr(obj, field_name), div))
        return new_obj
    else:
        return obj
"""

code = code.replace(from_code, to_code)

code = re.sub(
    r'if running_sum is None:\s*running_sum = tuple\(t\.clone\(\) if hasattr\(t, \'clone\'\) else t for t in b\) if isinstance\([a-zA-Z], tuple\) else b\.clone\(\)\s*else:\s*running_sum = tuple\(rs \+ t if hasattr\(t, \'clone\'\) else rs for rs, t in zip\(running_sum, b\)\) if isinstance\([a-zA-Z], tuple\) else running_sum \+ b',
    r'if running_sum is None:\n                            running_sum = recursive_clone(b)\n                        else:\n                            running_sum = recursive_add(running_sum, b)',
    code
)

code = code.replace("if isinstance(running_sum, tuple):\n            batch = tuple(rs / valid_count if hasattr(rs, 'clone') else rs for rs in running_sum)\n        else:\n            batch = running_sum / valid_count", "batch = recursive_div(running_sum, valid_count)")

with open('thesis/steering/scripts/steer_aurora.py', 'w') as f:
    f.write(code)
