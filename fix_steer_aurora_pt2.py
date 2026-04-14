import re

with open('thesis/steering/scripts/steer_aurora.py', 'r') as f:
    code = f.read()

# Fix order of del b and torch.save(b)
code = code.replace("""                try:
                    del b
                except NameError:
                    pass
                gc.collect()
                
                # Optionally save it locally so you don't have to re-extract it later
                try:
                    torch.save(b, download_dir / f"batch_{day_str}.pt")
                except:
                    pass""", """                # Optionally save it locally so you don't have to re-extract it later
                try:
                    torch.save(b, download_dir / f"batch_{day_str}.pt")
                except:
                    pass

                try:
                    del b
                except NameError:
                    pass
                gc.collect()""")

with open('thesis/steering/scripts/steer_aurora.py', 'w') as f:
    f.write(code)
