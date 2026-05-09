import os
path = 'configs/datasets/iris.yaml'
if os.path.exists(path):
    os.remove(path)
    print(f"Deleted {path}")
else:
    print(f"{path} not found")
