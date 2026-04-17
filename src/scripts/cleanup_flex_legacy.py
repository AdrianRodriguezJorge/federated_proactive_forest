import os
files = [
    'src/infrastructure/flex/flex_collect_trees_pf.py',
    'src/infrastructure/flex/flex_update_client_pf.py'
]
for f in files:
    if os.path.isfile(f):
        try:
            os.remove(f)
            print(f'Deleted {f}')
        except Exception as e:
            print(f'Error deleting {f}: {e}')
    else:
        print(f'File not found (already removed): {f}')
