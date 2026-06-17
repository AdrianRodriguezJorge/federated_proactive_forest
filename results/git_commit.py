import subprocess

print("Staging changes...")
subprocess.run(["git", "add", "-A"], check=True)

print("Committing...")
subprocess.run(["git", "commit", "-m", "Refactor benchmark and clean workspace, update datasets and README"], check=True)

print("Pushing...")
subprocess.run(["git", "push", "origin", "s1_s8"], check=True)

print("Git sync complete!")
