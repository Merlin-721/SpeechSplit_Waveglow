from pathlib import Path
import random

ROOT_DIR = ""

wavpaths = list(Path(ROOT_DIR).glob(f"**/*.wav"))
print(len(wavpaths))
random.shuffle(wavpaths)
train_prop = int(len(wavpaths) * 0.9)

with open(f"Waveglow/data/train_files.txt","w") as f:
  for item in wavpaths[:train_prop]:
    f.write('%s\n' % item)

with open(f"Waveglow/data/test_files.txt","w") as f:
  for item in wavpaths[train_prop:]:
    f.write('%s\n' % item)