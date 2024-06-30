import os

target_dir = "data/images/Crosswalk"

for filename in os.listdir(target_dir):
    newname = "Crosswalk " + filename.split()[-1]
    os.rename(os.path.join(target_dir, filename), os.path.join(target_dir, newname))
