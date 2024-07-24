import os

def rename(category, newname):
    target_dir = f"data/images/{category}"
    for filename in os.listdir(target_dir):
        newfilename = f"{newname} " + filename.split()[-1]
        os.rename(os.path.join(target_dir, filename), os.path.join(target_dir, newfilename))
