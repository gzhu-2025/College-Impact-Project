import os

def rename(category, newname):
    target_dir = f"data/images/{category}"
    i = 1
    for filename in os.listdir(target_dir):
        newfilename = f"{category}_{i}." + filename.split('.')[-1]
        os.rename(os.path.join(target_dir, filename), os.path.join(target_dir, newfilename))
        i+=1
