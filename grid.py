import numpy as np
import sys, os

label_dir = f"data/labels/{sys.argv[1]}"

grids = []
for filename in os.listdir(label_dir):
    path = os.path.join(label_dir, filename)
    
    print(filename)
    grid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    with open(path, 'r') as file:
        

        #['1', '0.522998', '0.492760', '0.470187', '0.435264']
        #['class', x-coord of center, y-coord of center, width, height]
        for line in file.readlines():
            coords = line.split()

            #numbering corners of box cw starting with top left
            #one = {'x': float(coords[1]) - float(coords[3])/2, 'y': float(coords[2]) - float(coords[4])/2}
            #two = {'x': float(coords[1]) + float(coords[3])/2, 'y': float(coords[2]) - float(coords[4])/2}
            #three = {'x': float(coords[1]) + float(coords[3])/2, 'y': float(coords[2]) + float(coords[4])/2}
            #four = {'x': float(coords[1]) - float(coords[3])/2, 'y': float(coords[2]) + float(coords[4])/2}

            label, x_center, y_center, width, height = coords
            top_row = int((float(y_center) - float(height)/2) / 0.25)
            bot_row = int((float(y_center) + float(height)/2) / 0.25)
            first_col = int((float(x_center) - float(width)/2) / 0.25)
            last_col = int((float(x_center) + float(width)/2) / 0.25)

            for j in range(16):
                row = int(j / 4)
                col = j % 4
                if (
                    row <= bot_row 
                    and row >= top_row 
                    and col <= last_col 
                    and col >= first_col
                ):
                    grid[j] = 1 
        
        for i in range(4):
            for j in range(4):
                print(grid[i * 4 + j], end=" ")
            print()

        file_number = int(filename.split("(")[-1].split(")")[0])
            
        grids.append([file_number] + grid)
        
np_grids = np.array(grids)

np.save(f"{sys.argv[1]}.npy", np_grids)

