import numpy as np
import os

data_dir = "data"
some_wrong = [False] * 20
some_right = [False] * 20
for file in os.listdir(data_dir):
    img_name = os.path.join(data_dir, file)
    data = np.load(img_name)

    if data["eef_pos"].shape[0] == 8:
        some_right[int(file[16:18])-1] = True
    else:
        some_wrong[int(file[16:18])-1] = True

for i in range(20):
    if some_wrong[i] and some_right[i]:
        print(i, "Mixed")
    elif some_wrong[i]:
        print(i, "All wrong")
    elif some_right[i]:
        print(i, "All correct")
    else:
        print(i, "This shouldn't happen")