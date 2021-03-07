import pause
from datetime import datetime
import torch 
import numpy as np 
# pause.until(datetime(2010, 12, 30, 22, 10, 59))
# with open("testcmd.txt", "a") as f:
#     f.write("hahaha")
# f.close
count_list = [89, 34, 900, 30, 1, 3]
count_dict = {k:v for k, v in enumerate(count_list)}
arr = np.array(count_list)
sort_list = torch.tensor(np.argsort(arr))
inputs = torch.rand(4, 3)
print(inputs)
l = [2, 3, 1, 5]
sorted_l = sorted(l, key=count_dict.get, reverse=True)
print(sorted_l)
l = torch.tensor(l)
sorted_l = torch.tensor(sorted_l)
index = l.argsort()[sorted_l.argsort().argsort()]
print(inputs[index,:])


