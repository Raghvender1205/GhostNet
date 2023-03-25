import torch
import time
from ghostnet import ghostnet

model = ghostnet().cuda()
x = torch.randn(32, 3, 224, 224)
t = time.time()
for _ in range(30):
    inputs = x.cuda()
    outputs = model(inputs)
    print(outputs)    
print("Total Time Taken: ", time.time() - t)