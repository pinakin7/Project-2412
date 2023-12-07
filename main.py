import torchvision.transforms

from PrivateModels.DPGAN.Generator import  Generator
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import  make_grid
net = Generator(1).cuda()
net.load_state_dict(torch.load("A:\Project\\2412\\models\\dpcgan\\Generator.pth"))

noise = torch.autograd.Variable(torch.randn(100, 100)).cuda()
labels = torch.autograd.Variable(torch.LongTensor(np.random.randint(0, 10, 100))).cuda()
out = 0.3081*net(noise, labels).detach().cpu()+0.1307


grid = make_grid(out.unsqueeze(1), nrow=10, padding=2, pad_value=0)
img = torchvision.transforms.ToPILImage()(grid)
torchvision.utils.save_image(grid, "out.png")