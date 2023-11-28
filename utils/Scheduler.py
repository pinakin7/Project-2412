import torch
import math


class LRScheduler(object):
    def __init__(self, optimizer, step_size=10, x_min=0.1, x_max=0.1, h_min=0.1, f_max=2.0, ideal_loss = math.log(4,10), min_lr = 1e-6):
        self.optimizer = optimizer
        self.ideal_loss = ideal_loss
        self.x_min = x_min
        self.x_max = x_max
        self.h_min = h_min
        self.f_max = f_max
        self.min_lr = 1e-6
        self.step_size=step_size

    def step(self):
        loss = self.optimizer.param_groups[0]['loss']
        base_lr = self.optimizer.param_groups[0]['lr']

        x = torch.abs(loss - self.ideal_loss)
        f_x = torch.clamp(torch.pow(self.f_max, x/self.x_max), min=1.0, max=self.f_max)
        h_x = torch.clamp(torch.pow(self.h_min, x / self.x_min), min=self.h_min, max=1.0)

        multiplier = f_x if loss>= self.ideal_loss else h_x

        # print(multiplier)
        # print(loss)
        new_lr = base_lr * multiplier
        print(new_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr if new_lr > self.min_lr else self.min_lr



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    model = torchvision.models.resnet18(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = LRScheduler(optimizer)
    lr_list = []
    for epoch in range(35):
        for i in range(1000):
            optimizer.param_groups[0]['loss'] = torch.randn(())
            optimizer.step()
            scheduler.step()
            lr_list.append(optimizer.param_groups[0]['lr'])
            # print(optimizer.param_groups[0]['lr'])
    plt.figure(figsize=(24, 8))
    plt.plot(lr_list)
    plt.show()