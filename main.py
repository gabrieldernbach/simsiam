import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.01, type=float)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        return self.bottleneck(x)


class SimDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x = self.X[item]
        a = self.transform(x)
        b = self.transform(x)
        return a, b


def sample_mnist_subset(size=10):
    train = datasets.MNIST(root='./data', train=True,
                           download=True, transform=None)
    X, Y = train.data, train.targets

    idxs = []
    for i in Y.unique():
        idx = torch.nonzero(Y == i)
        idxs.append(idx[torch.randperm(len(idx))[:size]])
    idxs = torch.cat(idxs)
    X, Y = X[idxs], Y[idxs]
    return X, Y


def main():
    args = parser.parse_args()

    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomResizedCrop(28, scale=(0.4, 1.0), ratio=(0.2, 2)),
        transforms.RandomPerspective(distortion_scale=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    X, Y = sample_mnist_subset(500)
    ds = SimDataset(X, tfs)
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=True)

    mlp = MLP()
    cnn = CNN()

    params = list(mlp.parameters()) + list(cnn.parameters())
    optimizer = torch.optim.SGD(params=params, lr=args.lr)

    def adjust_learning_rate(optimizer, epoch, args):
        lr = args.lr
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if args.resume:
        ckpt = torch.load("cnn_ckpt.pth")
        cnn.load_state_dict(ckpt)
        ckpt = torch.load("mlp_ckpt.pth")
        mlp.load_state_dict(ckpt)

    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss_avg = 0
        for idx, (x1, x2) in enumerate(dl, start=1):
            z1, z2 = cnn(x1), cnn(x2)
            p1, p2 = mlp(z1), mlp(z2)

            loss = D(p1, z2) / 2 + D(p2, z1) / 2
            loss_avg += (loss.item() - loss_avg) / idx
            print(f"{epoch}, loss {loss.item():.18f}, avg {loss_avg:.18f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"=====> {loss_avg}")
        torch.save(cnn.state_dict(), "cnn_ckpt.pth")
        torch.save(mlp.state_dict(), "mlp_ckpt.pth")


if __name__ == "__main__":
    main()
