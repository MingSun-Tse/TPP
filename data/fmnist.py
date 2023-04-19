from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


MEAN = (0.1307,)
STD  = (0.3081,)

def get_dataset(data_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = FashionMNIST(data_path,
                      train=True,
                      download=True,
                      transform=transform)
    test_set = FashionMNIST(data_path,
                     train=False,
                     download=True,
                     transform=transform)

    return train_set, test_set
