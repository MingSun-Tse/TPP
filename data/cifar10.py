from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


# Refer to official PyTorch ImageNet example
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_dataset(data_path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
        # Refer to: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_set = CIFAR10(data_path,
                        train=True,
                        download=True,
                        transform=transform_train)
    test_set = CIFAR10(data_path,
                       train=False,
                       download=True,
                       transform=transform_test)

    return train_set, test_set
