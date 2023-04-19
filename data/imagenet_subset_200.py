import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# Refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=MEAN, std=STD)

def get_dataset(data_path, train_folder='train', val_folder='val'):
    traindir = os.path.join(data_path, train_folder)
    valdir = os.path.join(data_path, val_folder)
    transforms_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
    transforms_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

    train_set = datasets.ImageFolder(traindir, transforms_train)
    test_set = datasets.ImageFolder(valdir, transforms_val)

    return train_set, test_set