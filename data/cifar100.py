from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

def get_dataset(data_path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD), # Refer to EigenDamage [Wang et al., 2019, ICML] code
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    train_set = CIFAR100(data_path,
                         train=True,
                         download=True,
                         transform=transform_train)
    test_set = CIFAR100(data_path,
                        train=False,
                        download=True,
                        transform=transform_test)

    return train_set, test_set
