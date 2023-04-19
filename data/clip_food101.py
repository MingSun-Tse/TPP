from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image



#@qw: use clip preprocess
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_dataset(data_path):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation = InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size=224, interpolation = InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_set = Food101(root = data_path,
                        split = 'train',
                        download = True,
                        transform = transform_train)
    test_set = Food101(root = data_path,
                       split = 'test',
                       download = True,
                       transform = transform_test)

    return train_set, test_set
