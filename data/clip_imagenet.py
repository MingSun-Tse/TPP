import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode




MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)

normalize = transforms.Normalize(mean=MEAN, std=STD)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

#@qw: transforms refer to https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py
def get_dataset(data_path, train_folder='train', val_folder='val'):
    traindir = os.path.join(data_path, train_folder)
    valdir = os.path.join(data_path, val_folder)
    transforms_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation = InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize])
    transforms_val = transforms.Compose([
                transforms.Resize(size=224, interpolation = InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize])

    train_set = datasets.ImageFolder(traindir, transforms_train)
    test_set = datasets.ImageFolder(valdir, transforms_val)

    return train_set, test_set