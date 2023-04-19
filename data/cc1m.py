import os
import io
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def pil_loader(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

# @qw: refer to: https://github.com/facebookresearch/OTTER/blob/main/data/dataset_factory.py
class ConceptualCaptions(Dataset):
    def __init__(self, root_dir, transform=None, aug=None):
        self.root_dir = root_dir

        label_dir = os.path.join(self.root_dir, "processed_labels.csv")
        self.data_df = pd.read_csv(label_dir)

        self.transform = transform
        self.aug = aug

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        j, filename, caption = self.data_df.iloc[idx]
        image_path = os.path.join(self.root_dir, filename)

        sample = pil_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.aug is not None:
            caption = self.aug.augment(caption)
        return sample, caption

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)

normalize = transforms.Normalize(mean=MEAN, std=STD)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

#@qw: refer to https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py
def get_dataset(data_path):
    
    transforms_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation = InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize])

    train_set = ConceptualCaptions(root_dir = data_path, transform = transforms_train)


    return train_set