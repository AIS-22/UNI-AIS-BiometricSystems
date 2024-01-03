import random
from typing import List, Any

import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms

from VeinImageType import VeinImageType
from data_loader.AbstractDataLoader import AbstractDataLoader

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.samples = [dataset.samples[i] for i in indices]
        self.transform = dataset.transform
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)

class GanDataLoader(AbstractDataLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, use_image_types: List[VeinImageType], dataset_name,
                  transform=transforms.ToTensor()) -> Subset[Any]:

        root_path = 'data/' + dataset_name + '/'
        # Load the entire dataset
        full_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        sub_folders = full_dataset.classes
        sub_folders.remove('genuine')
        sub_folders.remove('spoofed')
        indexes_list = []
        for sub_folder in sub_folders:
            full_path = root_path + sub_folder + '/'
            indexes = [i for i, (image_path, _) in enumerate(full_dataset.samples) if full_path in image_path]
            indexes_list.append(indexes)

        # get the smallest amount of images in a class
        amount = len(min(indexes_list, key=len))

        all_indexes = []
        for i in range(0, len(indexes_list)):
            # reduce each class to the smallest amount of images
            indexes_list[i] = random.sample(indexes_list[i], amount)

            # Combine all indexes
            all_indexes += indexes_list[i]

        # Shuffle indexes
        random.shuffle(all_indexes)

        # Get filtered dataset
        dataset = CustomDataset(full_dataset, all_indexes)
        print(dataset.classes)
        return dataset