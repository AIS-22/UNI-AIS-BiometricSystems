import random
from typing import List, Any

import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms

from VeinImageType import VeinImageType
from data_loader.AbstractDataLoader import AbstractDataLoader

from CustomDataset import CustomDataset


class ResizedDataLoader(AbstractDataLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, use_image_types: List[VeinImageType], dataset_name,
                  transform=transforms.ToTensor(), folder='') -> Subset[Any]:

        root_path = '../data_rs/' + dataset_name + '/'
        # Load the entire dataset
        full_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        indexes_list = []
        for i, path in enumerate(use_image_types):
            folder_part = ("_" + folder + '/') if "synthethic" in path.value else '/'
            full_path = root_path + path.value + folder_part
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
        return dataset
