import random
from typing import List, Any

import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms

from src.VeinImageType import VeinImageType
from src.data_loader.AbstractDataLoader import AbstractDataLoader

class PlusDataLoader(AbstractDataLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, use_image_types: List[VeinImageType], datasetName, transform=transforms.ToTensor(), folder="") -> Subset[Any]:

        root_path = 'data/' + datasetName + '/'
        # Load the entire dataset
        full_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        indexes_list = []
        for i, path in enumerate(use_image_types):
            full_path = root_path + path.value + "_" + folder + '/' if "synthethic" in path.value else root_path + path.value + '/'
            indexes = [i for i, (image_path, _) in enumerate(full_dataset.samples) if full_path in image_path]
            indexes_list.append(indexes)

        # get the smallest amount of images in a class
        amount = len(min(indexes_list, key=len))

        all_indexes = []
        for i in range(0, len(indexes_list)):
            # shorter each class to the smallest amount
            indexes_list[i] = random.sample(indexes_list[i], amount)

            # Shuffle indexes
            random.shuffle(indexes_list[i])

            # Combine all indexes
            all_indexes += indexes_list[i]

        # Shuffle indexes
        random.shuffle(all_indexes)

        # Get filtered dataset
        dataset = Subset(full_dataset, all_indexes)
        return dataset
