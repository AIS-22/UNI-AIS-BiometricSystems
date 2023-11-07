from typing import List

import torchvision

from torch.utils.data import Subset, random_split
import random
from src.VeinImageType import VeinImageType
from src.data_loader.AbstractDataLoader import AbstractDataLoader


def checkPath(full_path, image_path):
    if 'synthethic' in full_path:
        return full_path in image_path and '/003/' in image_path and '/004/' and ('/5_rs/' not in image_path or '/all_rs/' not in image_path or '/evaluation/' not in image_path)
    else:
        return full_path in image_path


class PlusDataLoader(AbstractDataLoader):

    def __init__(self):
        super().__init__()

    def _create_dataset(self, transform, use_image_types: List[VeinImageType]) -> None:

        root_path = 'data/PLUS/'
        # Load the entire dataset
        full_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)

        indexes_list = []
        for i, path in enumerate(use_image_types):
            full_path = root_path + path.value + '/'
            indexes = [i for i, (image_path, _) in enumerate(full_dataset.samples) if checkPath(full_path, image_path)]
            indexes_list.append(indexes)

        # get the smallest amount of images in a class
        amount = len(min(indexes_list, key=len))

        # Calculate counts for train and test
        train_count = int(0.8 * amount)

        indexes_train = []
        indexes_test = []
        for i in range(0, len(indexes_list)):
            # shorter each class to the smallest amount
            indexes_list[i] = random.sample(indexes_list[i], amount)

            # Shuffle indexes
            random.shuffle(indexes_list[i])

            # Combine all indexes
            indexes_train += indexes_list[i][:train_count]
            indexes_test += indexes_list[i][train_count:]

        # Shuffle train and test
        random.shuffle(indexes_train)
        random.shuffle(indexes_test)

        # train test split
        train_dataset = Subset(full_dataset, indexes_train)
        test_dataset = Subset(full_dataset, indexes_test)
        self.set_train_set(train_dataset)
        self.set_test_set(test_dataset)
