from typing import List, Any

from torch.utils.data import Subset
from torchvision.transforms import transforms

from src.VeinImageType import VeinImageType
from src.data_loader.AbstractDataLoader import AbstractDataLoader


class VeraDataLoader(AbstractDataLoader):

    def load_data(self, use_image_types: List[VeinImageType], dataset_name, transform=transforms.ToTensor(),
                  folder="") -> Subset[Any]:
        # TODO: Set __train_set and __test_set here!
        return None
