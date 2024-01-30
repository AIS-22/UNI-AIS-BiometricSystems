from abc import ABC, abstractmethod
from typing import List, Any

from torch.utils.data import Subset
from torchvision.transforms import transforms

from VeinImageType import VeinImageType


class AbstractDataLoader(ABC):

    @abstractmethod
    def load_data(self, use_image_types: List[VeinImageType], dataset_name, transform=transforms.ToTensor(),
                  folder="") -> Subset[Any]:
        pass
