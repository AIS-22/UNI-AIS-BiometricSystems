from abc import ABC, abstractmethod
from typing import List, Any

from torch.utils.data import Subset
from torchvision.transforms import transforms

from src.VeinImageType import VeinImageType


class AbstractDataLoader(ABC):

    @abstractmethod
    def load_data(self, use_image_types: List[VeinImageType], transform=transforms.ToTensor()) -> Subset[Any]:
        pass
