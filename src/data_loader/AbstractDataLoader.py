from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import DataLoader
from torchvision import transforms

from src.VeinImageType import VeinImageType


class AbstractDataLoader(ABC):
    def __int__(self):
        self.__train_set = None
        self.__test_set = None

    @abstractmethod
    def _create_dataset(self, transform, use_image_types: List[VeinImageType]) -> None:
        pass

    def load_data(self,
                  batch_size: int,
                  use_image_types: List[VeinImageType],
                  transform=transforms.ToTensor) -> tuple[DataLoader, DataLoader]:
        self._create_dataset(transform, use_image_types)
        assert self.__train_set is not None and self.__test_set is not None

        train_loader = DataLoader(self.__train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.__test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
