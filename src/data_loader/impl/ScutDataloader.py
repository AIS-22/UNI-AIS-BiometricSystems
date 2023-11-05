from typing import List

from src.data_loader.AbstractDataLoader import AbstractDataLoader
from src.VeinImageType import VeinImageType


class ScutDataLoader(AbstractDataLoader):

    def __create_dataset(self, transform, use_image_types: List[VeinImageType]) -> None:
        # TODO: Set __train_set and __test_set here!
        return None
