from typing import List
from src.data_loader.AbstractDataLoader import AbstractDataLoader
from src.VeinImageType import VeinImageType
import torchvision
import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms

class PlusDataLoader(AbstractDataLoader):

    def create_dataset(self, transform, use_image_types: List[VeinImageType]) -> None:
        # create a training dataset from genuine images
        #path = 'data/PLUS/' + use_image_types[0].value + '/'
        #path = 'data/PLUS/' 
        path ='/home/radovic/Documents/master_AIS/3_semester/biometric_system/PS/data/PLUS/'
        # self.__train_set = torchvision.datasets.ImageFolder(root=path, transform=transform)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add other transformations if needed
        ])
        a = torchvision.datasets.ImageFolder(root=path, transform=transform)

        # construct the full dataset
        dataset = torchvision.datasets.ImageFolder(path)
        b = dataset.imgs[0]
        c = dataset.class_to_idx
        # select the indices of all other folders
        idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx['genuine']]
        # build the appropriate subset
        subset = Subset(dataset, idx)
        
        # create a test dataset from spoofed images
        self.__test_set = torchvision.datasets.ImageFolder(root='data/PLUS/' + use_image_types[1].value, transform=transform)
        return None 

if __name__ == '__main__':
    data_loader = PlusDataLoader()
    train_loader, test_loader = data_loader.load_data(batch_size=16,
                                                      use_image_types=[VeinImageType.GENUINE, VeinImageType.SPOOFED])