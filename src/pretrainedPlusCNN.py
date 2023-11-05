"""
In this file we will train a pretrained CNN with the PLUS dataset.
"""
from torch import nn
from torchvision import models

from src.PlusDataLoader import PlusDataLoader
from src.ResnetClassifier import ResnetClassifier
from src.VeinImageType import VeinImageType


def main():
    """
    This gives an example how to use the pretrained CNN with the PLUS dataset. With just the genuine and spoofed data.
    """
    data_loader = PlusDataLoader()
    train_loader, test_loader = data_loader.load_data(batch_size=16,
                                                      use_image_types=[VeinImageType.GENUINE, VeinImageType.SPOOFED])

    model = ResnetClassifier(num_epochs=10,
                             learning_rate=0.001,
                             model_name='resnet18_gen_spoof',
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=1)
    model.train(train_loader, test_loader)
    model.evaluate(test_loader)


if __name__ == '__main__':
    main()
