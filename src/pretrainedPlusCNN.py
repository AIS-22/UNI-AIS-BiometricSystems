"""
In this file we will train a pretrained CNN with the PLUS dataset.
"""
from torch import nn
from torchvision import models

from src.VeinImageType import VeinImageType
from src.classifier.impl.ResnetClassifier import ResnetClassifier
from src.data_loader.impl.PlusDataLoader import PlusDataLoader


def main():
    """
    This gives an example how to use the pretrained CNN with the PLUS dataset. With just the genuine and spoofed data.
    """
    data_loader = PlusDataLoader()
    batch_size = 16
    dataset = data_loader.load_data(use_image_types=[VeinImageType.GENUINE, VeinImageType.SPOOFED])

    model = ResnetClassifier(num_epochs=10,
                             learning_rate=0.001,
                             batch_size=batch_size,
                             folds=5,
                             model_name='resnet18_gen_spoof',
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=3)

    model.train(dataset)
    model.save_accuracy()
    model.save_losses()
    model.save_confusion_matrix()


if __name__ == '__main__':
    main()
