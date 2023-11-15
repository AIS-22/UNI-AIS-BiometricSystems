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
    image_typ = [VeinImageType.GENUINE, VeinImageType.SPOOFED]
    datasetName = 'PLUS'
    folder = ''
    if folder == '':
        modelName = 'resnet18_' + datasetName + '_' + '_'.join(e.value for e in image_typ)
    else:
        modelName = 'resnet18_' + datasetName + '_' + folder + '_' + '_'.join(e.value for e in image_typ)

    dataset = data_loader.load_data(use_image_types=image_typ, datasetName=datasetName, folder=folder)
    model = ResnetClassifier(num_epochs=10,
                             learning_rate=0.001,
                             batch_size=16,
                             folds=5,
                             model_name=modelName,
                             dataset_name=datasetName,
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=3)

    model.train(dataset)
    model.save_model()
    model.save_accuracy()
    model.save_losses()
    model.save_confusion_matrix()


if __name__ == '__main__':
    main()
