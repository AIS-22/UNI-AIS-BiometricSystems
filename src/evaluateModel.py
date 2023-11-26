import torch
from torch import nn
from torchvision import models

from src.VeinImageType import VeinImageType
from src.data_loader.impl.PlusDataLoader import PlusDataLoader
from src.classifier.impl.ResnetClassifier import ResnetClassifier


def main():
    datasetName = 'PLUS'
    # "models/"+self.dataset_name +"/cnnParams_" + self.model_name + ".pt"
    image_typ = [VeinImageType.GENUINE, VeinImageType.SPOOFED]
    datasetName = 'PLUS'
    folder = ''
    if folder == '':
        modelName = 'resnet18_' + datasetName + '_' + '_'.join(e.value for e in image_typ)
    else:
        modelName = 'resnet18_' + datasetName + '_' + folder + '_' + '_'.join(e.value for e in image_typ)

    modelName = 'cnnParams_' + modelName + ".pt"
    model = ResnetClassifier(num_epochs=2,
                             learning_rate=0.001,
                             batch_size=16,
                             folds=2,
                             model_name=modelName,
                             dataset_name=datasetName,
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=3)

    model.load_model(modelName)

    data_loader = PlusDataLoader()
    dataset = data_loader.load_data(use_image_types=image_typ, datasetName=datasetName + '/val', folder=folder)

    model.evaluate(dataset)
    model.save_accuracy()
    model.save_confusion_matrix()


if __name__ == '__main__':
    main()
