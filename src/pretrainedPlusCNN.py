from torch import nn
from torchvision import models

from src.VeinImageType import VeinImageType
from src.classifier.impl.ResnetClassifier import ResnetClassifier
from src.data_loader.impl.PlusDataLoader import PlusDataLoader


def main():
    data_loader = PlusDataLoader()
    model_trained_types = [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE]
    dataset_name = 'PLUS'
    folder = '004'
    if folder == '':
        model_name = 'resnet18_' + dataset_name + '_' + '_'.join(e.value for e in model_trained_types)
    else:
        model_name = 'resnet18_' + dataset_name + '_' + folder + '_' + '_'.join(e.value for e in model_trained_types)

    dataset = data_loader.load_data(use_image_types=model_trained_types, dataset_name=dataset_name + '/train',
                                    folder=folder)
    model = ResnetClassifier(num_epochs=10,
                             learning_rate=0.001,
                             batch_size=16,
                             folds=5,
                             model_name=model_name,
                             dataset_name=dataset_name,
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=3)

    model.train(dataset)
    model.save_losses()
    model.save_model()


if __name__ == '__main__':
    main()
