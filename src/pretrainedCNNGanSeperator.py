import torch
from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.GanClassifier import GanClassifier
from data_loader.impl.GanDataLoader import GanDataLoader

import random
random.seed(42)
torch.manual_seed(42)


def main():
    data_loader = GanDataLoader()

    # options = [model_name, dataset_name]
    options = [
        ([VeinImageType.SYNTHETIC_CYCLE,
          VeinImageType.SYNTHETIC_DIST,
          VeinImageType.SYNTHETIC_DRIT,
          VeinImageType.SYNTHETIC_STAR], "PLUS"),
        ([VeinImageType.SYNTHETIC_CYCLE,
          VeinImageType.SYNTHETIC_DIST,
          VeinImageType.SYNTHETIC_DRIT,
          VeinImageType.SYNTHETIC_STAR], "PROTECT"),
        ([VeinImageType.SYNTHETIC_CYCLE,
          VeinImageType.SYNTHETIC_DIST,
          VeinImageType.SYNTHETIC_DRIT,
          VeinImageType.SYNTHETIC_STAR], "IDIAP"),
        ([VeinImageType.SYNTHETIC_CYCLE,
          VeinImageType.SYNTHETIC_DIST,
          VeinImageType.SYNTHETIC_DRIT,
          VeinImageType.SYNTHETIC_STAR], "SCUT")
    ]

    for model_trained_types, dataset_name in options:
        print(f"Training model for {dataset_name} with folder and {str(model_trained_types[1])}")
        model_name = f'resnet18_{dataset_name}_ganSeperator'

        dataset = data_loader.load_data(use_image_types=model_trained_types, dataset_name=dataset_name + '/train')
        print(len(dataset))
        model = GanClassifier(num_epochs=5,
                              learning_rate=0.0001,
                              batch_size=16,
                              folds=3,
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
