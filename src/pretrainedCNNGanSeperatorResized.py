from torch import nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image

from VeinImageType import VeinImageType
from classifier.impl.GanClassifier import GanClassifierResized
from data_loader.impl.GanDataLoader import GanDataLoaderResized


def main():
    data_loader = GanDataLoaderResized()

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
        print(f"Training GanSeperatorResized model for {dataset_name}")
        model_name = f'resnet18_resized_{dataset_name}_ganSeperator'

        preprocess = transforms.Compose([
            # transforms.Resize(256, interpolation=Image.LANCZOS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = data_loader.load_data(
            use_image_types=model_trained_types, dataset_name=dataset_name + '/train', transform=preprocess)
        model = GanClassifierResized(num_epochs=5,
                                     learning_rate=0.00001,
                                     batch_size=16,
                                     folds=3,
                                     model_name=model_name,
                                     dataset_name=dataset_name,
                                     model=models.resnet18(
                                         weights=models.ResNet18_Weights.DEFAULT),
                                     loss_function=nn.CrossEntropyLoss(),
                                     num_image_channels=3)

        model.train(dataset)
        model.save_losses()
        model.save_model()


if __name__ == '__main__':
    main()
