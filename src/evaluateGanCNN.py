from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.GanClassifier import GanClassifier
from data_loader.impl.GanDataLoader import GanDataLoader


def main():
    # options = [evaluation_types, (     information for model name:     )]
    # options = [evaluation_types, model_train_name , dataset_name, folder]
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
    for eval_types, dataset_name in options:
        print(f"Evaluate model for {dataset_name}")

        model_name = f'cnnParams_resnet18_{dataset_name}_ganSeperator.pt'
        model = GanClassifier(num_epochs=10,
                                 learning_rate=0.001,
                                 batch_size=16,
                                 folds=5,
                                 model_name=model_name,
                                 dataset_name=dataset_name,
                                 model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                 loss_function=nn.CrossEntropyLoss(),
                                 num_image_channels=3)

        data_loader = GanDataLoader()
        dataset = data_loader.load_data(use_image_types=eval_types, dataset_name=f'{dataset_name}/val')

        model.load_model(f"models/{dataset_name}/{model_name}", dataset)

        model.evaluate(dataset)
        model.save_val_accuracy()
        model.save_val_confusion_matrix()


if __name__ == '__main__':
    main()
