from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.GanClassifier import GanClassifierResized
from data_loader.impl.GanDataLoader import GanDataLoaderResized


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
    for _, model_ds in options:
        print(f"Evaluate model for {model_ds}")

        model_name = f'cnnParams_resnet18_resized_{model_ds}_ganSeperator.pt'
        model = GanClassifierResized(num_epochs=10,
                                     learning_rate=0.001,
                                     batch_size=16,
                                     folds=5,
                                     model_name=model_name,
                                     dataset_name=model_ds,
                                     model=models.resnet18(
                                         weights=models.ResNet18_Weights.DEFAULT),
                                     loss_function=nn.CrossEntropyLoss(),
                                     num_image_channels=3)
        for eval_types, eval_ds in options:
            data_loader = GanDataLoaderResized()
            dataset = data_loader.load_data(
                use_image_types=eval_types, dataset_name=f'{eval_ds}/val')

            model.load_model(f"models/{model_ds}/{model_name}", dataset)

            model.evaluate(dataset)
            model.save_val_accuracy(eval_ds=eval_ds, model_ds=model_ds)
            model.save_val_confusion_matrix(eval_ds=eval_ds, model_ds=model_ds)


if __name__ == '__main__':
    main()
