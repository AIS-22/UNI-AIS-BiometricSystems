from torch import nn
from torchvision import models

from src.VeinImageType import VeinImageType
from src.classifier.impl.ResnetClassifier import ResnetClassifier
from src.data_loader.impl.PlusDataLoader import PlusDataLoader


def main():
    model_trained_types = [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE]
    model_eval_types = [VeinImageType.GENUINE, VeinImageType.SPOOFED]
    dataset_name = 'PLUS'
    folder = '003'
    if folder == '':
        model_name = 'resnet18_' + dataset_name + '_' + '_'.join(e.value for e in model_trained_types)
    else:
        model_name = 'resnet18_' + dataset_name + '_' + folder + '_' + '_'.join(e.value for e in model_trained_types)

    model_name = 'cnnParams_' + model_name + ".pt"
    model = ResnetClassifier(num_epochs=10,
                             learning_rate=0.001,
                             batch_size=16,
                             folds=5,
                             model_name=model_name,
                             dataset_name=dataset_name,
                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                             loss_function=nn.CrossEntropyLoss(),
                             num_image_channels=3)

    model.load_model("models/" + dataset_name + "/" + model_name)
    data_loader = PlusDataLoader()

    dataset = data_loader.load_data(use_image_types=model_eval_types, dataset_name=dataset_name + '/val', folder=folder)

    model.evaluate(dataset)
    model.save_val_accuracy()
    model.save_val_confusion_matrix()


if __name__ == '__main__':
    main()
