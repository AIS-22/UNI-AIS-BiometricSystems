from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.ResnetClassifier import ResnetClassifier
from data_loader.impl.PlusDataLoader import PlusDataLoader


def main():
    # options = [evaluation_types, (     information for model name:     )]
    # options = [evaluation_types, model_train_name , dataset_name, folder]
    options = [
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SPOOFED], "PLUS", ''),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SPOOFED], "PROTECT", ''),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '110'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '110'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SPOOFED], "IDIAP", ''),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SPOOFED], "SCUT", ''),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED],
         [VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '008'),
    ]
    for eval_types, model_trained_types, dataset_name, folder in options:
        print("Evaluate model for " + dataset_name + " with " + folder + " folder and " + str(model_trained_types[1]))
        if folder == '':
            model_name = f'resnet18_{dataset_name}_' + '_'.join(e.value for e in model_trained_types)
        else:
            model_name = f'resnet18_{dataset_name}_{folder}_' + '_'.join(e.value for e in model_trained_types)

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

        dataset = data_loader.load_data(use_image_types=eval_types, dataset_name=dataset_name + '/val', folder=folder)

        model.evaluate(dataset)
        model.save_val_accuracy()
        model.save_val_confusion_matrix()


if __name__ == '__main__':
    main()
