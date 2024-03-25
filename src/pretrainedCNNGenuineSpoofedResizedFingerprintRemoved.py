from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.GenuineSpoofedResizedClassifier import SpoofedResizedClassifier
from data_loader.impl.ResizedDataLoader import ResizedDataLoader


def main():
    data_loader = ResizedDataLoader()

    # options = [model_name, dataset_name, folder]
    options = [
        # ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "PLUS", ''),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '003'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '005'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '006'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '003'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '004'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '005'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '006'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '003'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '005'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '006'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '003'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '004'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '005'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '006'),
        # ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "PROTECT", ''),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '110'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PROTECT", '010'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PROTECT", '010'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '010'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '110'),
        # ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "IDIAP", ''),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "IDIAP", '009'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "IDIAP", '009'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "IDIAP", '009'),
        # ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "SCUT", ''),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '007'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '008'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '007'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '007'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '008'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '007'),
        # ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '008'),
    ]
    gan_removal_options = [
        "bar",
        "mean",
        "peak"
    ]

    for model_trained_types, dataset_name, folder in options:
        print("Training model for " + dataset_name + " with " + folder + " folder and " + str(model_trained_types[1]))
        if folder == '':
            model_name = f'resnet18_resized_{dataset_name}_' + '_'.join(e.value for e in model_trained_types)
        else:
            model_name = f'resnet18_resized_{dataset_name}_{folder}_' + '_'.join(e.value for e in model_trained_types)
        for gan_removal_option in gan_removal_options:
            print("Training model for " + dataset_name + " with " + folder +
                  " folder with " + gan_removal_option + " gan removal")
            dataset = data_loader.load_data(use_image_types=model_trained_types, dataset_name=dataset_name + '/train_' + gan_removal_option,
                                            folder=folder)
            model = SpoofedResizedClassifier(num_epochs=10,
                                             learning_rate=0.001,
                                             batch_size=16,
                                             folds=5,
                                             model_name=model_name + '_' + gan_removal_option,
                                             dataset_name=dataset_name,
                                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                             loss_function=nn.CrossEntropyLoss(),
                                             num_image_channels=3)

            model.train(dataset)
            model.save_losses()
            model.save_model()


if __name__ == '__main__':
    main()
