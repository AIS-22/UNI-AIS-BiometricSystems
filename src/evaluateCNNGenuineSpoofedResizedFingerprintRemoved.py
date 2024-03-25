from torch import nn
from torchvision import models

from VeinImageType import VeinImageType
from classifier.impl.GenuineSpoofedResizedClassifier import SpoofedResizedClassifier
from data_loader.impl.ResizedDataLoader import ResizedDataLoader


def main():
    # options = [evaluation_set, model_set]
    # options = [(eval_types, eval_ds, eval_ds_folder),
    #            (model_types, model_ds, model_ds_folder)]
    options_m = [
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
    options = [
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "PLUS", ''),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '005'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PLUS", '006'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '005'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PLUS", '006'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '005'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PLUS", '006'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '003'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '004'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '005'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PLUS", '006'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "PROTECT", ''),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "PROTECT", '110'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '010'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "PROTECT", '110'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "IDIAP", ''),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "IDIAP", '009'),
        ([VeinImageType.GENUINE, VeinImageType.SPOOFED], "SCUT", ''),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_CYCLE], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DIST], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_DRIT], "SCUT", '008'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '007'),
        ([VeinImageType.GENUINE, VeinImageType.SYNTHETIC_STAR], "SCUT", '008'),
    ]
    gan_removal_options = [
        "bar",
        "mean",
        "peak"
    ]

    for model_types, model_ds, model_ds_folder in options_m:
        # for eval_types, model_trained_types, dataset_name, folder in options:
        print(f"Evaluate on all DS with model from {model_ds} DS")
        if model_ds_folder == '':
            model_name = f'resnet18_resized_{model_ds}_' + '_'.join(e.value for e in model_types)
        else:
            model_name = f'resnet18_resized_{model_ds}_{model_ds_folder}_' + '_'.join(e.value for e in model_types)

        model_name = f'cnnParams_{model_name}.pt'
        print(model_name)

        model = SpoofedResizedClassifier(num_epochs=10,
                                         learning_rate=0.001,
                                         batch_size=16,
                                         folds=5,
                                         model_name=model_name,
                                         dataset_name=model_ds,
                                         model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                         loss_function=nn.CrossEntropyLoss(),
                                         num_image_channels=3,
                                         num_inputs_nodes=(580, 280))

        for (eval_types, eval_ds, eval_ds_folder) in options:

            data_loader = ResizedDataLoader()
            dataset = data_loader.load_data(use_image_types=eval_types, dataset_name=f"{eval_ds}/val",
                                            folder=eval_ds_folder)
            print(f"Dataset: {dataset}")

            model.load_model(f"models/{model_ds}/{model_name}", dataset)

            model.evaluate(dataset)
            model.save_val_accuracy(eval_ds, eval_ds_folder, eval_types, model_ds, model_ds_folder, model_types)
            model.save_val_confusion_matrix(eval_ds, eval_ds_folder, eval_types, model_ds, model_ds_folder, model_types)


if __name__ == '__main__':
    main()
