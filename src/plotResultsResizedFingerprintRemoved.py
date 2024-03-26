import seaborn as sn
import matplotlib.pyplot as plt
from data_loader.impl.ResizedDataLoader import ResizedDataLoader
from classifier.impl.GenuineSpoofedResizedClassifier import SpoofedResizedClassifier
from VeinImageType import VeinImageType
from torchvision import models
from torch import nn
import os

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
options_gan_removal = [
    "bar",
    "mean",
    "peak"
]


def plotTrainWithoutEvalWithRemoval():

    for model_types, model_ds, model_ds_folder in options:
        for eval_types, eval_ds, eval_ds_folder in options:
            # create a subplots
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # for gan_rem_train in options:
            model = SpoofedResizedClassifier(num_epochs=10,
                                             learning_rate=0.001,
                                             batch_size=16,
                                             folds=5,
                                             model_name="",
                                             dataset_name="",
                                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                             loss_function=nn.CrossEntropyLoss(),
                                             num_image_channels=3,
                                             num_inputs_nodes=(580, 280))
            if model_ds_folder == '':
                model_name = f'{model_ds}_' + '_'.join(e.value for e in model_types)
            else:
                model_name = f'{model_ds}_{model_ds_folder}_' + '_'.join(e.value for e in model_types)
            data_loader = ResizedDataLoader()
            for gan_removal_eval in options_gan_removal:
                dataset = data_loader.load_data(use_image_types=eval_types,
                                                folder=eval_ds_folder,
                                                dataset_name=f'{eval_ds}/val_{gan_removal_eval}')
                model.load_model(
                    f"models/{model_ds}/cnnParams_resnet18_resized_{model_name}.pt", dataset)
                model.evaluate(dataset)
                # plot the confusion matrix
                sn.heatmap(model.df_cm, annot=True, ax=ax[options_gan_removal.index(gan_removal_eval)], fmt='g')
                ax[options_gan_removal.index(gan_removal_eval)].set_title(f"Eval: {gan_removal_eval}")
            if not os.path.exists(f"plots/mixed/fpRem/m_{model_ds}_e_{eval_ds}"):
                os.makedirs(f"plots/mixed/fpRem/m_{model_ds}_e_{eval_ds}")
            plt.savefig(f"plots/mixed/fpRem/m_{model_ds}_e_{eval_ds}/"
                        f"confusion_matrix_resized_model_{model_ds}_{'-'.join([e.value for e in model_types])}"
                        f"_{model_ds_folder}_eval_{eval_ds}-{'-'.join([e.value for e in eval_types])}_{eval_ds_folder}"
                        f"_m_orig_e_rem.png")
            # plt.savefig(f"results/{model_ds}/confusion_matrix_resized_{model_name}.png")
            plt.close()


def plotTrainWithEvalWithRemoval():
    # create a subplots
    for model_types, model_ds, model_ds_folder in options:
        for eval_types, eval_ds, eval_ds_folder in options:
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))
            # for gan_rem_train in options:
            model = SpoofedResizedClassifier(num_epochs=10,
                                             learning_rate=0.001,
                                             batch_size=16,
                                             folds=5,
                                             model_name="",
                                             dataset_name="",
                                             model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                                             loss_function=nn.CrossEntropyLoss(),
                                             num_image_channels=3,
                                             num_inputs_nodes=(580, 280))
            data_loader = ResizedDataLoader()
            if model_ds_folder == '':
                model_name = f'{model_ds}_' + '_'.join(e.value for e in model_types)
            else:
                model_name = f'{model_ds}_{model_ds_folder}_' + '_'.join(e.value for e in model_types)
            for gan_rem_eval in options_gan_removal:
                for gan_rem_train in options_gan_removal:
                    dataset = data_loader.load_data(use_image_types=eval_types,
                                                    folder=eval_ds_folder,
                                                    dataset_name=f'{eval_ds}/val_{gan_rem_eval}')
                    model.load_model(
                        f"models/{model_ds}/cnnParams_resnet18_resized_{model_name}_{gan_rem_train}.pt",
                        dataset)
                    model.evaluate(dataset)
                    # plot the confusion matrix
                    sn.heatmap(model.df_cm,
                               annot=True,
                               ax=ax[options_gan_removal.index(gan_rem_train)][options_gan_removal.index(gan_rem_eval)],
                               fmt='g')
                    ax[options_gan_removal.index(gan_rem_train)][options_gan_removal.index(gan_rem_eval)].set_title(
                        f"Train: {gan_rem_train} Eval: {gan_rem_eval}")
            plt.savefig(f"plots/mixed/fpRem/m_{model_ds}_e_{eval_ds}/"
                        f"confusion_matrix_resized_model_{model_ds}_{'-'.join([e.value for e in model_types])}"
                        f"_{model_ds_folder}_eval_{eval_ds}-{'-'.join([e.value for e in eval_types])}_{eval_ds_folder}"
                        f"_m_rem_e_rem.png")
            plt.close()


def main():
    plotTrainWithoutEvalWithRemoval()
    plotTrainWithEvalWithRemoval()


if __name__ == "__main__":
    main()
