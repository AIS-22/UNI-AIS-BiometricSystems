import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os
import pandas as pd


def print_accuracy(dataset_name):
    for filename in os.listdir('results/' + dataset_name):
        if 'accuracy' in filename:
            accuracy = np.load('results/' + dataset_name + '/' + filename)
            print('Modelname: ', filename)
            print('Accuracy:', accuracy)
            print('\n')


def plot_confusion_matrix(dataset_name):
    datasets = ['PLUS', 'PROTECT', 'IDIAP', 'SCUT']
    for model_ds in datasets:
        for eval_ds in datasets:
            for filename in os.listdir(f'results/mixed/m_{model_ds}_e_{eval_ds}'):
                if 'conf_matrix' in filename:
                    file = filename.split('.')[0]
                    cm = pd.read_csv(f'results/mixed/m_{model_ds}_e_{eval_ds}/{file}.csv',
                                     index_col=0, header=0, sep=',')
                    plt.figure(figsize=(15, 10))
                    sn.set(font_scale=1.4)
                    sn.heatmap(cm,
                               annot=True,
                               cmap='Blues',
                               fmt=".0f",
                               annot_kws={'fontsize': 20})
                    plt.xticks(rotation=45)
                    plt.savefig(f'plots/mixed/m_{model_ds}_e_{eval_ds}/{file}.png', bbox_inches='tight')
                    print('Confusion Matrix Plot saved as: ' + f'plots/mixed/m_{model_ds}_e_{eval_ds}/{file}.png')
                    plt.close()


def plot_loss_results(dataset_name):
    for filename in os.listdir('results/' + dataset_name):
        if 'loss' in filename:
            file = filename.split('.')[0]
            split_path = file.split('_')

            categories = [split_path[-2], split_path[-1]]
            # load dic from file
            loss = np.load('results/' + dataset_name + '/' + filename, allow_pickle=True)
            plt.figure(figsize=(20, 10))
            plt.plot(range(1, 11), loss[:, 0], c='r', label='Train Loss')
            plt.plot(range(1, 11), loss[:, 1], c='g', label='Validation Loss')
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(title='Loss Functions ' + categories[0] + " - " + categories[1])
            plt.savefig('plots/' + dataset_name + '/' + file + '.png')
            print('Loss Plot saved as: ' + 'plots/' + dataset_name + '/' + file + '.png')
            plt.close()


def main():
    datasets = ['mixed']
    for dataset_name in datasets:
        print('Plotting results for ' + dataset_name)
        print_accuracy(dataset_name)
        plot_confusion_matrix(dataset_name)
        plot_loss_results(dataset_name)


if __name__ == '__main__':
    main()
