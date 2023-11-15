import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import os

def print_accuracy(datasetName):
    for filename in os.listdir('results/'+datasetName):
        if 'accuracy' in filename:
            accuracy = np.load('results/' + datasetName + '/' + filename)
            print('Modelname: ', filename)
            print('Accuracy:', accuracy)
            print('\n')

def plot_confusion_matrix(datasetName):
    for filename in os.listdir('results/'+datasetName):
        if 'conf_matrix' in filename:
            file = filename.split('.')[0]
            splitted_path = file.split('_')

            categories = [splitted_path[-2], splitted_path[-1]]

            # load dic from file
            conf_matrix = np.load('results/' + datasetName + '/' + filename, allow_pickle=True)
            cmap = plt.get_cmap('tab20')

            plt.figure(figsize=(15, 10))
            sn.set(font_scale=1.4)
            sn.heatmap(conf_matrix, vmin=0, vmax=np.max(conf_matrix),
                       xticklabels=categories,
                       yticklabels=categories,
                       annot=True,
                       cmap='Blues',
                       fmt=".0f",
                       annot_kws={'fontsize': 20})
            plt.xticks(rotation=45)
            plt.savefig('plots/' + datasetName + '/conf_matrix_' + file + '.png')


def plot_loss_results(datasetName):

    for filename in os.listdir('results/'+datasetName):
        if 'loss' in filename:
            file = filename.split('.')[0]
            splitted_path = file.split('_')

            categories = [splitted_path[-2], splitted_path[-1]]
            # load dic from file
            loss = np.load('results/'+datasetName + '/' + filename, allow_pickle=True)
            plt.figure(figsize=(20, 10))
            plt.plot(range(1, 11), loss[:, 0], c='r', label='Train Loss')
            plt.plot(range(1, 11), loss[:, 1], c='g', label='Validation Loss')
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(title='Loss Functions ' + categories[0] + " - " + categories[1])
            plt.savefig('plots/' + datasetName + '/loss_comparison_' + file + '.png')


def main():
    datasetName = 'PLUS'
    print_accuracy(datasetName)
    plot_confusion_matrix(datasetName)
    plot_loss_results(datasetName)


if __name__ == '__main__':
    main()
