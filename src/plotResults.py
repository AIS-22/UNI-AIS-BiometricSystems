import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def print_accuracy():
    accuracy = np.load('results/resnet18_gen_spoof_accuracy.npy')
    print('accuracy:', accuracy)


def plot_confusion_matrix():
    categories = ['genuine', 'spoofed']

    # load dic from file
    conf_matrix = np.load('results/resnet18_gen_spoof_conf_matrix.npy', allow_pickle=True)
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
    plt.savefig('plots/conf_matrix.png')


def plot_loss_results():
    # load dic from file
    loss = np.load('results/losses_resnet18_gen_spoof.npy', allow_pickle=True)
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, 11), loss[:, 0], c='r', label='Train Loss')
    plt.plot(range(1, 11), loss[:, 1], c='g', label='Validation Loss')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Loss Functions')
    plt.savefig('plots/loss_comparison.png')


def main():
    print_accuracy()
    plot_confusion_matrix()
    plot_loss_results()


if __name__ == '__main__':
    main()
