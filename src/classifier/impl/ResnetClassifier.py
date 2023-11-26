from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.nn.modules.loss import Module
from torch.utils.data import DataLoader, Subset
from torchvision.models.resnet import ResNet

from src.classifier.AbstractClassifier import AbstractClassifier


def _get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class ResnetClassifier(AbstractClassifier):
    def __init__(self,
                 num_epochs: int,
                 learning_rate: float,
                 model_name: str,
                 dataset_name: str,
                 model: ResNet,
                 loss_function: Module,
                 num_image_channels: int,
                 batch_size: int,
                 folds: int,
                 num_inputs_nodes: Tuple[int, int] = (736, 192),
                 num_output_nodes: int = 2
                 ) -> object:

        self.num_inputs_nodes = num_inputs_nodes
        self.num_output_nodes = num_output_nodes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = model
        self.criterion = loss_function
        self.optimizer = None
        self.num_image_channels = num_image_channels
        self.validation_loss = []
        self.accuracy = []
        self.confusion_matrix = []
        self.folds = folds
        self.batch_size = batch_size
        self.device = _get_device()

    def train(self, dataset):
        # Create a KFold object
        kfold = KFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Get the total length of your dataset
        dataset_length = len(dataset)

        # This should change the input conv layer, maybe there is no need for defining the new image sizes
        self.model.conv1 = nn.Conv2d(self.num_image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_output_nodes)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Iterate over the folds
        for fold, (train_idx, test_idx) in enumerate(kfold.split(range(dataset_length))):
            print('-----------------------------------------------------------------------')
            print(f"Fold {fold + 1}:")

            # Create train and validation subsets
            train_set = Subset(dataset, train_idx)
            test_set = Subset(dataset, test_idx)

            # Create DataLoader for training and validation
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

            self.train_fold(train_loader, test_loader)

    def train_fold(self, train_loader, test_loader):

        losses = np.zeros((self.num_epochs, 2))
        for epoch in range(self.num_epochs):
            print(f'Start to train epoch: {epoch + 1}')
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset.indices)
            print(f'Train Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}')

            self.model.eval()
            correct = 0
            total = 0
            test_loss = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)  # Update the total count of processed samples
                    test_loss += self.criterion(outputs, labels).item() * images.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            test_loss /= len(test_loader.dataset.indices)
            print(f'Validation accuracy: {accuracy:.4f} loss: {test_loss} in epoch: {epoch + 1}')
            losses[epoch, 0] = epoch_loss
            losses[epoch, 1] = test_loss

        self.validation_loss.append(losses)

    def evaluate(self, val_loader):
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        confusion_matrix = np.zeros((2, 2))

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)  # Update the total count of processed samples
                correct += (predicted == labels).sum().item()

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')

        self.accuracy.append(accuracy)
        self.confusion_matrix.append(confusion_matrix)

    def save_model(self):
        torch.save(self.model.state_dict(), "models/"+self.dataset_name +"/cnnParams_" + self.model_name + ".pt")
        print("Model saved")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded")

    def save_losses(self):
        losses = np.zeros((self.num_epochs, 2))
        # Add up losses of each fold run
        for loss in self.validation_loss:
            losses += loss

        # Get average
        losses /= self.folds

        np.save('results/'+self.dataset_name +'/losses_' + self.model_name + '.npy', losses)
        print('Losses saved')

    def save_accuracy(self):
        acc = np.average(self.accuracy)
        np.save('results/' + self.dataset_name + '/accuracy_' + self.model_name + '.npy', acc)
        print('Accuracy saved')

    def save_confusion_matrix(self):
        conf_matrix = np.zeros((2, 2))
        # Add up losses of each fold run
        for conf_element in self.confusion_matrix:
            conf_matrix += conf_element

        np.save('results/' + self.dataset_name + '/conf_matrix_' + self.model_name + '.npy', conf_matrix)
        print('Confusion matrix saved')
