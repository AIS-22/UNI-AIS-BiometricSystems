from typing import Tuple
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.nn.modules.loss import Module
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet

from classifier.AbstractClassifier import AbstractClassifier
from CustomDataset import CustomDataset


def _get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class GanClassifier(AbstractClassifier):
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
                 num_inputs_nodes: Tuple[int, int] = (736, 192)
                 ) -> object:

        self.num_inputs_nodes = num_inputs_nodes
        self.num_output_nodes = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = model
        self.criterion = loss_function
        self.optimizer = None
        self.num_image_channels = num_image_channels
        self.validation_loss = []
        self.val_loss = pd.DataFrame()
        self.accuracy = pd.DataFrame()
        self.confusion_matrix = []
        self.df_cm = pd.DataFrame()
        self.folds = folds
        self.batch_size = batch_size
        self.device = _get_device()

    def train(self, dataset):
        # Create a KFold object
        kfold = KFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Get the total length of your dataset
        dataset_length = len(dataset)
        self.num_output_nodes = len(dataset.classes)

        # This should change the input conv layer, maybe there is no need for defining the new image sizes
        self.model.conv1 = nn.Conv2d(self.num_image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_input_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_input_features, self.num_output_nodes)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

        # Iterate over the folds
        for fold, (train_idx, test_idx) in enumerate(kfold.split(range(dataset_length))):
            print('-----------------------------------------------------------------------')
            print(f"Fold {fold + 1}:")

            # Create train and validation subsets
            train_set = CustomDataset(dataset, train_idx)
            test_set = CustomDataset(dataset, test_idx)

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

            epoch_loss = running_loss / len(train_loader.dataset)
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
            test_loss /= len(test_loader.dataset)
            print(f'Validation accuracy: {accuracy:.4f} loss: {test_loss} in epoch: {epoch + 1}')
            losses[epoch, 0] = epoch_loss
            losses[epoch, 1] = test_loss
        self.validation_loss.append(losses)

    def evaluate(self, val_set):
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                predicted = torch.max(outputs.data, 1)[1]
                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=val_set.classes, columns=val_set.classes)

        accuracy = correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')

        self.accuracy = accuracy
        self.df_cm = df_cm

    def save_model(self):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), f"models/{self.dataset_name}/cnnParams_{self.model_name}.pt")
        print("Model saved")

    def load_model(self, model_path, dataset):
        num_ftrs = self.model.fc.in_features
        self.num_output_nodes = len(dataset.classes)
        self.model.fc = nn.Linear(num_ftrs, self.num_output_nodes)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print("Model loaded")

    def save_losses(self):
        losses = np.zeros((self.num_epochs, 2))
        # Add up losses of each fold run
        for loss in self.validation_loss:
            losses += loss

        # Get average
        losses /= self.folds
        df_losses = pd.DataFrame(losses, columns=['train_loss', 'val_loss'])

        df_losses.to_csv(f'results/{self.dataset_name}/losses_{self.model_name}.csv')
        print('Losses saved')

    def save_val_accuracy(self):
        np.save(f'results/{self.dataset_name}/accuracy_{self.model_name}.npy', self.accuracy)
        print('Accuracy saved')

    def save_val_confusion_matrix(self):
        self.df_cm.to_csv(f'results/{self.dataset_name}/conf_matrix_{self.model_name}.csv')
        print('Confusion matrix saved')


class GanClassifierResized(AbstractClassifier):
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
                 num_inputs_nodes: Tuple[int, int] = (736, 192)
                 ) -> object:

        self.num_inputs_nodes = num_inputs_nodes
        self.num_output_nodes = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = model
        self.criterion = loss_function
        self.optimizer = None
        self.num_image_channels = num_image_channels
        self.validation_loss = []
        self.accuracy = float
        self.confusion_matrix = []
        self.df_cm = pd.DataFrame()
        self.folds = folds
        self.batch_size = batch_size
        self.device = _get_device()

    def train(self, dataset):
        # Create a KFold object
        kfold = KFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Get the total length of your dataset
        dataset_length = len(dataset)
        self.num_output_nodes = len(dataset.classes)

        # This should change the input conv layer, maybe there is no need for defining the new image sizes
        # self.model.conv1 = nn.Conv2d(self.num_image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_input_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_input_features, self.num_output_nodes)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

        # Iterate over the folds
        for fold, (train_idx, test_idx) in enumerate(kfold.split(range(dataset_length))):
            print('-----------------------------------------------------------------------')
            print(f"Fold {fold + 1}:")

            # Create train and validation subsets
            train_set = CustomDataset(dataset, train_idx)
            test_set = CustomDataset(dataset, test_idx)

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

            epoch_loss = running_loss / len(train_loader.dataset)
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
            test_loss /= len(test_loader.dataset)
            print(f'Validation accuracy: {accuracy:.4f} loss: {test_loss} in epoch: {epoch + 1}')
            losses[epoch, 0] = epoch_loss
            losses[epoch, 1] = test_loss

        self.validation_loss.append(losses)

    def evaluate(self, val_set):
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                predicted = torch.max(outputs.data, 1)[1]
                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=val_set.classes, columns=val_set.classes)

        accuracy = correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')

        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
        self.df_cm = df_cm

    def save_model(self):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), f"models/{self.dataset_name}/cnnParams_{self.model_name}.pt")
        print("Model saved")

    def load_model(self, model_path, dataset):
        num_ftrs = self.model.fc.in_features
        self.num_output_nodes = len(dataset.classes)
        self.model.fc = nn.Linear(num_ftrs, self.num_output_nodes)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print("Model loaded")

    def save_losses(self):
        losses = np.zeros((self.num_epochs, 2))
        # Add up losses of each fold run
        for loss in self.validation_loss:
            losses += loss

        # Get average
        losses /= self.folds
        df_losses = pd.DataFrame(losses, columns=['train_loss', 'val_loss'])

        df_losses.to_csv(f'results/{self.dataset_name}/losses_{self.model_name}.csv')
        print('Losses saved')

    def save_val_accuracy(self, eval_ds, model_ds):
        if not os.path.exists("results/mixed/ganSeperator/"):
            os.makedirs("results/mixed/ganSeperator/")
        np.save(
            f"results/mixed/ganSeperator/"
            f"accuracy_ganSeperator_resized_model_{model_ds}_eval_{eval_ds}.npy",
            self.accuracy)
        print('Accuracy saved')

    def save_val_confusion_matrix(self, eval_ds, model_ds):
        if not os.path.exists("results/mixed/ganSeperator/"):
            os.makedirs("results/mixed/ganSeperator/")
        self.df_cm.to_csv(
            f"results/mixed/ganSeperator/"
            f"conf_matrix_ganSeperator_resized_model_{model_ds}_eval_{eval_ds}.csv")
        print('Confusion matrix saved')
