from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.modules.loss import Module
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
                 model: ResNet,
                 loss_function: Module,
                 num_image_channels: int,
                 num_inputs_nodes: Tuple[int, int] = (736, 192),
                 num_output_nodes: int = 2,
                 should_save_model: bool = True):

        self.num_inputs_nodes = num_inputs_nodes
        self.num_output_nodes = num_output_nodes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model = model
        self.criterion = loss_function
        self.optimizer = None
        self.num_image_channels = num_image_channels
        self.should_save_model = should_save_model

    def train(self, train_loader, val_loader):
        device = _get_device()

        # This should change the input conv layer, maybe there is no need for defining the new image sizes
        self.model.conv1 = nn.Conv2d(self.num_image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_output_nodes)
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        losses = np.zeros((self.num_epochs, 2))
        for epoch in range(self.num_epochs):
            print(f'Start to train epoch: {epoch + 1}')
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset.samples)
            print(f'Train Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}')

            self.model.eval()
            correct = 0
            total = 0
            test_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)  # Update the total count of processed samples
                    test_loss += self.criterion(outputs, labels).item() * images.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            test_loss /= len(val_loader.dataset.samples)
            print(f'Validation accuracy: {accuracy:.4f} loss: {test_loss} in epoch: {epoch + 1}')
            losses[epoch, 0] = epoch_loss
            losses[epoch, 1] = test_loss
        np.save('results/losses_' + self.model_name + '.npy', losses)
        print('Model trained and losses saved')

        if self.should_save_model:
            self.save_model()

    def evaluate(self, val_loader):
        device = _get_device()
        self.model.to(device)
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)  # Update the total count of processed samples
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1_score:.4f}')

        np.save('results/' + self.model_name + '_results.npy', accuracy)

    def save_model(self):
        torch.save(self.model.state_dict(), "models/cnnParams_" + self.model_name + ".pt")
        print("Model saved")

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("Model loaded")
