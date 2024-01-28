from torch.utils.data import Dataset
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(self, dataset, indices, classes=None):
        self.samples = [dataset.samples[i] for i in indices]
        self.transform = dataset.transform
        self.class_to_idx = dataset.class_to_idx
        new_classes = []
        if classes is None:
            # Remove classes that are not in the dataset
            for s in self.samples:
                if [k for k, v in self.class_to_idx.items() if v == s[1]][0] not in new_classes:
                    # add the class names from class_to_idx to the newClasses list
                    new_classes.append([k for k, v in self.class_to_idx.items() if v == s[1]][0])
            self.classes = new_classes
            if len(self.classes) == 0:
                raise (RuntimeError("Dataset classes not found."))
            # remove classes from class_to_idx that are not in the dataset
            for k, v in list(self.class_to_idx.items()):
                if k not in self.classes:
                    del self.class_to_idx[k]
            # update class_to_idx
            self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
            # update samples from the old class indices from the dataset
            # class_to_idx to the new class indices fron self.class_to_idx
            self.samples = [
                (s[0],
                 self.class_to_idx[[k for k, v in dataset.class_to_idx.items() if v == s[1]][0]])
                for s in self.samples]
        else:
            self.classes = classes
            # update class_to_idx
            self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
            # update samples from the old class indices from the dataset
            # class_to_idx to the new class indices fron self.class_to_idx
            self.samples = [
                (s[0],
                 [v for k, v in self.class_to_idx.items() if k in s[0]][0])
                for s in self.samples]
        self.targets = torch.tensor([self.samples[i][1] for i in range(len(self.samples))])

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)
