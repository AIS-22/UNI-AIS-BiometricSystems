from abc import ABC, abstractmethod


class AbstractClassifier(ABC):
    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def evaluate(self, X) -> None:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self, path) -> None:
        pass
