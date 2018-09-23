"""
Abstract class which is inherited by models in this project.
"""

from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def load(self, save_path):
        raise NotImplementedError()

    @abstractmethod
    def train(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError()

    @staticmethod
    def preprocess(x):
        raise NotImplementedError()

    @staticmethod
    def postprocess(x):
        raise NotImplementedError()

    @abstractmethod
    def build(self):
        raise NotImplementedError()
