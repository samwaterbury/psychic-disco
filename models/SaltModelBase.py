"""
Abstract class which is inherited by models in this project.
"""

from abc import ABC, abstractmethod


class SaltModelBase(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, x_train, y_train, x_valid, y_valid, update_cutoff):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def build(self):
        raise NotImplementedError()
