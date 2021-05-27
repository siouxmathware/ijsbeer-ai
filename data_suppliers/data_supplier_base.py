"""
Module that contains the trainer base class.
"""
import logging
from abc import ABC, abstractmethod
import os.path as op

LOGGER = logging.getLogger(__name__)


class DataSupplierBase(ABC):
    """
    Abstract baseclass for trainer
    """
    @abstractmethod
    def __init__(self, **data_parameters):
        """
        :param model_parameters: Should contain the arguments to initiate the model.
        :param data_parameters: Should contain the arguments to initiate the data supplier.
        :param train_parameters: Should contain arguments to be able to train
        """

    @abstractmethod
    def __call__(self):
        pass

    @staticmethod
    def make_parameter_path(file_path):
        return op.join(op.dirname(file_path), 'parameters.json')
