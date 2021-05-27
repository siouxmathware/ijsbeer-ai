from abc import ABC, abstractmethod
import os.path as op
from typing import Tuple


class Bert(ABC):
    """
    Abstract Base Class for the BERT to be used in the pipeline. Has a factory method
    :meth:`create_bert <lib.ner_bert.inference.Bert.create_bert>` that returns either a real BERT in TensorFlow-style
    based on a .ht weights file or a mocked BERT. PyTorch is no longer supported.
    """
    @abstractmethod
    def __init__(self, nr_tags: int, weights_path: str = None):
        """
        :param nr_tags: Number of tags used by the BERT
        :param weights_path: Path to the weights file.
        """
        pass

    @abstractmethod
    def __call__(self, sentences):
        """
        :param sentences: Sentences to do NER on
        :return: Predictions in numpy array form, it is left to the data supplier class to handle the translation to
        readable values.
        """
        pass

    @classmethod
    def create_bert(cls, vocab: Tuple[str, ...], weights_path: str):
        """
        :param vocab: Number of tags used by the BERT
        :param weights_path: Path to the weights file.
        :return: An instance of the appropriate BERT class, determined by the extension of the weights file:

            * .h5 results in a TensorFlow BERT

            * no extension, e.g. by passing an empty string for the weights, results in a mocked BERT

            Other implementations are no longer supported.
        """
        extension = op.splitext(weights_path)[1]
        if extension == '.pt':
            raise NotImplementedError('Using PyTorch models is no longer supported as of 84f4e7589789f.')
        elif extension == '.h5':
            from models.nlp.bert_home_trained.model import BertTF
            return BertTF(len(vocab), weights_path)
        elif extension == '':
            from models.nlp.mocks.mock_bert import MockBertModel
            nr_categories = (len(vocab) - 2) // 2
            return MockBertModel(nr_categories, 'mock', None, None)
        else:
            raise ValueError("Invalid weights file")
