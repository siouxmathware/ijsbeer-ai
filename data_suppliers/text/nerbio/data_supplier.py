"""
Module containing generators for text files datasets that contain BIO labels.
"""

import logging
import os.path as op
from official.nlp.bert import tokenization

from data_suppliers.data_supplier_base import DataSupplierBase
from common import bucketfs
from data_suppliers.text.nerbio.data_classes import InputData

LOGGER = logging.getLogger(__name__)

VOCAB_FILE = "vocab.txt"


class DataSupplier(DataSupplierBase):
    """
    Represents BIO datasets suitable for tf keras model fit method. Assumes data fits completely in memory.
    """
    parameter_path = DataSupplierBase.make_parameter_path(__file__)

    def __init__(self, bert_layer, input_folder, batch_size,
                 dataset_types=("train", "validation"), split_long_sentences=True):
        """
        Expects input folder where there exists 4 txt files.
        - train.txt/validation.txt/test.txt where per line there is 1 word and label, separated by a \t. Sentences are
        separated by a double \n character, (empty line).
        - vocab.txt where each line should contain a single entity
        """
        super().__init__()
        # Initiate parameters
        self.batch_size = batch_size

        # Initiate config
        self.vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(self.vocab_file, do_lower_case)
        self.max_length = bert_layer.input_shape[0][1]

        # Initiate vocab entities
        with bucketfs.mount(input_folder, is_dir=True) as source:
            with open(f'{source}/{VOCAB_FILE}', mode='r') as f:
                tags = f.read().split("\n")
            self.nr_tags = len(tags)
            self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
            self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

            # Initiate datasets
            self.datasets = {
                dataset_type: InputData.from_gt_file(op.join(source, dataset_type + '.txt'),
                                                     self.tokenizer, self.max_length,
                                                     do_one_hot=True, vocab=self.tag2idx,
                                                     split_long_sentences=split_long_sentences)
                for dataset_type in dataset_types
            }

    def __call__(self):
        # Train data

        train: InputData = self.datasets['train']
        validation: InputData = self.datasets['validation']

        fit_argument_dict = {
            "x": train.get_x(),
            "y": train.get_y(),
            "validation_data": (validation.get_x(), validation.get_y()),
            "batch_size": self.batch_size
        }

        return fit_argument_dict

    def batcher(self, set_name, start_batch=0, cond=None):
        """
        Return batches of the data
        :param set_name: which data_set to choose
        :param start_batch: where to start, which element
        :param cond: a function of start_batch and dataset.size - self.batch_size
        :return:
        """
        dataset: InputData = self.datasets[set_name]
        x = dataset.get_x()
        y = dataset.get_y()
        sentence_data = dataset.sentence_data
        if cond is None:
            def cond(_start_batch, _dataset_size_minus_batch_size):
                return True
        while cond(start_batch, dataset.size - self.batch_size):
            end_batch = start_batch + self.batch_size
            if end_batch > dataset.size:
                start_batch = end_batch % dataset.size
                end_batch = start_batch + self.batch_size
            batch_x = [x_el[start_batch:end_batch, :] for x_el in x]
            batch_y = y[start_batch:end_batch]
            batch_sentence_data = sentence_data[start_batch:end_batch]
            yield batch_x, batch_y, batch_sentence_data
            start_batch = end_batch
        return
