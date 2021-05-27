"""
Module containing generators for text files datasets that contain BIO labels.
"""

import logging
import os.path as op
from official.nlp.bert import tokenization

from data_suppliers.data_supplier_base import DataSupplierBase
from common import bucketfs
from data_suppliers.text.nerbio.data_classes import InputData, InputDataSentence
from data_suppliers.text.nerbio import data_classes

LOGGER = logging.getLogger(__name__)

VOCAB_FILE = "vocab.txt"


class DataGenerator(DataSupplierBase):
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
        self.input_folder = input_folder
        self.dataset_types = dataset_types
        self.split_long_sentences = split_long_sentences

        # Initiate config
        self.vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(self.vocab_file, do_lower_case)
        self.max_length = bert_layer.input_shape[0][1]

        self.vocab = self.initiate_vocab()
        self.nr_batches = self._compute_nr_batches()
        self.nr_tags = len(self.vocab)

    def _get_validation_data(self):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            validation_data = InputData.from_gt_file(op.join(source, 'validation.txt'),
                                                     self.tokenizer, self.max_length,
                                                     do_one_hot=True, vocab=self.vocab,
                                                     split_long_sentences=self.split_long_sentences)
            return validation_data

    def __call__(self):
        # Train data
        validation: InputData = self._get_validation_data()

        fit_argument_dict = {
            "x": self.train_generator(),
            "validation_data": (validation.get_x(), validation.get_y()),
            "batch_size": self.batch_size,
            "steps_per_epoch": self.nr_batches['train']
        }

        return fit_argument_dict

    def _compute_nr_batches(self):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            nr_batches = {dataset: self._compute_nr_batches_single(op.join(source, dataset + '.txt'))
                          for dataset in self.dataset_types}
        return nr_batches

    def _compute_nr_batches_single(self, file_path):
            with open(file_path, mode="r", encoding='ISO-8859-1') as f:
                # Note that the line below always rounds up, which is maybe not ideal, but kept for simplicity
                # the computation is innaccurate anyway, as it ignores the fact that long lines are split of ignored
                length = sum(1 for line in f.readlines() if line == '\n') // self.batch_size + 1
            return length

    def sentences_generator(self, dataset_type, label_sep="\t"):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            file_path = op.join(source, dataset_type + '.txt')
            while True:
                with open(file_path, mode="r", encoding='ISO-8859-1') as f:
                    line = f.readline()
                    sentence_with_labels = []
                    while line:
                        if line == '\n' or len(sentence_with_labels) > 3*self.max_length:
                            # if a line is empty, we have a sentence break
                            # TODO: bit of a hack to deal with improper files to also break when the sentence gets long
                            if sentence_with_labels:  # do not yield empty sentence
                                yield sentence_with_labels
                                sentence_with_labels = []  # restart the list
                        else:  # add the word and tag to the list
                            # note that we split on \n and take the first part to robustly remove \n at the end
                            word_with_label = tuple(line.split('\n')[0].split(label_sep))
                            assert len(word_with_label) == 2
                            sentence_with_labels.append(word_with_label)
                        line = f.readline()

    def generator(self, dataset_type):
        sentence_generator = self.sentences_generator(dataset_type)
        gathered = []
        while True:
            if len(gathered) > self.batch_size:
                batch_raw = gathered[:self.batch_size]
                gathered = gathered[self.batch_size:]
                yield InputData(batch_raw, self.tokenizer, self.max_length, self.vocab)
            else:
                sentences_with_labels = next(sentence_generator)
                sentences, labelss = data_classes.split_sentence_labels([sentences_with_labels])
                # note that sentences and labelss will always have length 1 for this approach
                sentece_data_list = InputDataSentence.get_fitting_sentence_data_list(
                    sentences[0], self.tokenizer, self.max_length, self.vocab,
                    labels=labelss[0], split_long_sentences=self.split_long_sentences
                )
                gathered += sentece_data_list

    def train_generator(self):
        for input_data in self.generator('train'):
            yield input_data.get_x(), input_data.get_y()

    def initiate_vocab(self):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            with open(f'{source}/{VOCAB_FILE}', mode='r') as f:
                tags = f.read().split("\n")
        return tags


