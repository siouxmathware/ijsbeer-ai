"""
Module containing generators for text files datasets that contain BIO labels.
This particular generator simply produces chunks of max_length tokens each.
"""

import logging
import os.path as op
from official.nlp.bert import tokenization

from data_suppliers.data_supplier_base import DataSupplierBase
from common import bucketfs
from data_suppliers.text.nerbio.data_classes import InputData, InputDataSentence, Tokenized

LOGGER = logging.getLogger(__name__)

VOCAB_FILE = "vocab.txt"


class DataGeneratorChunk(DataSupplierBase):
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

        if 'validation' in dataset_types:
            self.validation_data = self._get_validation_data()

    def _get_validation_data(self):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            validation_data = InputData.from_gt_file(op.join(source, 'validation.txt'),
                                                     self.tokenizer, self.max_length,
                                                     do_one_hot=True, vocab=self.vocab,
                                                     split_long_sentences=self.split_long_sentences)
            return validation_data

    def __call__(self):
        # Train data
        validation: InputData = self.validation_data

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
                # Note that the line below always rounds down, which is maybe not ideal, but kept for simplicity
                # the computation is innaccurate anyway, as it ignores the fact that long lines are split of ignored
                length = sum(1 for line in f.readlines() if line == '\n') // self.batch_size
            return length

    def word_generator(self, dataset_type, label_sep="\t"):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            file_path = op.join(source, dataset_type + '.txt')
            while True:
                with open(file_path, mode="r", encoding='ISO-8859-1') as f:
                    line = f.readline()
                    while line:
                        if line == '\n':  # if a line is empty, we have a sentence break
                            yield Tokenized('', ['[SEP]'], self.tokenizer, mask=0, label='<PAD>')
                        else:
                            word_with_label = tuple(line.split('\n')[0].split(label_sep))
                            assert len(word_with_label) == 2
                            word, label = word_with_label
                            yield Tokenized(word, self.tokenizer.tokenize(word), self.tokenizer, label=label)
                        line = f.readline()

    def sentence_generator(self, dataset_type, label_sep="\t"):
        word_generator = self.word_generator(dataset_type, label_sep)
        gathered_words = []
        overlap = None
        while True:
            word = next(word_generator)
            gathered_words.append(word)
            if len([token for w in gathered_words for token in w.tokens]) > self.max_length:
                # if the sentence has become too long, move the last words to the next batch
                batch_words = gathered_words[:-1]
                overlap = overlap or len(batch_words) // 2
                gathered_words = gathered_words[overlap:]
                overlap = len(gathered_words) - 1
                sentence = [w.word for w in batch_words]
                yield InputDataSentence(sentence, batch_words, self.max_length, vocab=self.vocab)

    def generator(self, dataset_type):
        sentence_generator = self.sentence_generator(dataset_type)
        while True:
            gathered = [next(sentence_generator) for _ in range(self.batch_size)]
            yield InputData(gathered, self.tokenizer, self.max_length)

    def train_generator(self):
        for input_data in self.generator('train'):
            yield input_data.get_x(), input_data.get_y()

    def initiate_vocab(self):
        with bucketfs.mount(self.input_folder, is_dir=True) as source:
            with open(f'{source}/{VOCAB_FILE}', mode='r') as f:
                tags = f.read().split("\n")
        return tags
