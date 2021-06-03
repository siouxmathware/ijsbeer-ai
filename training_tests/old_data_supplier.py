"""
Module containing generators for text files datasets that contain BIO labels.
"""

import logging

import numpy as np
from tensorflow import one_hot
from official.nlp.bert import tokenization

from data_suppliers.data_supplier_base import DataSupplierBase
from common import bucketfs

LOGGER = logging.getLogger(__name__)

VOCAB_FILE = "vocab.txt"


class DataSupplier(DataSupplierBase):
    """
    Represents BIO datasets suitable for tf keras model fit method. Assumes data fits completely in memory.
    """
    parameter_path = DataSupplierBase.make_parameter_path(__file__)

    def __init__(self, bert_layer, input_folder, batch_size,
                 dataset_types=("train", "validation")):
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
        self.datasets = dict()
        for dataset_type in dataset_types:
            self.datasets[dataset_type] = Dataset(input_folder, dataset_type,
                                                  self.tokenizer, self.tag2idx, self.max_length)

    def __call__(self):
        # Train data
        train_token_ids = self.list2array([sentence["token_ids"] for sentence in self.datasets["train"].sentences])
        train_input_mask = self.list2array([sentence["input_mask"] for sentence in self.datasets["train"].sentences])
        train_input_type_ids = self.list2array([[0]*len(sentence) for sentence in self.datasets["train"].sentences])
        train_is_heads = self.list2array([sentence["is_heads"] for sentence in self.datasets["train"].sentences])
        train_labels = self.list2array([sentence["label_ids"] for sentence in self.datasets["train"].sentences])

        # Validation data
        validation_token_ids = self.list2array([sentence["token_ids"] for sentence in
                                                self.datasets["validation"].sentences])
        validation_input_mask = self.list2array([sentence["input_mask"] for sentence in
                                                 self.datasets["validation"].sentences])
        validation_input_type_ids = self.list2array([[0]*len(sentence) for sentence in
                                                     self.datasets["validation"].sentences])
        validation_is_heads = self.list2array([sentence["is_heads"] for sentence in
                                               self.datasets["validation"].sentences])
        validation_labels = self.list2array([sentence["label_ids"] for sentence in
                                             self.datasets["validation"].sentences])

        train_labels = one_hot(train_labels, self.nr_tags)
        validation_labels = one_hot(validation_labels, self.nr_tags)

        fit_argument_dict = {
            "x": [train_token_ids, train_input_mask, train_input_type_ids, train_is_heads],
            "y": train_labels,
            "validation_data": ([validation_token_ids, validation_input_mask, validation_input_type_ids,
                                 validation_is_heads], validation_labels),
            "batch_size": self.batch_size
        }

        return fit_argument_dict

    def list2array(self, some_list, dtype='int32'):
        n = len(some_list)
        ar = np.zeros((n, self.max_length), dtype=dtype)
        for i, j in enumerate(some_list):
            ar[i][0:len(j)] = j
        return ar


class Dataset(object):
    """
    Represents a dataset. Input file should contain sentences of which
    """

    def __init__(self, folder, set_type, tokenizer, tag2idx, max_length):
        # Open file and convert to sentences
        with bucketfs.mount(folder, is_dir=True) as source:
            with open(f'{source}/{set_type}.txt', mode="r", encoding='ISO-8859-1') as f:
                sentences, labels = text_to_sentences(f.read())

        # Let us convert each sentence into tokens
        self.nr_tokens_list = [[len(tokenizer.tokenize(word)) for word in sentence] for sentence in sentences]
        self.tokens_list = [[x for y in [tokenizer.tokenize(word) for word in sentence] for x in y]
                            for sentence in sentences]
        self.sentences = []
        nr_ignored_sentences = 0
        nr_ignored_tokens = 0

        for sentence, tokens, nr_tokens, labels_per_sentence, index in zip(sentences, self.tokens_list,
                                                                              self.nr_tokens_list, labels,
                                                                              range(len(sentences))):
            if sum(nr_tokens) > max_length-2:
                nr_ignored_sentences += 1
                nr_ignored_tokens += sum(nr_tokens)
                LOGGER.warning(f"Sentence {index} with {sum(nr_tokens)} tokens is too long and will be ignored!")
            else:
                tokens_plus = ["[CLS]"] + tokens + ["[SEP]"]
                self.sentences.append({
                    'is_heads': [0] + [x for y in [[1] + [0] * (i - 1) for i in nr_tokens] for x in y] + [0],
                    'label_ids': [0] + [x for y in [[tag2idx[label]] + [0] * (i - 1) for i, label in
                                                    zip(nr_tokens, labels_per_sentence)] for x in y] + [0],
                    'tokens': ["[CLS]"] + tokens + ["[SEP]"],
                    'token_ids': tokenizer.convert_tokens_to_ids(tokens_plus),
                    'sentence': sentence,
                    'labels': labels_per_sentence,
                    'input_mask': [0] + [1]*sum(nr_tokens) + [0]
                })

        nr_words = len([w for s in sentences for w in s])
        nr_tokens = sum([sum(x) for x in self.nr_tokens_list])

        LOGGER.info("As input %d sentences were received which have been tokenized with ratio %d/%d=%.3f"
                    % (len(sentences), nr_words, nr_tokens, nr_words / nr_tokens))
        LOGGER.info(f"In total there are {nr_ignored_sentences} sentences ignored which represents "
                    f"{nr_ignored_tokens} out of {nr_tokens} tokens.")


def sentences_to_text(sentences, labels, sentence_sep="\n\n", word_sep="\n", label_sep="\t"):
    """
    Converts sentences (a list of list of words) and labels to text.
    """
    text = sentence_sep.join(
        [word_sep.join([w + label_sep + l for w, l in zip(sentence, sentence_labels)]) for sentence, sentence_labels in
         zip(sentences, labels)])
    return text


def text_to_sentences(text, sentence_sep="\n\n", word_sep="\n", label_sep="\t"):
    """
    Converts text to sentences (a list of list of word) and labels.
    """
    def word_label_condition(word_label):
        """
        When words and labels are taken into account
        """
        c1 = label_sep in word_label
        c2 = c1 and (len(word_label.split(label_sep)[0]) > 0)
        c3 = c1 and (len(word_label.split(label_sep)[1]) > 0)
        return c2 and c3

    def sentence_condition(sentence):
        """
        When sentences are taken into account
        """
        return len(sentence) > 0

    sentences_with_labels = [[tuple(word_label.split(label_sep)) for word_label in sentence.split(word_sep) if
                              word_label_condition(word_label)] for sentence in text.split(sentence_sep) if
                             sentence_condition(sentence)]
    sentences = [[word for word, label in sentence] for sentence in sentences_with_labels]
    labels = [[label for word, label in sentence] for sentence in sentences_with_labels]

    return sentences, labels
