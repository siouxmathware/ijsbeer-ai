from typing import List
import logging
import numpy as np
from tensorflow import one_hot


LOGGER = logging.getLogger(__name__)


class Tokenized:
    def __init__(self, word, tokens, tokenizer, mask=1, input_type=0, label: str=None):
        self.word = word
        self.tokens = tokens
        self.ids = tokenizer.convert_tokens_to_ids(tokens)
        self.is_heads = [mask] + [0]*(len(tokens) - 1)  # use mask here so it's zero for [CLS] etc.
        self.masks = [mask] * len(tokens)
        self.input_types = [input_type] * len(tokens)
        self.labels = [label] + ['<PAD>']*(len(tokens) - 1)


class InputDataSentence:
    def __init__(self, sentence, tokenized, max_length, vocab, do_one_hot=True):
        """
        Class that defines a sentence of input data
        :param sentence: the text, kept primarily as a reference/sanity check
        :param tokenized: the tokenized words
        :param max_length: the maximum length of a tokenized "sentence"
        :param vocab: a list of vocabulary for the network to be used (e.g. ['O', '<PAD>', 'B-PER', ...]
        :param do_one_hot: whether the labels are one_hot encodings
        """
        self.sentence = sentence
        self.tokenized = tokenized
        self.max_length = max_length
        self.idx2tag = {idx: tag for idx, tag in enumerate(vocab)}
        self.tag2idx = {tag: idx for idx, tag in self.idx2tag.items()}
        self.do_one_hot = do_one_hot

        self.masks = [mask for tokenized_word in self.tokenized for mask in tokenized_word.masks]
        self.input_types = [input_type for tokenized_word in self.tokenized
                            for input_type in tokenized_word.input_types]
        self.is_heads = [is_head for tokenized_word in self.tokenized for is_head in tokenized_word.is_heads]
        self.labels = [self.tag2idx.get(label, None)
                       for tokenized_word in self.tokenized for label in tokenized_word.labels]
        self.ids = [idx for tokenized_word in self.tokenized for idx in tokenized_word.ids]
        self.fits = len(self.ids) <= self.max_length

    @classmethod
    def get_sentence_data_list(cls, sentence, tokenizer, max_length, vocab, labels, do_one_hot=True):
        tokenized_words = [Tokenized(word, tokenizer.tokenize(word), tokenizer, label=label)
                           for word, label in zip(sentence, labels)]
        tokenized = [Tokenized('', ['[CLS]'], tokenizer, mask=0, label='<PAD>')
                     ] + tokenized_words + [Tokenized('', ['[SEP]'], tokenizer, mask=0, label='<PAD>')]
        return cls(sentence, tokenized, max_length, vocab, do_one_hot=do_one_hot)

    @classmethod
    def get_fitting_sentence_data_list(cls, sentence, tokenizer, max_length, vocab, labels,
                                       split_long_sentences=False, do_one_hot=True):
        """
        :param sentence: A sentence to tokenize
        :param tokenizer: The tokenizer to use
        :param max_length: max_length for a sentence
        :param vocab: a list of vocabulary for the network to be used (e.g. ['O', '<PAD>', 'B-PER', ...]
        :param labels: labels (if any)
        :param split_long_sentences: whether to split long sentences or just ignore them
        :param do_one_hot: whether the labels are one_hot encodings
        :return: a list of tuples [(sentence, sentence_data), ...]
        """
        labels = labels if labels is not None else [None for _ in sentence]
        sentence_data = cls.get_sentence_data_list(sentence, tokenizer, max_length, vocab,
                                                   labels=labels, do_one_hot=do_one_hot)
        if sentence_data.fits:
            return [sentence_data]
        else:
            if split_long_sentences:
                LOGGER.info(f"Sentence too long, {len(sentence_data.ids)} > {max_length}, splitting!")
                halfway = len(sentence) // 2  # TODO: Maybe make something smarter here
                return cls.get_fitting_sentence_data_list(
                    sentence[:halfway], tokenizer, max_length, vocab,
                    labels=labels[:halfway],
                    split_long_sentences=split_long_sentences,
                    do_one_hot=do_one_hot
                ) + cls.get_fitting_sentence_data_list(
                    sentence[halfway:], tokenizer, max_length, vocab,
                    labels=labels[halfway:],
                    split_long_sentences=split_long_sentences,
                    do_one_hot=do_one_hot
                )
            else:
                LOGGER.info(f"Sentence too long, {len(sentence_data.ids)} > {max_length}, ignoring!")
                return []  # for now return nothing, as before

    def inverse_apply_labels(self, labels, mix_fun=None, trim_ends=False):
        """
        Applies labels for this sentence back to the words
        :param labels: A list of labels, one for each token in the sentence
        :param mix_fun: A function that takes a list of labels (one for each token of a word) as input and returns the
        "mixed" label, by default the mixing function return the first element of the list
        :param trim_ends: Whether to trim the ends ([CLS] and [SEP] labels)
        :return: A list of word, label pairs
        """
        if mix_fun is None:
            def mix_fun(label_list):
                return label_list[0]
        assert len(labels) in (len(self.labels), self.max_length), \
            f"The list of labels should either be as long as the number of tokens in the sentence, or the max_length " \
            f"for this datasupplier. Got a list of length {len(labels)}, it should be either " \
            f"{len(self.labels)} or {self.max_length}"
        i_token = 0
        word_labels = []
        for tokenized_word in self.tokenized:
            word_label_list = []
            for _ in tokenized_word.tokens:
                label = labels[i_token]
                word_label_list.append(label)
                i_token = i_token + 1
            if word_label_list:
                raw_mixed_labels = mix_fun(word_label_list)
                if self.do_one_hot:
                    mixed_labels = {self.idx2tag[i]: l for i, l in enumerate(raw_mixed_labels)}
                else:
                    mixed_labels = {tag: int(tag == raw_mixed_labels) for idx, tag in self.idx2tag.items()}
            else:
                mixed_labels = None
            word_labels.append((tokenized_word.word, mixed_labels))
        if trim_ends:
            word_labels = word_labels[1:-1]
        return word_labels


class InputData:
    def __init__(self, sentence_data: List[InputDataSentence], tokenizer, max_length: int,
                 do_one_hot=True):
        self.sentence_data = sentence_data

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ids = self._list2array([data.ids for data in self.sentence_data])
        self.masks = self._list2array([data.masks for data in self.sentence_data])
        self.input_types = self._list2array([data.input_types for data in self.sentence_data])
        self.is_heads = self._list2array([data.is_heads for data in self.sentence_data])

        nr_tags = self._safe_get_nr_tags(sentence_data)

        if not any(label is None for data in self.sentence_data for label in data.labels):
            # only perform list2array if there are not None labels
            self.labels = self._list2array([data.labels for data in self.sentence_data])
            if do_one_hot:
                self.labels = one_hot(self.labels, nr_tags)
        else:
            LOGGER.warning(f"Sentence labels contain None elements, will not create a label field.")
        self.size = self.ids.shape[0]

    @classmethod
    def from_gt_file(cls, file_path, tokenizer, max_length, do_one_hot, vocab, split_long_sentences):
        """
        Read a ground truth file into an InputData object
        :param file_path: the path of the file to read, should be tab separated, word\tBIO
        :param tokenizer: tokenizer for BERT
        :param max_length: the max length for a sequence
        :param do_one_hot: whether to use one_hot encoding
        :param vocab: a list of the vocab O, <PAD>, B-PER, ...
        :param split_long_sentences: whether to split long sentences into shorter ones or to ignore them
        :return: Class InputData object
        """
        with open(file_path, mode="r", encoding='ISO-8859-1') as f:
            sentences_with_labels = text_to_sentences(f.read())
        sentences, labels = split_sentence_labels(sentences_with_labels)

        sentence_data = []
        for sentence, labels in zip(sentences, labels):
            appenda = InputDataSentence.get_fitting_sentence_data_list(sentence, tokenizer, max_length, vocab,
                                                                       labels=labels,
                                                                       split_long_sentences=split_long_sentences)
            for appendum in appenda:
                sentence_data.append(appendum)
        return cls(sentence_data, tokenizer, max_length,
                   do_one_hot=do_one_hot)

    @classmethod
    def from_sentences(cls, sentences, tokenizer, max_length, vocab,
                       split_long_sentences=True, do_one_hot=True):
        """
        Read a ground truth file into an InputData object
        :param sentences: a list of list of words
        :param tokenizer: tokenizer for BERT
        :param max_length: the max length for a sequence
        :param do_one_hot: whether to use one_hot encoding
        :param vocab: a list of the vocab O, <PAD>, B-PER, ...
        :param split_long_sentences: whether to split long sentences into shorter ones or to ignore them
        :return:
        """
        sentence_data = []
        for sentence in sentences:
            appenda = InputDataSentence.get_fitting_sentence_data_list(
                sentence, tokenizer, max_length, vocab,
                labels=None,  # no labels!
                split_long_sentences=split_long_sentences
            )
            sentence_data += appenda
        return cls(sentence_data, tokenizer, max_length, do_one_hot=do_one_hot)

    @staticmethod
    def _safe_get_nr_tags(sentence_data: List[InputDataSentence]):
        nr_tags = len(sentence_data[0].tag2idx)
        for data in sentence_data[1:]:
            assert len(data.tag2idx) == nr_tags, f'Sentences in InputData object should all have same number of tags'
        return nr_tags

    def get_x(self):
        return [self.ids, self.masks, self.input_types, self.is_heads]

    def get_y(self):
        return self.labels

    def _list2array(self, list_of_lists, dtype='int32'):
        n = len(list_of_lists)
        array = np.zeros((n, self.max_length), dtype=dtype)
        for i, one_list in enumerate(list_of_lists):
            array[i][0:len(one_list)] = one_list
        return array

    def inverse_apply_labels(self, labelss, **kwargs):
        """
        Map a set of labels back onto the words in the sentences, basically wraps the method
        InputDataSentence.inverse_apply_labels
        :param labelss: a list of a list of labels
        :param kwargs: all kwargs are passed on to InputDataSentence.inverse_apply_labels
        :return: a list of lists of word, results pairs
        """
        assert len(labelss) == len(self.sentence_data), f'Length of results does not match length of sentence_data!\n' \
                                                        f'{len(labelss)} != {len(self.sentence_data)}'
        return [
            sentence_data.inverse_apply_labels(labels, **kwargs)
            for sentence_data, labels in zip(self.sentence_data, labelss)
        ]


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

    return sentences_with_labels


def split_sentence_labels(sentences_with_labels):
    sentences = [[word for word, label in swl] for swl in sentences_with_labels]
    labels = [[label for word, label in swl] for swl in sentences_with_labels]
    return sentences, labels


def get_max_label(labels):
    max_dict = max(labels.items(), key=lambda x: x[1])[0]
    return max_dict

