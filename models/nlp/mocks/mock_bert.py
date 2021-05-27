from models.nlp.mocks import mock_bert_layer, mock_tokenizer
import numpy as np


class MockHistory:
    def __init__(self, categories):
        all_categories = categories + ['loss']
        all_categories = all_categories + [f'val_{cat}' for cat in all_categories]
        self.history = {cat.lower(): [i] for i, cat in enumerate(all_categories)}


class MockBertModel:
    def __init__(self, nr_categories, config, bert_trainable, topping_sizes):
        self.nr_categories = nr_categories
        self.nr_tags = 2*nr_categories + 1 + 1
        self.config = config
        self.bert_trainable = bert_trainable
        self.topping_sizes = topping_sizes
        self.layers = [mock_bert_layer.MockBertLayer()]
        self.output_shape = [self.nr_tags]
        self.loss = None
        self.optimzer = None
        self.metrics = None
        self.tokenizer = mock_tokenizer.MockTokenizer()
        self.max_length = 512

    def summary(self, print_fn=None):
        print_fn = print_fn if print_fn is not None else print
        print_fn(f'This is a mocked BERT with {self.nr_categories} categories')

    def compile(self, loss, optimizer, metrics, **kwargs):
        self.loss = loss
        self.optimzer = optimizer
        self.metrics = metrics

    def fit(self, **kwargs):
        print("TRAINING the mocked BERT")
        history = MockHistory([self.get_name(metric) for metric in self.metrics])
        return history

    @staticmethod
    def get_name(metric):
        if isinstance(metric, str):
            return metric
        else:
            return metric.name

    def __call__(self, x):
        tokens = x[0]
        result = np.zeros(tokens.shape + (self.nr_tags,))
        for i, row in enumerate(tokens):
            result[i, :, i % self.nr_tags] = 1
        return result


def get_bert_ner_model(nr_categories, config, bert_trainable, topping_sizes):
    model = MockBertModel(nr_categories, config, bert_trainable, topping_sizes)
    return model
