import os.path as op
import string

TEST_DATA_DIR = op.join(op.dirname(__file__), '..', 'data')


class MockTokenizer:
    def __init__(self):
        self.normal = [c for c in string.ascii_lowercase]
        self.special = ['<PAD>', '[CLS]', '[SEP]']
        self.unknown = '[UNK]'
        self.ids = {token: i for i, token in enumerate(self.normal + self.special + [self.unknown])}

    def tokenize(self, word):
        if word in self.special:
            return [self.special[word]]
        else:
            return [letter if letter in self.ids else self.unknown for letter in word]

    def convert_tokens_to_ids(self, tokens):
        return [self.ids[token] for token in tokens]
