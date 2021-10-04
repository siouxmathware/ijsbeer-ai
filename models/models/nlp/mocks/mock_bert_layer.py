import os.path as op
from training_tests.data import test_data


class AssetPath:
    def __init__(self, value=None):
        self.value = value or op.join(test_data.TEST_DATA_DIR, 'bert_vocab.txt')

    def numpy(self):
        return self.value


class VocabFile:
    def __init__(self, vocab_file):
        self.asset_path = AssetPath(vocab_file)


class DoLowerCase:
    def __init__(self, value=True):
        self.value = value

    def numpy(self):
        return self.value


class ResolvedObject:
    def __init__(self, vocab_file=None, do_lower_case=True):
        self.vocab_file = VocabFile(vocab_file)
        self.do_lower_case = DoLowerCase(do_lower_case)


class MockBertLayer:
    def __init__(self, max_size=512):
        self.resolved_object = ResolvedObject()
        self.input_shape = [[0, max_size]]
        self.name = 'bert mocked'
