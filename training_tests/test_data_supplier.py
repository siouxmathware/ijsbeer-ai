import os.path as op
import unittest
import numpy as np

from training_tests.old_data_supplier import DataSupplier as OldDataSupplier
from data_suppliers.text.nerbio.data_supplier import DataSupplier
from data_suppliers.text.nerbio.data_classes import InputDataSentence
from models.nlp.mocks.mock_bert_layer import MockBertLayer


THIS_DIR = op.dirname(__file__)


class TestDataSupplier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bert_layer = MockBertLayer()

        data_path = op.join(THIS_DIR, '..', 'data', 'input', 'smaller')
        cls.old_data_supplier = OldDataSupplier(bert_layer, data_path, 5)
        cls.data_supplier = DataSupplier(bert_layer, data_path, 5, split_long_sentences=False)
        cls.data_supplier_split = DataSupplier(bert_layer, data_path, 5, split_long_sentences=True)

    def testSplitSupplier(self):
        self.data_supplier_split()

    def testDataSupplier(self):
        old_data = self.old_data_supplier()
        data = self.data_supplier()

        self.assertEqual(old_data.keys(), data.keys())
        for key, old_val in old_data.items():
            val = data[key]
            if isinstance(val, list) or isinstance(val, tuple):
                for i, (old_el, el) in enumerate(zip(old_val, val)):
                    self.assertTrue(np.allclose(old_el, el))
            else:
                self.assertTrue(np.allclose(old_val, val))

    def testLabelInversion(self):
        sentence = 'This sentence is not trivially tokenizable'.split(' ')
        tokenizer = self.data_supplier.tokenizer
        max_length = 512
        test_vocab = ['<PAD>', 'O'] + [f'{bi}-CAT{i}' for i in range(50) for bi in "BI"]
        test_input_labels = test_vocab[10:10+len(sentence)]
        sentence_data = InputDataSentence.get_sentence_data_list(sentence, tokenizer, max_length,
                                                                 labels=test_input_labels,
                                                                 vocab=test_vocab,
                                                                 do_one_hot=False)
        # self.assertGreater(len(sentence_data.labels), len(test_input_labels))
        tags = [sentence_data.idx2tag[idx] for idx in sentence_data.labels]
        inverted_labels = sentence_data.inverse_apply_labels(tags,
                                                             trim_ends=True)
        self.assertEqual([dict_from_label(test_vocab, i) for i in test_input_labels],
                         [label[1] for label in inverted_labels])

        test_result_labels = test_vocab[20:20+len(sentence_data.labels)]
        inverted_labels = sentence_data.inverse_apply_labels(test_result_labels,
                                                             mix_fun=lambda x: x[-1],
                                                             trim_ends=False)
        self.assertEqual(dict_from_label(test_vocab, test_result_labels[-2]), inverted_labels[-2][1])


def dict_from_label(vocab, label):
    d = {tag: 0 for tag in vocab}
    d[label] = 1
    return d


if __name__ == "__main__":
    TestDataSupplier.setUpClass()
    tdg = TestDataSupplier()
    tdg.testDataSupplier()
