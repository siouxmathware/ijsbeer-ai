import os.path as op
import unittest
import numpy as np

from data_suppliers.text.nerbio.data_supplier import DataSupplier
from data_suppliers.text.nerbio.data_generator import DataGenerator
from data_suppliers.text.nerbio.data_generator_chunk import DataGeneratorChunk
from models.nlp.mocks.mock_bert_layer import MockBertLayer


THIS_DIR = op.dirname(__file__)


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = op.join(THIS_DIR, '..', 'data', 'input', 'smaller')
        cls.batch_size = 5
        cls.bert_layer = MockBertLayer()
        cls.data_supplier = DataSupplier(cls.bert_layer, data_path, cls.batch_size, split_long_sentences=True)
        cls.data_generator = DataGenerator(cls.bert_layer, data_path, cls.batch_size, split_long_sentences=True)
        cls.data_generator_chunk = DataGeneratorChunk(cls.bert_layer, data_path, cls.batch_size)

        data_path = op.join(THIS_DIR, '..', 'data', 'input', 'second')
        cls.big_data_generator = DataGenerator(cls.bert_layer, data_path, cls.batch_size,
                                               dataset_types=('train', 'validation'), split_long_sentences=True)

    def testLengths(self):
        self.assertEqual(self.big_data_generator.nr_batches['train'], 314)

    def testEqual(self):
        sup_batcher = self.data_supplier.batcher('train')

        sup_x, sup_y, sup_sentences_data = next(sup_batcher)
        data_generator = self.data_generator.generator('train')
        gen_data = next(data_generator)
        gen_x = gen_data.get_x()
        gen_y = gen_data.get_y()
        gen_sentences_data = gen_data.sentence_data

        for sup_x_el, gen_x_el in zip(sup_x, gen_x):
            self.assertTrue(np.allclose(sup_x_el, gen_x_el[:self.batch_size]))
        self.assertTrue(np.allclose(sup_y, gen_y[:self.batch_size]))
        self.assertEqual(' '.join(sup_sentences_data[0].sentence), ' '.join(gen_sentences_data[0].sentence))

    def testTwoRuns(self):
        generator = self.big_data_generator.train_generator()
        for i in range(2 * self.big_data_generator.nr_batches['train']):
            next(generator)

    def testDataCorrectness(self):
        for dataset_type in self.big_data_generator.dataset_types:
            sentence_generator = self.big_data_generator.sentences_generator(dataset_type)
            sentence = '<start>'
            for i in range(
                    2 * self.big_data_generator.nr_batches[dataset_type] * self.big_data_generator.batch_size
            ):
                try:
                    sentence = next(sentence_generator)
                except AssertionError:
                    print(sentence[-1])

    def testChunk(self):
        generator = self.data_generator.generator('train')
        data0 = next(generator)
        generator_chunk = self.data_generator_chunk.generator('train')
        data1 = next(generator_chunk)
        data2 = next(generator_chunk)
        ids0, masks0, types0, heads0 = tuple(data0.get_x())
        ids1, masks1, types1, heads1 = tuple(data1.get_x())
        ids2, masks2, types2, heads2 = tuple(data2.get_x())
        ids0_test = ids0[masks0 > 0]
        ids1_test = ids1[::2][masks1[::2] > 0]  # select only even rows as odd ones are offsets!
        n_test = min(len(ids0_test), len(ids1_test))
        equals: np.ndarray = ids0_test[:n_test] == ids1_test[:n_test]
        self.assertTrue(equals.all())
        equals: np.ndarray = ids0 == ids1
        self.assertFalse(equals.all())
        equals: np.ndarray = ids1 == ids2
        self.assertFalse(equals.all())


if __name__ == "__main__":
    TestDataGenerator.setUpClass()
    tdg = TestDataGenerator()
    tdg.testDataCorrectness()
