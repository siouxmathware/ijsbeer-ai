import os.path as op

from lib.ner_lists import ner_lists
from lib import constants

from tests import test_tools


class TestNerLists(test_tools.TestTools):
    @classmethod
    def setUpClass(cls):
        constants.DATA_DIR = test_tools.TEST_DATA_DIR
        ner_lists_config = {
            "data_dir": ["ner_lists", "NA"],
            "entity_types": ["person", "location", "date"],
            "cutoff_score": 0.92,
            "word_getter": "NER"
        }
        cls.ner_lists = ner_lists.NerLists(**ner_lists_config)

    def setUp(self):
        self.maxDiff = 1000

    def test_ner_list_small_sample(self):
        self.out = self.ner_lists(self.input)
        self.compare_output()

    def test_ner_list_sample(self):
        self.out = self.ner_lists(self.input)
        self.compare_output()

    def test_ner_false_word(self):
        # test that a list entity is correctly found if there is a ner: False element in between.
        out = self.ner_lists(self.input)[0]
        self.assertEqual(out[0]['labels']['lists'][0]['bio'], 'B')
        self.assertEqual(len(out[1]), 2)  # assert that not extra keys are given to the ner: False word
        self.assertEqual(out[2]['labels']['lists'][0]['bio'], 'I')


if __name__ == "__main__":
    TestNerLists.setUpClass()
    tnl = TestNerLists(method_name='test_ner_list_small_sample')
    tnl.test_ner_list_small_sample()
