import os.path as op

from lib.ner_lists import ner_lists
from lib import constants

from tests import test_tools


class TestNerListsOnBertResults(test_tools.TestTools):
    @classmethod
    def setUpClass(cls):
        constants.DATA_DIR = test_tools.TEST_DATA_DIR
        ner_lists_config = {
            "data_dir": ["ner_lists", "SZSA"],
            "entity_types": ["location"],
            "cutoff_score": 0.85,
            "word_getter": "BERT"
        }
        cls.nl_location = ner_lists.NerLists(**ner_lists_config)
        ner_lists_config["entity_types"] = ["person"]
        cls.nl_person = ner_lists.NerLists(**ner_lists_config)

    def setUp(self):
        self.maxDiff = 1000

    def test_ner_list_on_bert_results_small_sample(self):
        self.out = self.nl_person(self.input)
        self.compare_output()

    def test_ner_list_on_bert_results_sample(self):
        self.out = self.nl_location(self.input)
        found_names = [list_label['list_name'] for list_label in self.out[0][34]['labels']['lists']]
        assert 'voc_opvarenden' not in found_names
        self.compare_output()
        # test if it knows both about person-results and location results after doing both
        self.out = self.nl_person(self.out)
        found_names = [list_label['list_name'] for list_label in self.out[0][34]['labels']['lists']]
        self.assertIn('atlas-mutual-heritage', found_names)
        self.assertIn('voc_loc', found_names)


if __name__ == "__main__":
    TestNerListsOnBertResults.setUpClass()
    tnl = TestNerListsOnBertResults(method_name='test_ner_list_on_bert_results_small_sample')
    tnl.test_ner_list_on_bert_results_small_sample()
