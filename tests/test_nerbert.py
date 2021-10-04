import os.path as op

from lib.ner_bert import ner_bert
from lib import constants

from tests import test_tools


class TestNerBert(test_tools.TestTools):
    @classmethod
    def setUpClass(cls):
        constants.DATA_DIR = test_tools.TEST_DATA_DIR
        cls.ner_bert = ner_bert.MultipleBerts(
            berts_to_use={'mock': ['person', 'location', 'time']},
        )

    def setUp(self):
        self.maxDiff = 1000

    def test_add_entity_be_chars(self):
        self.out = []
        for words in self.input:
            arg = [w.copy() for w in words]
            self.ner_bert._add_entity_be_chars(arg)
            self.out.append(arg)
        self.compare_output()


if __name__ == "__main__":
    TestNerBert.setUpClass()
    tnl = TestNerBert(method_name='test_add_entity_be_chars')
    tnl.test_add_entity_be_chars()
