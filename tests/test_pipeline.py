import unittest
import os.path as op
import json

from lib.pipeline import Pipeline
from lib import schema
from lib import constants

from tests import test_tools


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls, berts_to_initialize='mock', run=True):
        # constants.DATA_DIR = test_tools.TEST_DATA_DIR
        # some unconventional argument handling as the default cannot be set to None as
        # the default test behaviour is different to default run behaviour.
        if berts_to_initialize == 'empty':
            cls.berts_to_initialize = {}
        elif berts_to_initialize == 'mock':
            cls.berts_to_initialize = {'mock': ('person', 'location', 'time')}
        elif berts_to_initialize == 'test':
            cls.berts_to_initialize = {'40-10-1-split-minus-137-fixed': ('person', 'location', 'time')}
        else:
            raise ValueError
        with open(op.join(test_tools.TEST_DATA_DIR, 'AN_disk1_ZIPs_7538_alto.txt')) as f:
            data = f.readlines()
        cls.input_string = ''.join(data[:100000])
        with open(op.join(test_tools.TEST_DATA_DIR, 'server', 'config_test.json')) as f:
            config = json.load(f)
        config['ner_bert']['berts_to_use'] = cls.berts_to_initialize
        cls.pipeline = Pipeline(config)
        if run:
            for _ in range(10):
                cls.result = cls.pipeline(cls.input_string)

    def testRepetition(self,):
        result_repeated = self.pipeline(self.input_string)
        self.assertEqual(self.result, result_repeated)

    def testSteps(self):
        """
        Test whether the result is equal when performing all steps separately as when running the whole thing.
        :return:
        """
        sts_result = self.pipeline(self.input_string, steps=('string_to_sentences',))
        schema.Validator(ignore=('post_correction', 'ner_bert', 'ner_lists', 'modernisation'))(sts_result)

        post_correction_result = self.pipeline(sts_result, steps=('post_correction',))
        schema.Validator(ignore=('ner_bert', 'ner_lists', 'modernisation'))(post_correction_result)

        ner_bert_result = self.pipeline(post_correction_result, steps=('ner_bert',))
        schema.Validator(ignore=('ner_lists', 'modernisation'))(ner_bert_result)

        ner_lists_result = self.pipeline(ner_bert_result, steps=('ner_lists',))
        schema.Validator(ignore=('modernisation',))(ner_lists_result)

        modernisation_result = self.pipeline(ner_lists_result, steps=('modernisation',))
        schema.Validator()(modernisation_result)

        self.assertEqual(self.result, modernisation_result)

    def testZZSchema(self):
        # this is done last as it modifies the result
        self.verify_schema(self.berts_to_initialize, self.result)

    @staticmethod
    def verify_schema(berts_to_initialize, result):
        if berts_to_initialize == {}:
            # Remove the empty BERT dict from the result for online testing with no BERTs
            # Note that this is different to the "ignore" kwarg for the Validator, as that only changes what is required
            # When the pipeline is run with no BERTs, the BERT dict exists, but it is empty
            for sentence in result:
                for word in sentence:
                    if 'labels' in word.keys():
                        del word['labels']['BERT']
            schema.Validator(ignore=('ner_bert',))(result)
        else:
            schema.Validator()(result)


if __name__ == "__main__":
    TestPipeline.setUpClass(berts_to_initialize='mock', run=True)
    tp = TestPipeline()
    # tp.testSteps()
