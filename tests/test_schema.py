import unittest
import os.path as op
import json
import jsonschema
import copy

from lib import schema


TEST_DATA_DIR = op.join(op.dirname(__file__), 'test_data')

unittest.TestLoader.sortTestMethodsUsing = None  # to test the tests in the order I write them


class TestSchema(unittest.TestCase):
    def setUp(self):
        with open(op.join(TEST_DATA_DIR, 'format_proposal_20210315.json'), 'r') as f:
            self.base_json_data = json.load(f)

    def testSchema(self):
        jsonschema.Draft7Validator.check_schema(schema.Validator().schema)
        schema.Validator()(self.base_json_data)

    def testNullChars(self):
        json_data = copy.deepcopy(self.base_json_data)
        json_data[0][0]['begin_char'] = None
        json_data[0][0]['end_char'] = None
        schema.Validator()(json_data)

    def testBadLabels(self):
        self._testBadOrNoValue((0, 0, 'labels', 'BERT'), 'X')
        self._testBadOrNoValue((0, 0, 'labels', 'lists'), 'X')
        self._testBadOrNoValue((0, 0, 'labels', 'lists', 0), 'X', test_no_value=False)

    def testBadBio(self):
        self._testBadOrNoValue((0, 0, 'labels', 'BERT', 'location', 'bio'), 'X')
        self._testBadOrNoValue((0, 0, 'labels', 'BERT', 'location', 'label_probabilities'), 'X')
        self._testBadOrNoValue((0, 0, 'labels', 'BERT', 'location', 'label_probabilities', '<PAD>'), 'X')
        self._testBadOrNoValue((0, 0, 'labels', 'lists', 0, 'bio'), 'X')

    def _testBadOrNoValue(self, keys, bad_value, test_no_value=True):
        json_data = copy.deepcopy(self.base_json_data)
        el = json_data
        for key in keys[:-1]:
            el = el[key]
        # test bad_value
        el[keys[-1]] = bad_value
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            schema.Validator()(json_data)
        # test no value
        if test_no_value:
            del el[keys[-1]]
            with self.assertRaises(jsonschema.exceptions.ValidationError):
                schema.Validator()(json_data)


if __name__ == "__main__":
    tp = TestSchema()
    tp.setUp()
    tp.testSchema()
    tp.testBadBio()
    tp.testBadLabels()
