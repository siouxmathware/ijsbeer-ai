import unittest

from lib.post_correction import post_correction
from lib.string_to_sentences import string_to_sentences
from lib import schema
from lib import constants

from tests import test_tools


class TestPostCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        constants.DATA_DIR = test_tools.TEST_DATA_DIR
        pass

    def testStringToSentences(self):
        text = 'Hello J. Jacobs! This is a new sen-\ntence. This is too, but this is not. A:o 2020 maybe. This: Too?'
        ref_sentences = ['Hello| |J.| |Jacobs|!| ',
                         'This| |is| |a| |new| |sen-\ntence|.| ',
                         'This| |is| |too,| |but| |this| |is| |not|.| ',
                         'A:o| |2020| |maybe|.| ',
                         'This:| |Too|?']
        res_sentences = string_to_sentences.StringToSentences()(text)
        for ref_s, res_s in zip(ref_sentences, res_sentences):
            self.assertEqual(ref_s, '|'.join(w['word'] for w in res_s))
        self.assertEqual(len(ref_sentences), len(res_sentences))
        # self.assertEqual(res_sentences[1][-3]['end_char'], 39, 'Final character after line break incorrect!')

    def testSample(self):
        text = 'Dit woord is A:o 2020 fout: vergaene. Tweede zin.'
        ref_sentences = ['Dit woord is A:o 2020 fout: vergaene. ', 'Tweede zin.']
        sentences = string_to_sentences.StringToSentences()(text)
        res_sentences = post_correction.PostCorrection()(sentences)
        for ref_s, res_s in zip(ref_sentences, res_sentences):
            self.assertEqual(ref_s, ''.join(w['post_correction'] for w in res_s))
        validator = schema.Validator(ignore=('modernisation', 'ner_bert', 'ner_lists'))
        validator(res_sentences)


if __name__ == "__main__":
    TestPostCorrection.setUpClass()
    tp = TestPostCorrection()
    tp.testSample()
