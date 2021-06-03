import unittest
import os.path as op
import pandas as pd

from lib.string_to_sentences import string_to_sentences
from lib.post_correction import post_correction
from lib.modernisation import modernisation
from lib import schema
from lib import constants

from tests import test_tools


class WordMaker:
    def __init__(self):
        self.i = 0

    def __call__(self, text, **kwargs):
        begin = self.i
        self.i += len(text)
        d = {'word': text,
             'post_correction': text,
             'begin_char': begin,
             'end_char': self.i,
             'ner': True}
        d = {**d, **kwargs}
        return d


class TestModernisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        constants.DATA_DIR = test_tools.TEST_DATA_DIR
        pass

    def testWithPC(self):
        text = """Met d'E: Agtb: Philip, Christian enz; naar Amsterdam etc;."""
        ref = """Met| |de edele| |Achtbare| |Philip,| |Christian| |enz.| |naar| |Amsterdam| |etc."""
        sentences = string_to_sentences.StringToSentences()(text)
        corrected = post_correction.PostCorrection()(sentences)
        modernised = modernisation.Modernisation()(corrected)
        self.assertEqual(ref, '|'.join(w['modernisation'] for w in modernised[0]))

    def testLineBreaks(self):
        text = """wandel„
„de, Re„
gent, mo-
dern, spatie 
-rijk, zon
der, word-
indict, sllbl-
wrd, regex-
word"""
        ref_corrected = """wandel„
„de,| |Re„
gent,| |mo-
dern,| |spatie 
-rijk,| |zon|
|der,| |word-
indict,| |sllbl-
wrd,| |regex-
word"""
        ref_modern = """wandelde,
| |Regent,
| |modern,
| |spatierijk,
| |zon|
|der,| |wordindictfound,
| |syllableword found,
| |regexword
"""
        sentences = string_to_sentences.StringToSentences()(text)
        corrected = post_correction.PostCorrection()(sentences)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(ref_corrected, '|'.join(w['post_correction'] for w in corrected[0]))
        mod = modernisation.Modernisation()
        modernised = mod(corrected)
        self.assertEqual(len(modernised), 1)
        self.assertEqual(ref_modern, '|'.join(w['modernisation'] for w in modernised[0]))

    def testSample(self):
        word_maker = WordMaker()
        sentences = [
            ['wordindict', 'sllblwrd', 'regexword'],
            ['Wordindict', 'Sllblwrd', 'Regexword'],
            ['sijn', 'schijnen', 'zijn', 'zijnde'],
            ['Comp:', 'Comp:s', 'Comp=s', 'agtb:', 'agtb=re', 'E:', 'Ed=s'],
            ['aan \n„thonen', 'aan\n„thonen', 'Jndia', '`t', 'deses'],
            ['verzocht', ',', 'duizend', ',', 'ventiel', ';', 'd\'E:'],
            ['niet', 'nog', 'zal', 'wij', 'alhier', 'andere', 'ik', 'deze', 'maar', 'uit', 'oude', 'quod', 'goed'],
            ['A:o', 'Comp.', '9:e', 'Ed:l', 'ed:n'],
            ['Achtb=re', 'Agtb=re'],
        ]
        ref_sentences = [
            'wordindictfound| |syllableword found| |regexword| ',
            'Wordindictfound| |Syllableword found| |Regexword| ',
            'zijn| |schijnen| |zijn| |zijnde| ',
            'Comp.| |Comps.| |Comps.| |achtbare| |achtbare| |Edele| |Edelen| ',
            'aantonen\n| |aantonen\n| |India| |het| |van deze| ',
            'verzocht| |,| |duizend| |,| |ventiel| |;| |de edele| ',
            'niet| |nog| |zal| |wij| |alhier| |andere| |ik| |deze| |maar| |uit| |oude| |wat| |goed| ',
            'Anno| |Comp.| |9e| |Edele| |edelen| ',
            'Achtbare| |Achtbare| ',
        ]

        sentences = [[word_maker(w) for word in sentence for w in (word, ' ')] for sentence in sentences]
        res_sentences = modernisation.Modernisation()(sentences)
        for ref_s, res_s in zip(ref_sentences, res_sentences):
            self.assertEqual(ref_s, '|'.join(w['modernisation'] for w in res_s))
        validator = schema.Validator(ignore=('ner_bert', 'ner_lists'))
        validator(res_sentences)

    def testScores(self):
        def flat_score(df_s):
            return df_s['correct'].sum() / df_s['false'].sum()

        word_maker = WordMaker()

        df = pd.read_csv(op.join(test_tools.TEST_DATA_DIR, 'dictionary_old_new_dutch.txt'), sep='@',
                         header=None, keep_default_na=False)
        df.columns = ['historic', 'modern']
        sentences = [
            [word_maker(word[1:-1]) for word in df['historic']],
            [word_maker(word) for word in df['modern']]
        ]
        moderniser = modernisation.Modernisation()
        df_score = self._computeScore(moderniser, sentences)
        baseline = flat_score(df_score)
        print(f'Baseline result is {baseline}')
        self.assertGreaterEqual(baseline, 1.985)

    @staticmethod
    def _computeScore(moderniser, sentences):
        ress = moderniser(sentences)
        df_score = pd.DataFrame(columns=['correct', 'false'], index=['historic', 'modern'], data=0)
        for historic_w, modern_w in zip(*ress):
            if historic_w['modernisation'] == modern_w['word']:
                df_score.loc['historic', 'correct'] += 1
            else:
                df_score.loc['historic', 'false'] += 1
            if modern_w['modernisation'] == modern_w['word']:
                df_score.loc['modern', 'correct'] += 1
            else:
                df_score.loc['modern', 'false'] += 1
        return df_score

    def testNotModernizeEntities(self):
        if constants.DATA_DIR != test_tools.TEST_DATA_DIR:
            # do not build a skip, this should not happen, but is probably not a big problem
            print("This test will only work on test data")
        wm = WordMaker()
        pairs = [
            ('wordindict', 'O'),
            ('wordindict', 'B'),
            ('wordindict', 'O'),
            ('wordindict', 'I'),
            ('wordindict', 'O'),
        ]
        sentences = [[wm(word, bio=bio) for word, bio in pairs]]
        mod = modernisation.Modernisation()
        mod(sentences)
        for word, pair in zip(sentences[0], pairs):
            # test that a word is modernised IFF it is not an entity
            self.assertEqual(pair[1] == 'O', word['post_correction'] != word['modernisation'])
        pass


if __name__ == "__main__":
    TestModernisation.setUpClass()
    tp = TestModernisation()
    tp.testLineBreaks()
    pass
