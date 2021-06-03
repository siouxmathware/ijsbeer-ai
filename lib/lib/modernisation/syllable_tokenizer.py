import os.path as op
import re
from typing import List, Pattern


class SyllableTokenizer:
    """
    Splits words into syllables based on some basic rules from (modern) Dutch.
    Four basic rules are followed:

    1. Two vowels are separated by a single consonant: break before that consonant

    2. Two vowels are separated by a multiple consonants: break after the first consonant

    3. Compound words must be split according to their components, e.g. broodoven (bread oven) becomes brood.oven and
       not broo.doven as rule 1. would suggest. *This rule is currently not implemented.*

    4. The syllable formed must be pronouncable. This rule is implemented by modifying rule 2. to only allow syllables
       to start with combinations of consonants that are considered pronouncable in Dutch. This list is given by
       :download:`syll_starts.txt </../../lib/modernisation/syll_starts.txt>`. The list was formed by studying all
       consonant combinations that exist at the beginning of Dutch words from a frequency table.

    See `www.dutchgrammar.com <https://www.dutchgrammar.com/nl/?n=SpellingAndPronunciation.05>`_ for more elaborate
    examples.
    """

    SAFETY_PATTERN = re.compile("^[a-z ]+$")
    IJ_PATTERN = re.compile('ij')
    IJ_SUB_PATTERN = re.compile('ÿ')

    def __init__(self, pad: str = '<PAD>', start: str = '[CLS]', end: str = '[SEP]', unk: str = '[UNK]'):
        """
        :param pad: Representation of the pad-token
        :param start: Representation of the start-token
        :param end: Representation of the end-token
        :param unk: Representation for any unknown token
        """
        self.pad = pad
        self.start = start
        self.end = end
        self.unk = unk
        self.rules = self._get_rules()

    def encode(self, word: str):
        """
        :param word: Word to be split
        :return: List of tokens for the word
        """
        return [token for token in self._split_word(word, self.rules)]

    def decode(self, tokens: List[str]):
        """
        :param tokens: Tokens of the word
        :return: Word formed from the tokens
        """
        return ''.join(token for token in tokens)

    @staticmethod
    def _get_rules():
        ks = r"[aeiouyÿ]"
        mks = r"[^aeiouyÿ]"

        with open(op.join(op.dirname(__file__), 'syll_starts.txt')) as f:
            syll_starts = [line.strip() for line in f.readlines() if line]
        combos = '|'.join(f'({valid})' for valid in syll_starts)
        rules = [
            r"(?<=\w)()(?=\W)",  # before a space
            r"(?<=\W)()(?=\w)",  # after a space
            f"(?:(?:uw)|(?:ÿ)|(?:{mks}y))()(?={ks})",  # for kauw-en, rij-en, par-ty-en
            f"(?:{ks})({mks})(?={ks})",  # for ra-ken
            f'(?:{ks})(?:{mks}+?)({combos})(?={ks})',
            f'(?:{ks})(?:{mks}+)({mks})(?={ks})'
        ]
        return [re.compile(rule) for rule in rules]

    @staticmethod
    def _safety_check(word: str):
        return SyllableTokenizer.SAFETY_PATTERN.match(word) is not None

    @staticmethod
    def _split_piece(rule: Pattern, word_piece: str):
        # if len(word_piece) <= 2:
        #     # Do not do anything on short word pieces, they cannot be split anyway.
        #     return [word_piece]
        ids = [None] + [match.regs[1][0] for match in rule.finditer(word_piece)] + [None]
        return [word_piece[id0:id1] for id0, id1 in zip(ids[:-1], ids[1:])]

    @staticmethod
    def _split_word(word: str, rules: List[Pattern]):
        word = SyllableTokenizer.IJ_PATTERN.sub('ÿ', word)
        word_pieces = SyllableTokenizer._split_word_rec(word, rules)
        return [SyllableTokenizer.IJ_SUB_PATTERN.sub('ij', wp) for wp in word_pieces]

    @staticmethod
    def _split_word_rec(word_piece: str, rules: List[Pattern]):
        if not rules or len(word_piece) <= 2:
            return [word_piece]
        curr_rule = rules[0]
        curr_pieces = SyllableTokenizer._split_piece(curr_rule, word_piece)
        rec_pieces = [
            piece for curr_piece in curr_pieces for piece in SyllableTokenizer._split_word_rec(curr_piece, rules[1:])
        ]
        return rec_pieces
