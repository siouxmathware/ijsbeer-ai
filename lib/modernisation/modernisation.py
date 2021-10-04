from typing import Dict, List, Tuple, Union
import re
import logging

from lib.modernisation.regex_rules import RegexRules
from lib.modernisation.syllable_corrector import SyllableCorrector


LOGGER = logging.getLogger(__name__)


class Modernisation:
    """
    Class responsible for translating historical Dutch into modern Dutch.
    It uses a variety of techniques.
    The goal is solely to aid of the user reading the historical texts.
    """
    def __init__(self,
                 modernisable_entities: Union[List[str], Tuple[str, ...]] = ("O", "B-time", "I-time"),
                 min_word_length=3
                 ):
        """
        :param modernisable_entities: Which entity types can be modernised.
        :param min_word_length: Minimum length of words for which to apply full modernisation. Shorter words are only
            matched against line-break and abbreviation patterns.
        """
        self.regex_rules = RegexRules()
        self.syllable_corrector = SyllableCorrector()
        self.modernisable_entities = modernisable_entities
        self.word_part_split_pattern = re.compile("([ :;.,])")
        self.min_word_length = min_word_length

    def __call__(self, sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words
        :return: The same list of lists, with the fields "remove_whitespace_for_modernisation" and "modernisation" added
            to each word. The former is currently unused, but could be used if modernisation is extended to consider
            multiple words.
        """
        if not all('bio' in word for sentence in sentences for word in sentence if word['ner']):
            LOGGER.info('Not all NER words contained the "bio" key, these words will be modernised.')

        for sentence in sentences:
            for word in sentence:
                word['remove_whitespace_for_modernisation'] = False  # TODO: is this still a necessary property
                word['modernisation'] = self._per_word(word)

        return sentences

    def _per_word(self, word: Dict):
        """
        :param word: The word dict to be modernised. The dictionary is used to determine if a word is part of an entity
            that should not be modernised, i.e. part of a person or location name.
        :return: Modernised form of the word. Capitalization of the first letter is kept as-is, other characters will
            all be lowercase.
        """
        if word['ner']:
            if word.get('bio', 'O') not in self.modernisable_entities:
                # do not modernize words that are recognized as (parts of) entities
                return word['post_correction']
        word_pc = word['post_correction']
        captialized = word_pc[0].isupper()
        modernized_lower: str = self._per_word_lower(word_pc.lower())
        return modernized_lower.capitalize() if captialized else modernized_lower

    def _per_word_lower(self, word_lower: str):
        """
        :param word_lower: Lower case form of the word to be modernized
        :return: Modernised form of the word
        """
        word_lower, _ = self.regex_rules.regex_subs(word_lower, 'line_breaks')
        word_lower, touched = self.regex_rules.regex_subs(word_lower, 'abbreviations')
        if touched or len(word_lower) < self.min_word_length:
            return word_lower
        parts = self.word_part_split_pattern.split(word_lower)
        return ''.join(self._per_part(part) for part in parts)

    def _per_part(self, part: str):
        """
        :param part: A true "word" part of each word element.
        :return: The modernised form for this word part. Modernisation is done by:

            1. Dict lookup through known dictionaries

            2. The :ref:`SyllableCorrector`

            3. Heuristic regular expression substitutions.

            If the word is changed by any of the steps, subsequent steps are not executed.
        """
        # each of the calls below return the form of word, potentially changed, and an indicator whether it was
        # "touched" or not. If the words was "touched" it means it was either modified or determined to be in its
        # correct form already. The "common_operations" regexs are exempt from this, some of them are only starting
        # points, at least for now...
        word_dict_lookup, touched = self.regex_rules.dict_lookup(part)
        if touched:
            return word_dict_lookup
        word_syll, touched = self.syllable_corrector(part)
        if touched:
            return word_syll
        return part


if __name__ == "__main__":
    m = Modernisation()
    print(m._per_word({"ner": True, "post_correction": "sijnde"}))
