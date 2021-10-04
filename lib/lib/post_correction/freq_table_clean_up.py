import logging
import os.path as op
import re

import lib.post_correction.spellchecker_local as spellchecker

from lib import constants
LOGGER = logging.getLogger(__name__)


TABLES = {
    'post_correction': {
        'file': 'tag_de_tekst_dict_for_pyspellchecker.json.gz',
        'threshold': 15,
        'min_length': 4,
    },
}


class FreqTableCleanUp:
    """
    Loads a pyspellchecker dictionary and clean up words with this dict. Current implementation is only used for fixing
    transcription errors in the input. Using it to also improve modernisation of words has proven ineffective.
    """
    def __init__(self, table_name: str):
        """
        :param table_name: Name of frequency table to use. Currently only post_correction is available.
        :type table_name: str
        """
        self.word_part_split_pattern = re.compile('([ :;.,])')

        LOGGER.info(f"Initializing spellchecker for {table_name}")
        table = TABLES[table_name]
        local_dict_path = op.join(constants.DATA_DIR, 'post_correction', table['file'])
        self.spell = spellchecker.SpellChecker(distance=1, case_sensitive=False, local_dictionary=local_dict_path)
        if self.spell.word_frequency.total_words == 0:
            LOGGER.error(f"Cannot initialize frequency table, probably missing file {local_dict_path}")
        """
        LD=1 and threshold=15 for cut-off can be adapted, set to 1 for now.
        Input should be a word.
        """
        self.threshold = table['threshold']
        self.min_length = table['min_length']

        if self.threshold:
            self.spell.word_frequency.remove_by_threshold(threshold=self.threshold)

    def __call__(self, word_form: str):
        """
        :param word_form: The word form to clean, i.e. passing "Weerld!" could result in "Wereld!"
        :type word_form: str
        :return: "cleaned" form of the word, i.e. a close match from the frequency table if that word is more likely.
        """
        parts = self.word_part_split_pattern.split(word_form)
        return ''.join(self._per_part(part) for part in parts)

    def _per_part(self, word_part: str):
        if len(word_part) > 4 and not word_part[0].isupper():
            return self.spell.correction(word_part)
        else:
            return word_part
