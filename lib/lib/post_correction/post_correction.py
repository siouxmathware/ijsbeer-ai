"""
This module is responsible for 3 things:
- Split a string into words and sentences
- Fix mistakes made by OCR
- Indicate per word if it is suitable for applying named entity recognition (ner)
"""
from typing import List, Dict

from lib.post_correction.freq_table_clean_up import FreqTableCleanUp


class PostCorrection:
    """
    Responible for the correction of mistakes by the Handwritten Text Recognition (HTR) input.
    """
    def __init__(self, **_):
        """
        Initializes the class' :class:`FreqTableCleanUp <lib.post_correct.freq_table_cleanup.FreqTableCleanUp>` object
        :param _: Unused kwarg argument, included for symmetry with the other pipeline steps.
        """
        self.freq_table_clean_up = FreqTableCleanUp('post_correction')

    def __call__(self, sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words
        :return:  The sentences where each word has been provided with a "post_correction" key containing the corrected
            form of the word.
        """

        for sentence in sentences:
            for word in sentence:
                word['post_correction'] = self.freq_table_clean_up(word['word'])

        return sentences


class DummyPostCorrection:
    """
    If no correction is required, the "post_correction" key must still be applies as it is used as the input for BERT.
    """
    def __init__(self):
        """
        Empty initializer.
        """
        pass

    def __call__(self, sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words
        :return:  The sentences where each word has been provided with a "post_correction" key containing the **same**
            form of the word.
        """

        for sentence in sentences:
            for word in sentence:
                word['post_correction'] = word['word']

        return sentences
