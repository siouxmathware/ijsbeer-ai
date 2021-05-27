import os.path as op
from typing import List, Dict, Union

from lib.ner_lists.direct_finder import DirectFinder
from lib.ner_lists.fuzzy_permutative_finder import FuzzyPermutativeFinder
from lib.ner_lists.fuzzy_finder import FuzzyFinder


class NerLists:
    """
    Find entities by comparing words with lists
    """
    def __init__(self, data_dir: Union[List[str], str], word_getter: str, cutoff_score: float, entity_types: List[str]):
        """
        :param data_dir: Path of the lists, given as either a string or a list of strings, is processed using
            `os.path.join`.
        :param word_getter: Indicates how the words to be searched through are achieved. Currently only "NER" and
            "BERT" are supported.
        :param cutoff_score: The cut-off score below which matches will not be returned.
        :param entity_types: Which entity types to search for, e.g. `["person", "location"]`.
        """
        self.fuzzy_permutative_finders = {
            entity_type: FuzzyPermutativeFinder(
                op.join(*data_dir), entity_type, word_getter, cutoff_score)
            for entity_type in entity_types
        }
        self.fuzzy_finders = {
            entity_type: FuzzyFinder(
                op.join(*data_dir), entity_type, word_getter, cutoff_score)
            for entity_type in entity_types
        }
        self.direct_finders = {
            entity_type: DirectFinder(
                op.join(*data_dir), entity_type, word_getter)
            for entity_type in entity_types
        }

    def __call__(self, sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words
        :return: The same sentences, adorned with information found in various types of lists.
        """
        self.prefill_empty_labels(sentences)

        # Obtain list_results
        for fuzzy_permutative_finder in self.fuzzy_permutative_finders.values():
            fuzzy_permutative_finder(sentences)
        for fuzzy_list_finder in self.fuzzy_finders.values():
            fuzzy_list_finder(sentences)
        for direct_list_finder in self.direct_finders.values():
            direct_list_finder(sentences)
        return sentences

    @staticmethod
    def prefill_empty_labels(sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words
        :return: None, the word dicts are modified by adding the "labels" key, if the word has "ner" set to True.
        """
        for sentence in sentences:
            for word in sentence:
                if word['ner']:
                    labels_dict = word.get('labels', dict())
                    labels_list = labels_dict.get('lists', [])
                    labels_dict['lists'] = labels_list
                    word['labels'] = labels_dict
