from os import path as op
import logging
from typing import List, Dict

from lib import constants
from lib.ner_lists.entity import Entity
from lib.ner_lists.finder import Finder, FindEntities
from lib.ner_lists.fuzzy_matcher import FuzzyListMatcher
from lib.ner_lists.fuzzy_permutative_finder import remove_overlap

LOGGER = logging.getLogger(__name__)


class FuzzyFinder(Finder):
    """
    Finds entities from lists where the searchable forms are *not* to be permuted but *are* to be matched fuzzily.
    An object is instantiated for each different type of entity in the project, e.g. person, location and date.
    """
    def __init__(self, data_dir: str, entity_type: str, word_getter: str, cutoff_score: float):
        """
        :param data_dir: Path to the data files to use relative to the pipeline_data director, e.g. "ner_lists/SZSA".
        :param entity_type: See :class:`Finder`
        :param word_getter: See :class:`Finder`
        :param cutoff_score: The lowest score for entities to still be considered a hit.
        """
        filepath = op.join(constants.DATA_DIR, data_dir, 'fuzzy', entity_type)

        searchable_lists = self.create_lists(filepath)

        if not searchable_lists:
            LOGGER.info(f"Cannot create fuzzy_lists for {filepath}, probably missing txt-files")
        entity_finders = {
            name: FindEntitiesGivenList(searchable_list, cutoff_score=cutoff_score)
            for name, searchable_list in searchable_lists.items()
        }
        super().__init__(entity_type, entity_finders, word_getter)


class FindEntitiesGivenList(FindEntities):
    """
    Finds entities from a *single* list where the searchable forms are *not* to be permuted but *are* to be matched
        fuzzily.
    """
    def __init__(self, fuzzy_list: Dict, cutoff_score: float):
        """
        :param fuzzy_list: The list to search, given by a dictionary where the keys are the searchables and the values
            represent the (canonical) information on each entity.
        :param cutoff_score: The lowest score for entities to still be considered a hit.
        """
        self.tm = FuzzyListMatcher(fuzzy_list, cutoff_score)
        super().__init__()

    def __call__(self, i_sentence: int, sentence: List[Dict]) -> List[Entity]:
        entities = list(self.tm.find_matches(i_sentence, sentence))
        # delete overlapping entities
        final_entities = remove_overlap(entities, len(sentence))
        return final_entities
