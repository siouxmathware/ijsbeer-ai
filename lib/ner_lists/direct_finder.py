from os import path as op
from typing import Dict, List

from lib import constants
from lib.ner_lists.entity import Entity
from lib.ner_lists.finder import Finder, FindEntities


class DirectFinder(Finder):
    """
    Finds entities from lists where the searchable forms are *not* to be permuted *nor* to be matched fuzzily.
    An object is instantiated for each different type of entity in the project, e.g. person, location and date.
    """
    def __init__(self, data_dir: str, entity_type: str, word_getter: str):
        """
        :param data_dir: Path to the data files to use relative to the pipeline_data director, e.g. "ner_lists/SZSA".
        :param entity_type: See :class:`Finder`
        :param word_getter: See :class:`Finder`
        """
        filepath = op.join(constants.DATA_DIR, data_dir, 'direct', entity_type)
        searchable_lists = self.create_lists(filepath)
        entity_finders = {
            name: FindEntitiesGivenDirectList(searchable_list)
            for name, searchable_list in searchable_lists.items()
        }

        super().__init__(entity_type, entity_finders, word_getter)


class FindEntitiesGivenDirectList(FindEntities):
    """
    Finds entities from a *single* list where the searchable forms are *not* to be permuted *nor* to be matched fuzzily.
    """
    def __init__(self, direct_list: Dict):
        """
        :param direct_list: The list to search, given by a dictionary where the keys are the searchables and the values
            represent the (canonical) information on each entity.
        """
        self.grouped_list = self._group_list_by_length(direct_list)
        super().__init__()

    def __call__(self, i_sentence: int, sentence: List[Dict]) -> List[Entity]:
        results = []
        self.sentence_length = len(sentence)
        for i in self.grouped_list.keys():
            for j in range(self.sentence_length-i+1):
                candidate = tuple(w.lower() for w in sentence[j:j+i])
                if candidate in self.grouped_list[i]:
                    for occurence in self.grouped_list[i][candidate]:
                        results.append(Entity(
                            sentence=i_sentence,
                            begin=j,
                            end=j+i,
                            score=1.,
                            searchable=' '.join(occurence["searchable"]),
                            # TODO: return the tuple and then match it later
                            canonical_form=occurence["canonical_form"],
                            # TODO: The occurence needs to have the extra_attributes separate
                            extra_attributes=occurence["extra_attributes"]
                        ))
        return results

    @staticmethod
    def _group_list_by_length(_list):
        grouped_list = dict()
        for searchable, canonical in _list.items():
            nr_of_elements = len(searchable)
            if nr_of_elements not in grouped_list.keys():
                grouped_list[nr_of_elements] = {}
            grouped_list[nr_of_elements][searchable] = canonical
        return grouped_list
