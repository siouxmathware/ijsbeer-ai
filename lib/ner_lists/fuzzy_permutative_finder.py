from os import path as op
import logging
from typing import List, Dict

from lib import constants
from lib.ner_lists.entity import Entity
from lib.ner_lists.finder import Finder, FindEntities
from lib.ner_lists.fuzzy_matcher import FuzzyTreeMatcher

LOGGER = logging.getLogger(__name__)


class FuzzyPermutativeFinder(Finder):
    """
    Finds entities from lists where the searchable forms *are* to be permuted *and* to be matched fuzzily.
    An object is instantiated for each different type of entity in the project, e.g. person, location and date.
    """
    def __init__(self, data_dir: str, entity_type: str, word_getter: str, cutoff_score: float):
        """
        :param data_dir: Path to the data files to use relative to the pipeline_data director, e.g. "ner_lists/SZSA".
        :param entity_type: See :class:`Finder`
        :param word_getter: See :class:`Finder`
        :param cutoff_score: The lowest score for entities to still be considered a hit.
        """
        filepath = op.join(constants.DATA_DIR, data_dir, 'fuzzy_permutative', entity_type)

        all_lists = self.create_lists(filepath)

        searchable_trees = {name: create_tree(all_list) for name, all_list in all_lists.items()}
        if not searchable_trees:
            LOGGER.info(f"Cannot create fuzzy_permutative for {filepath}, probably missing txt-files")
        entity_finders = {
            name: FindEntitiesGivenTree(searchable_tree, cutoff_score=cutoff_score)
            for name, searchable_tree in searchable_trees.items()
        }
        super().__init__(entity_type, entity_finders, word_getter)


def create_tree(all_list: Dict) -> Dict[str, Dict]:
    """
    :param all_list: A dictionary of searchable-canonical pairs
    :return: A dictionary where the keys are the first word that would be searched, the values are nested dicts of the
        same form as the input, i.e. searchable-canonical pairs.
    """
    freq_table = create_freq_table(all_list)
    tree = {}
    for searchable, canonical in all_list.items():
        ordered = sorted(searchable, key=lambda x: freq_table[x], reverse=False)
        key_word = ordered[0]
        key_list = tree.get(key_word, {})
        key_list[searchable] = canonical
        tree[key_word] = key_list
    return tree


def create_freq_table(all_list: Dict):
    """
    :param all_list: A dictionary of searchable-canonical pairs.
    :return: A dictionary stating how often each searchable element occurs in the list.
    """
    table = {}
    for element in all_list:
        for word in element:
            table[word] = table.get(word, 0) + 1
    return table


class FindEntitiesGivenTree(FindEntities):
    def __init__(self, tree, cutoff_score):
        """
        :param tree: The list to search, given by a dictionary where the keys are the searchables and the values
            represent the (canonical) information on each entity.
        :param cutoff_score: The lowest score for entities to still be considered a hit.
        """
        self.tm = FuzzyTreeMatcher(tree, cutoff_score)
        super().__init__()

    def __call__(self, i_sentence, sentence) -> List[Entity]:
        entities = list(self.tm.find_matches(i_sentence, sentence))
        # delete overlapping entities
        final_entities = remove_overlap(entities, len(sentence))
        return final_entities


def remove_overlap(entities: List[Entity], sentence_length: int) -> List[Entity]:
    """
    :param entities: The found entities by the permutative searching.
    :param sentence_length: The length of the sentence in which the entities occur.
    :return: A trimmed list of found entities where overlapping results are discarded.
    """
    final_matches = []
    base = [0] * sentence_length
    for i, x in enumerate(entities):
        match_length = x.end - x.begin
        if base[x.begin:x.end] == [0] * match_length:
            base[x.begin:x.end] = [i + 1] * match_length
        else:
            # Gather all overlapping matches:
            overlappings = list(set(base[x.begin:x.end]) - {0})
            distances = [entities[y - 1].score for y in overlappings]
            current_distance = x.score

            # TODO: below is a hack to cope with the siatuation when "distances" is emptu
            if not distances or current_distance > max(distances):
                # Delete all overlapping
                for k in range(len(base)):
                    if base[k] in overlappings:
                        base[k] = 0
                # Add new match
                base[x.begin:x.end] = [i + 1] * match_length
    for i in list(set(base) - {0}):
        final_matches.append(entities[i - 1])
    return final_matches
