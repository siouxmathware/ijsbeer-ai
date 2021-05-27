import csv
import itertools
import logging
import os
from os import path as op
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

from lib.ner_lists.entity import Entity
from lib.ner_lists.get_searchable_words import GetWords

LOGGER = logging.getLogger(__name__)


class FindEntities(metaclass=ABCMeta):
    """
    Abstract base class for the FindEntitiesGiven... classes
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """"""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        :param i_sentence: A number indicating which sentence we are searching
        :param sentence: List of words. Note that this is not a "true sentence" in the sense
            used throughout the project. It is a subset of all words, depending on the object's :const:`get_words`.
        :return: A list of found entities.
        """
        pass


class Finder:
    """
    Finds entities from lists. Derived classes exist for different types of matchings.
    """
    def __init__(self, entity_type: str, entity_finders: Dict[str, FindEntities], word_getter: str):
        """
        :param entity_type: The type of entity for which the finder is searching, e.g. "person".
        :param entity_finders: The entity finders that look through a specific list or tree.
        :param word_getter: Passed on the class
            :class:`GetWords <lib.ner_lists.get_searchable_words>`
            to determine how to get the list of searchable words.
        """
        self.get_words = GetWords.from_string(word_getter)(entity_type=entity_type)
        self.entity_finders = entity_finders

    def __call__(self, sentences: List[List[Dict]]):
        """
        :param sentences: List of lists of words.
        :return: None, the words are modified
        """
        searchable_word_dicts = self.get_words(sentences)
        for list_name, entity_finder in self.entity_finders.items():
            for i, sentence in enumerate(searchable_word_dicts):
                found_list_elements = entity_finder(i, [w['word'] for w in sentence])
                for entity in found_list_elements:
                    self._merge_results(list_name, entity, sentence)

    @staticmethod
    def _merge_results(list_name, ent: Entity, sentence: List[Dict]):
        """
        :param list_name: Name of the list
        :param ent: The entity found
        :param sentence: The sentence in which the list was found. Note that this is not a "true sentence" in the sense
            used throughout the project. It is a subset of all words, depending on the object's :const:`get_words`.
        :return:
        """
        ent_length = ent.end - ent.begin
        bio_labels = ['B'] + ['I'] * (ent_length - 1)
        for word, bio in zip(itertools.islice(sentence, ent.begin, ent.end), bio_labels):
            lists = word['labels'].get('lists', [])
            lists.append(ent.to_dict(list_name=list_name, bio=bio))
            word['labels']['lists'] = lists

    @staticmethod
    def create_lists(dirpath: str) -> Dict[str, Dict]:
        """
        :param dirpath: The directory where to search for files.
        :return: A dictionary where each key-value-pair is a combination of a filename with a corresponding dictionary
            of searchables.
        """
        if not op.exists(dirpath):
            return {}
        filenames_of_lists = [list_name
                              for list_name in sorted(os.listdir(dirpath))
                              if list_name.endswith(".txt") or list_name.endswith(".csv")]
        names_of_listfiles = [filename for filename in filenames_of_lists]
        dict_of_files = {}
        for name_of_listfile in names_of_listfiles:
            name_of_list, extension = op.splitext(name_of_listfile)
            assert extension in ('.txt', '.csv'), f"Wrong extension! {extension}"
            dict_of_files[name_of_list] = load_file(dirpath, name_of_list, extension)
        return dict_of_files


def load_file(dirpath: str, list_name: str, extension: str) -> Dict[Tuple[str, ...], List[Dict]]:
    """
    :param dirpath: The directory where the file is located
    :param list_name: The name of the file
    :param extension: The extension of the file
    :return: A dicrionary with the key "searchable" pointing to a list of dictionaries
        containing the keys "searchable", "canonical_form" and "extra_attributes". The first is given
        by a tuple of words (to allow reordering of the elements later), the second as a string and the third as a
        dictionary.

        Example:

        .. code-block:: python

            dict_searchables[('middelburg',)] = [
                {
                    'searchable': ('middelburg',),
                    'canonical_form': 'Maspeth',
                    'extra_attributes': {'geometry': 'Point(40.75 -73.9333)'}
                },
                {
                    'searchable': ('middelburg',),
                    'canonical_form': 'Middelburg',
                    'extra_attributes': {'geometry': 'Point(51.5021 3.6141)'}
                },
                {
                    'searchable': ('middelburg',),
                    'canonical_form': 'Middelburg',
                    'extra_attributes': {'geometry': 'Point(0 0)'}
                }
            ]
    """
    LOGGER.warning(f"Processing list {list_name} with extension {extension}")
    dict_searchables = {}
    with open(op.join(dirpath, list_name + extension), newline='', encoding='ISO-8859-1') as fp:
        if extension == '.csv':
            reader = csv.reader(fp, delimiter=',', dialect=csv.excel, quotechar='|')
            _ = next(reader)  # human readable headers can be tossed out
            headers = next(reader)
        elif extension == '.txt':
            reader = csv.reader(fp, delimiter='\t', quotechar="|")
            headers = ['canonical_form']
        for row in reader:
            assert len(row) == len(headers)
            row_dict = {}
            for index, row_field in enumerate(row):
                key = headers[index]
                if key != "not_used":
                    row_dict[key] = row_field
            searchable = row_dict.get("searchable", row_dict.get("canonical_form"))
            row_dict["searchable"] = tuple(searchable.lower().split(" "))
            canonical_dict = {
                "searchable": row_dict.pop("searchable"),
                "canonical_form": row_dict.pop("canonical_form"),
                "extra_attributes": row_dict
            }
            key = canonical_dict["searchable"]
            if key not in dict_searchables.keys():
                dict_searchables[key] = [canonical_dict]
            else:
                dict_searchables[key].append(canonical_dict)
    return dict_searchables
