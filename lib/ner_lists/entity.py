from typing import Tuple, Dict


class Entity:
    """
    Defines an Entity found in the text, contains all necessary properties as fields.
    """
    def __init__(self, sentence: int, begin: int, end: int, score: float,
                 searchable: str, canonical_form: str, extra_attributes: Dict[str, str]):
        """
        :param sentence: The number of the sentence in which the entity was found
        :param begin: Begin character of the entity
        :param end: End character of the entity
        :param score: Score of the match
        :param searchable: The "searchable" form of the word(s) that was/were matched
        :param canonical_form: The canonical form of the entity
        :param extra_attributes: Any extra attributes
        """
        self.sentence = sentence
        self.begin = begin
        self.end = end
        self.score = score
        self.searchable = searchable
        self.canonical_form = canonical_form
        self.extra_attributes = extra_attributes

    def to_dict(self, **kwargs):
        """
        :param kwargs: Additional key-value pairs to place on the dict.
        :return: A dictionary containing relevant information on the entity for the output json as data
        """
        d = dict(
            score=self.score,
            searchable=self.searchable,
            canonical_form=self.canonical_form,
            extra_attributes=self.extra_attributes,
        )
        return {**d, **kwargs}
