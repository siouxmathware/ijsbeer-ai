import json
import jsonschema
import os.path as op

from typing import List, Dict, Tuple, Optional


class Validator:
    """
    Valicator class that allows validating (partial) results of the pipeline.
    The schema used is given by :download:`format.schema.json </../../lib/format.schema.json>`.
    Currently only used during testing due to possible performance impact during production.

    TODO: Make the validator available via the API.
    """
    IGNORABLE = ('post_correction', 'ner_bert', 'ner_lists', 'modernisation')

    def __init__(self, ignore: Tuple = ()):
        """
        :param ignore: Which steps to ignore in the validation, should be one of
            :attr:`IGNORABLE <lib.schema.Validator.IGNORABLE>`.
            Useful when calling only a subset of all possible steps of the pipeline.
        """
        assert all(ig in self.IGNORABLE for ig in ignore)
        with open(op.join(op.dirname(__file__), 'format.schema.json')) as f:
            self.schema = json.load(f)
        if 'post_correction' in ignore:
            self._ignore_post_correction()
        if 'ner_bert' in ignore:
            self._ignore_ner_bert()
        if 'ner_lists' in ignore:
            self._ignore_ner_lists()
        if 'modernisation' in ignore:
            self._ignore_modernisation()
        # if 'ner_bert' in ignore and 'ner_lists' in ignore:
        #     self._ignore_ner()

    @classmethod
    def from_include(cls, include: Optional[Tuple] = None):
        """
        :param include: Which steps to include. If `None` (default) all steps will be included.
        :return: Instance of the Validator that considers only the specified steps.
        """
        if include is None:
            # ignore no steps in the default case.
            return cls()
        assert all(inc in cls.IGNORABLE for inc in include)
        ignore = tuple(ig for ig in cls.IGNORABLE if ig not in include)
        return cls(ignore)

    def __call__(self, json_data: List[List[Dict]]):
        """
        :param json_data: Pipeline output data that is to be verified against the schema.
        :return: None, exception is raised if validation fails.
        """
        jsonschema.validate(json_data, self.schema)

    def _ignore_post_correction(self):
        self.schema['definitions']['word']['required'].remove('post_correction')

    def _ignore_ner_bert(self):
        self.schema['definitions']['labels']['required'].remove('BERT')

    def _ignore_ner_lists(self):
        self.schema['definitions']['labels']['required'].remove('lists')

    def _ignore_modernisation(self):
        self.schema['definitions']['word']['required'].remove('modernisation')
        self.schema['definitions']['word']['required'].remove('remove_whitespace_for_modernisation')

    def _ignore_ner(self):
        self.schema['definitions']['word']['required'].remove('labels')
