import logging
from typing import Dict, Tuple, List, Union


class Pipeline:
    """
    Responsible for the whole AI pipeline:

    #. :mod:`String to sentences <lib.string_to_sentences.string_to_sentences>`

    #. :mod:`Post-correction <lib.post_correction.post_correction>`

    #. :mod:`NER using BERT <lib.ner_bert.ner_bert>`

    #. :mod:`NER using lists <lib.ner_lists.ner_lists>`

    #. :mod:`Modernisation <lib.modernisation.modernisation>`
       (performed last so that named entities can be exempted from modernisation

    """
    def __init__(self, config: Dict[str, Dict]):
        """
        """
        # Initializing
        if 'string_to_sentences' in config:
            from lib.string_to_sentences.string_to_sentences import StringToSentences
            self.string_to_sentences = StringToSentences(**config['string_to_sentences'])
        if 'post_correction' in config:
            from lib.post_correction.post_correction import PostCorrection
            self.post_correction = PostCorrection(**config['post_correction'])
        if 'ner_bert' in config:
            from lib.ner_bert.ner_bert import MultipleBerts
            self.ner_bert = MultipleBerts(**config['ner_bert'])
        if 'ner_lists' in config:
            from lib.ner_lists.ner_lists import NerLists
            self.ner_lists = NerLists(**config['ner_lists'])
        if 'modernisation' in config:
            from lib.modernisation.modernisation import Modernisation
            self.modernisation = Modernisation(**config['modernisation'])

        self.possible_steps = tuple(
            step
            for step in ('string_to_sentences', 'post_correction', 'ner_bert', 'ner_lists', 'modernisation')
            if step in config
        )
        logging.info(f"Initialized pipeline with steps {self.possible_steps}")

    def __call__(self, input_data: Union[str, List[List[Dict]]], steps: Tuple[str, ...] = None):
        """
        :param input_data: Either a string of historical Dutch text, if the step "string_to_sentences" is involved,
            or a dictionary corresponding to the :download:`json schema </../../lib/format.schema.json>` if it is not.
            The format of the input data in the latter case should correspond to some extent to the remaining steps that
            are to be executed. The table below provides an overview of which steps are required and/or prefered to
            have  been executed for each subsequent step. A required step must either be executed in the same call to
            the pipeline, or the results of that step must be present in the input data.

            .. list-table:: Dependency on previous steps
               :widths: 25 25 25 25
               :header-rows: 1

               * - Step - requires
                 - STS
                 - PC
                 - BERT
               * - Post-correction
                 - yes
                 - -
                 - -
               * - NER using BERT
                 - yes
                 - yes [1]_
                 - -
               * - NER using Lists
                 - yes
                 - yes [1]_
                 - depends [2]_
               * - Modernisation
                 - yes
                 - yes [1]_
                 - preferably [3]_

            .. [1] This behaviour is subject to change. If processing ground truth texts, the post-correction step
               should be skipped. Skipping this step altogether, however, results in the absence of the
               "post_correction" key on which subsequent steps rely.
            .. [2] Depending on the configuration for the project at hand, the lists are only searched to match entities
               found by BERT. For such a project, e.g. SZSA, the BERT step is a prerequisite for the lists step.
            .. [3] The modernisation is configured to leave certain found entities unchanged. This is only possible if
        :param steps: Which steps to execute within this call to the pipelines
        :return: Dict with a format according to `lib.schema (format.schema.json)` Depending on which parts of the
            pipeline are called, the required input and expected output will be different.
        """
        steps = steps if steps is not None else self.possible_steps

        assert all(s in self.possible_steps for s in steps), f"Invalid steps {steps}, choose from {self.possible_steps}"
        if 'string_to_sentences' in steps:
            logging.info('Doing string_to_sentences')
            assert isinstance(input_data, str)
            obj = self.string_to_sentences(input_data)
        else:
            obj = input_data
            # if post_correction step is not required, the input_data must be a list of list of dicts
            assert isinstance(obj, list)
            assert all(isinstance(s, list) for s in obj)
            assert all(isinstance(w, dict) for s in obj for w in s)
        if 'post_correction' in steps:
            logging.info('Doing post_correction')
            obj = self.post_correction(obj)
        if 'ner_bert' in steps:
            logging.info('Doing ner_bert')
            obj = self.ner_bert(obj)
        if 'ner_lists' in steps:
            logging.info('Doing ner_lists')
            obj = self.ner_lists(obj)
        if 'modernisation' in steps:
            logging.info('Doing modernisation')
            obj = self.modernisation(obj)

        return obj


if __name__ == "__main__":
    import json
    with open("server/config_NA.json") as f:
        config = json.load(f)

    p = Pipeline(config)
    result = p("Rijste van de aangecomene, en ver„\n1roi")
    pass