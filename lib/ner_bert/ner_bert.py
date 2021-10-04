import logging
import os.path as op
from typing import Dict, List, Tuple, Optional
import numpy as np

from lib.ner_bert.inference import Bert
from lib.constants import DATA_DIR
from data_suppliers.text.nerbio.data_classes import InputData

LOGGER = logging.getLogger(__name__)
BERT_CATEGORIES = ('person', 'location', 'time')

BERTS = {
    '4-1-0-split': {
        'vocab': ('<PAD>', 'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME'),
        'weights_path': op.join(
            DATA_DIR, 'ner_bert', '4-1-0-split', 'weights.01-0.8802.h5'),
        'map': {'person': 'PER', 'location': 'LOC', 'time': 'TIME'},
        'word_form': 'post_correction'
    },
    '40-10-1-split': {
        'vocab': ('<PAD>', 'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME'),
        'weights_path': op.join(
            DATA_DIR, 'ner_bert', '40-10-1-split',
            'generator-40-10-1-split-t16-sigmoid-retry-082423_weights.02-0.9011.h5'),
        'map': {'person': 'PER', 'location': 'LOC', 'time': 'TIME'},
        'word_form': 'post_correction'
    },
    '40-10-1-split-minus-137': {
        'vocab': ('<PAD>', 'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME'),
        'weights_path': op.join(
            DATA_DIR, 'ner_bert', '40-10-1-split-minus-137',
            'sigmoid-40-10-1-split-minus-137-093503_weights.04-0.9119.h5'),
        'map': {'person': 'PER', 'location': 'LOC', 'time': 'TIME'},
        'word_form': 'post_correction'
    },
    '40-10-1-split-minus-137-fixed': {
        'vocab': ('<PAD>', 'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME'),
        'weights_path': op.join(
            DATA_DIR, 'ner_bert', '40-10-1-split-minus-137-fixed',
            'common-logging-fixed-gsutils-cp-163512_weights.08-0.9248.h5'),
        'map': {'person': 'PER', 'location': 'LOC', 'time': 'TIME'},
        'word_form': 'post_correction'
    },
    'mock': {
        'vocab': ('<PAD>', 'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-TIME', 'I-TIME'),
        'weights_path': '',
        'map': {'person': 'PER', 'location': 'LOC', 'time': 'TIME'},
        'word_form': 'word'
    }
}


class MultipleBerts:
    """
    Responsible for inference on multiple ner BERTS.
    """

    def __init__(self, berts_to_use: Dict[str, List[str]]):
        """
        :param berts_to_use: The keys of the dict indicate which BERTs to use and the values are tuples of the entities
            for which the keys should be used.
        """
        self.berts_to_use = berts_to_use
        # Initialize NERs
        self.berts = dict()
        for key in self.berts_to_use.keys():
            vocab = BERTS[key]["vocab"]
            weights_path = BERTS[key]["weights_path"]
            # logger.info('Initializing BERT-' + key + ' with entities: ' + vocab)
            self.berts[key] = Bert.create_bert(vocab, weights_path)

    def __call__(self, sentences):
        """

        :param sentences:
        :return: The same sentences, with named entity recognition information added. In particular the keys `"labels"`
            and `"bio"` are added. The former includes detailed information on the BERT NER task within the `"BERT"`
            key below it. The key `"bio"` contains a top-level conclusion of what type of entity we are dealing with.
        """
        # Do inference over multiple bert ners
        ner_sentences = [[word for word in sentence if word['ner']] for sentence in sentences]
        bert_ner_results = self._run_berts(ner_sentences)

        # Merge the results
        # we use the mutability of each word dict
        # while adding the result to each word, we don't have to know where it fits in the list of sentences
        all_ner_words = [word
                         for sentence in ner_sentences
                         for word in sentence]
        all_results = [{model_name: word_result}
                       for model_name, model_result in bert_ner_results.items()
                       for sentence_result in model_result
                       for word_result in sentence_result]
        assert len(all_ner_words) == len(all_results), f'Length of results and words does not match ' \
                                                       f'{len(all_ner_words)} != {len(all_results)}'
        for ner_word, results in zip(all_ner_words, all_results):
            ner_word['labels'] = ner_word.get('labels', {})  # add the labels key if it does not yet exist
            ner_word['labels']['BERT'] = self._map_bert_results(results)
            ner_word['bio'] = self._draw_conclusion(ner_word)
        self._add_entity_be_chars(all_ner_words)
        return sentences

    @staticmethod
    def _add_entity_be_chars(words: List[Dict]):
        """
        :param words: NER words that have bio-encoded entities
        :return: None, the words are mutated
        """
        entity_words = []
        for word in words:
            if word['bio'][0] in ('O', 'B'):
                # a B or O indicates that whatever entity was in "entity_words" has to be processed,
                # i.e. an entity is "complete".
                MultipleBerts._add_entity_be_chars_to_words(entity_words)
                if word['bio'][0] == 'B':
                    # start a new entity_words on a 'B'
                    entity_words = [word]
                else:
                    # clear the entity words on an 'O' (not strictly necessary, but neater)
                    entity_words = []
            elif word['bio'][0] in 'I':
                entity_words.append(word)
            else:
                raise ValueError
        # after the last word, add the begin_chars to the entities too
        MultipleBerts._add_entity_be_chars_to_words(entity_words)

    @staticmethod
    def _add_entity_be_chars_to_words(entity_words: List[Dict]):
        """
        :param entity_words: The words belonging to a certain entity.
        :return: None, the words are mutabted.
        """
        chars = [[word['begin_char'], word['end_char']] for word in entity_words]
        for entity_word in entity_words:
            entity_word['entity_chars'] = chars

    def _draw_conclusion(self, word: Dict):
        """
        :param word: The word for which to draw a conclusion on its entity
        :return: A string representation of the BIO notation, given in unabbreviated form, e.g. `"B-person"` and not
            the abbreviated form used by BERT `"B-PER"`. If multiple BERTs are in use (not currently the case) then the
            last BERT in the `self.berts_to_use` dict to predict a "B" or "I" is given priority.
        """
        conclusion = 'O'
        for bert in self.berts_to_use:
            label = word['labels']['BERT'][f'max_{bert}']
            if label not in ('O', '<PAD>'):
                # invert the mapping of BERT-entity names to interface names
                conclusion = label[:2] + {v: k for k, v in BERTS[bert]['map'].items()}[label[2:]]
            del word['labels']['BERT'][f'max_{bert}']
        # N.B. this strategy means the *LAST* BERT to say "not O/<PAD>" is given priority.
        return conclusion

    def _map_bert_results(self, all_label_probabilities: Dict[str, Dict[str, np.float64]]):
        """
        :param all_label_probabilities: The results of all the BERTs for a particular word, given as a Dictionary with a
            key for each BERT, with each key pointing to a Dictionary with *BERT vocab* keys and probability values.
        :return: A dictionary with the likelihood for each *true vocab*.
        """
        bert_labels = {}
        # for each BERT that has been run, apply the results from its output to the format we want
        for bert, entities in self.berts_to_use.items():
            test_word, label_probabilities = all_label_probabilities[bert]
            relevant_bi_keys = [bi + '-' + BERTS[bert]['map'][entity]
                                for entity in entities for bi in ('B', 'I')]
            relevant_o_keys = ['O', '<PAD>']
            relevant_keys = relevant_bi_keys + relevant_o_keys
            # note that below we convert to float (from np.float) for json serializability
            relevant_label_probabilities = {tag: float(label_probabilities[tag])
                                            for tag in relevant_keys}
            max_dict = max(relevant_label_probabilities.items(), key=lambda x: x[1])[0]
            for entity in entities:
                entity_bi_keys = [bi + '-' + BERTS[bert]['map'][entity] for bi in ('B', 'I')]
                bert_labels[entity] = self._map_bert_entity_result(
                    relevant_label_probabilities, relevant_o_keys, entity_bi_keys, max_dict)
            bert_labels[f'max_{bert}'] = max_dict
        return bert_labels

    @staticmethod
    def _map_bert_entity_result(label_probs: Dict[str, float], o_keys: List[str], bi_keys: List[str], max_dict: str):
        """
        :param label_probs: The label probailities relevant for this BERT
        :param o_keys: The `"O"` and `"<PAD>"` keys.
        :param bi_keys: The `"B"` and `"I"` keys relevant for this BERT
        :param max_dict: The most likely category according to this BERT
        :return: A dictionary with the label probabilities and bio-conclusion for this entity type.
        """
        entity_labels = {
            'bio': 'O',  # by default, everythin is assumed to be O
            'label_probabilities': {o_key: label_probs[o_key] for o_key in o_keys}
            # this only INITIALIZES the label_probabilities, they will be expanded below
        }
        for bi_key in bi_keys:
            bi = bi_key[0]
            entity_labels['label_probabilities'][bi] = label_probs[bi_key]
            if max_dict == bi_key:
                entity_labels['bio'] = bi  # take the B or the I from max_dict
        return entity_labels

    def _run_berts(self, ner_sentences: List[List[Dict]]):
        """
        :param ner_sentences: List of lists of words that have the `"ner"` flag True.
        :return: The probabilities of the different entities according to the specific model
        """
        results = dict()
        for model_name, bert in self.berts.items():
            # First obtain only the words that are suitable for ner (ignore newlines etc.)
            word_form = BERTS[model_name]['word_form']
            vocab = BERTS[model_name]['vocab']

            sentences_for_ner = [[word[word_form] for word in sentence] for sentence in ner_sentences]
            input_data = InputData.from_sentences(sentences_for_ner, bert.tokenizer, bert.max_length, vocab)

            # Inference
            result = bert(input_data.get_x())
            results[model_name] = input_data.inverse_apply_labels(result, trim_ends=True)

        return results
