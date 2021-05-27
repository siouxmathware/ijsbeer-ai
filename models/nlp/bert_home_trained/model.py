import tensorflow as tf
import tensorflow_hub as hub
from official.nlp.bert import tokenization
from models.nlp.bio_f1 import BioF1
import re
import logging


LOGGER = logging.getLogger(__name__)

CONFIGS = {
    "second": {
        "nr_tags": 8,
        "weights_path": "../nationaal_archief_ner/data/trained_models/second/weights.03-0.0284.h5"
    }
}


def get_bert_ner_model(config):
    """
    Returns a keras model based on a pre-trained BERT with some extra layers on top for named entity
    recognition (ner).
    :param config: name of BERT configuration as specified in CONFIGS above
    """

    model = BertTF(CONFIGS[config]['nr_tags'], CONFIGS[config]['weights_path'])

    return model


class BertTF:
    def __init__(self, nr_tags, weights_path, batch_size=4):
        self.model: tf.keras.Model = tf.keras.models.load_model(
            weights_path,
            custom_objects={  # must include these explicitly
                'KerasLayer': hub.KerasLayer,
                'BioF1': BioF1
            },
        )
        bert_layer = self.model.layers[[layer.name.startswith('bert') for layer in self.model.layers].index(True)]
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        if re.search(r'_cased_', bert_layer.name):
            do_lower_case = False
        elif re.search(r'_uncased_', bert_layer.name):
            do_lower_case = True
        else:
            LOGGER.error(f'Could not infer BERT casing from BERT layer name {bert_layer.name}')
            raise ValueError
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
        self.max_length = self.model.input_shape[0][-1]
        self.nr_tags = nr_tags
        self.batch_size = batch_size

    def __call__(self, x):
        return self.model.predict(x)
