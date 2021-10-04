from tensorflow.keras.layers import Layer, Input, Dense, GlobalAveragePooling1D, InputLayer, Multiply, Permute
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_hub as hub

CONFIGS = {
    "bert_en_uncased_L-12_H-768_A-12/2": {
        "tf_hub_location": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
        "max_seq_length": 512
    },
    "bert_multi_cased_L-12_H-768_A-12/2": {
        "tf_hub_location": "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",
        "max_seq_length": 512
    },
    "bert_multi_uncased_L-12_H-768_A-12/2": {
        "tf_hub_location": "https://tfhub.dev/tensorflow/bert_multi_uncased_L-12_H-768_A-12/2",
        "max_seq_length": 512
    }
}


def get_bert_ner_model(nr_categories, config, bert_trainable, topping_sizes):
    """
    Returns a keras model based on a pre-trained BERT with some extra layers on top for named entity
    recognition (ner).
    :param nr_categories: number of categories in the NER
    :param config: name of BERT configuration as specified in CONFIGS above
    :param bert_trainable: whether to train the BERT part of the network too
    :param topping_sizes: sizes of intermediate layers in the topping, empty if a single layer
    """
    # Obtain the bert_v2 layer and properties
    bert_layer = hub.KerasLayer(CONFIGS[config]["tf_hub_location"], trainable=bert_trainable, name=config)
    max_seq_length = CONFIGS[config]["max_seq_length"]

    # Define the input, 4 of which 3 for BERT and 1 to indicate the correct labels
    input_word_ids = Input(shape=(max_seq_length,), name="input_word_ids", dtype="int32")
    input_mask = Input(shape=(max_seq_length,), name="input_mask", dtype="int32")
    input_type_ids = Input(shape=(max_seq_length,), dtype="int32", name='input_type_ids')
    valid_ids = Input(shape=(max_seq_length,), name="valid_ids")

    # Construct the model
    nr_entities = 1 + 1 + 2 * nr_categories  # <PAD>, O, B-FIRST, I-FIRST, B-SECOND, I-SECOND, etc.
    _, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
    output = _add_topping(nr_entities, sequence_output, topping_sizes)

    model = Model([input_word_ids, input_mask, input_type_ids, valid_ids], [output])

    return model


def _add_topping(nr_entities, prev_output, topping_sizes):
    if topping_sizes:
        for i, topping_size in enumerate(topping_sizes):
            prev_output = Dense(topping_size, activation='relu', name=f'topping{i}')(prev_output)
    output = Dense(nr_entities, activation='softmax', name='output')(prev_output)
    return output
