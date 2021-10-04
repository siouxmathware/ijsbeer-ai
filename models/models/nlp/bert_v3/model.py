from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow_hub as hub

CONFIGS = {
    "bert_multi_cased_L-12_H-768_A-12/3": {
        "tf_hub_location": "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
        "max_seq_length": 512
    }
}


def get_bert_ner_model(nr_categories, bert_config):
    """
    Returns a keras model based on a pre-trained BERT with some extra layers on top for named entity
    recognition (ner).
    """
    # Obtain the bert_v2 layer and properties
    bert_layer = hub.KerasLayer(CONFIGS[bert_config]["tf_hub_location"], trainable=False, name=bert_config)
    max_seq_length = CONFIGS[bert_config]["max_seq_length"]

    # Define the input, 4 of which 3 for BERT and 1 to indicate the correct labels
    layer_inputs = {
        'input_word_ids': Input(shape=(max_seq_length,), name="input_word_ids", dtype="int32"),
        'input_mask': Input(shape=(max_seq_length,), name="input_mask", dtype="int32"),
        'input_type_ids': Input(shape=(max_seq_length,), dtype="int32", name='input_type_ids')
    }
    model_inputs = {
        **layer_inputs,
        **{'valid_ids': Input(shape=(max_seq_length,), name="valid_ids")}
    }

    # Construct the model
    nr_entities = 1 + 1 + 2 * nr_categories  # <PAD>, O, B-FIRST, I-FIRST, B-SECOND, I-SECOND, etc.
    sequence_output = bert_layer(layer_inputs)
    output = Dense(nr_entities, activation='sigmoid', name='dense')(sequence_output)

    model = Model(model_inputs, [output])

    return model
