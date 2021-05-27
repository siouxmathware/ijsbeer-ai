"""
Module that contains common model initiation functions.
"""
import os
import logging
import json

from tensorflow.keras.models import load_model

from common.common import load_and_merge_parameters
from models.nlp.bert_v2.model import get_bert_ner_model as get_v2
from models.nlp.bert_v3.model import get_bert_ner_model as get_v3
from models.nlp.mocks.mock_bert import get_bert_ner_model as get_mock

LOGGER = logging.getLogger(__name__)
AVAILABLE_MODELS = {
    "bert_ner_v2": {
        "model": get_v2,
        "parameter_path": "models/nlp/bert_v2/parameters.json"
    },
    "bert_ner_v3": {
        "model": get_v3,
        "parameter_path": "models/nlp/bert_v3/parameters.json"
    },
    "bert_mock": {
        "model": get_mock,
        "parameter_path": "models/nlp/mocks/parameters.json"
    }
}


def initiate_correct_model(model_name=None, model_path=None, **model_parameters):
    """
    Either specify a model path to continue training and existing model or specify a model name to initiate a new model.
    Returns a initiated model of type tf.keras.Model
    """
    # Validate input
    if model_name is None and model_path is None:
        raise ValueError("Please provide either a model name to initiate a model or a model path to load an "
                         "existing model.")
    if model_path is not None and not os.path.isfile(model_path):
        raise ValueError(f"Please provide a valid path to an existing model, path {model_path} does not represent "
                         f"a file")
    if model_name is not None and model_name not in AVAILABLE_MODELS.keys():
        raise ValueError(f"When specifying a model name {model_name}, please make sure model is available, is one "
                         f"of {[name for name in AVAILABLE_MODELS.keys()]}.")

    # Decide whether to load an existing model or initiate a new model
    if model_path is not None:
        LOGGER.info(f"Loading existing model at {model_path}.")
        return load_model(model_path, compile=False)
    else:
        # Merge parameters
        model_parameters = load_and_merge_parameters(model_parameters, AVAILABLE_MODELS[model_name]["parameter_path"])
        LOGGER.info(f"Model parameters:\n{json.dumps(model_parameters, indent=4)}")

        return AVAILABLE_MODELS[model_name]["model"](**model_parameters)
