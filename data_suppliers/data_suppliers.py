"""
Module that contains common data supplier initialization functions.
"""
import logging
import json

from common.common import load_and_merge_parameters
from data_suppliers.text.nerbio.data_supplier import DataSupplier
from data_suppliers.text.nerbio.data_generator import DataGenerator
from data_suppliers.text.nerbio.data_generator_chunk import DataGeneratorChunk

LOGGER = logging.getLogger(__name__)
AVAILABLE_DATA_SUPPLIERS = {
    "ner_bio_text2fit": DataSupplier,
    "ner_bio_generator": DataGenerator,
    "ner_bio_generator_chunk": DataGeneratorChunk,
}


def initiate_correct_data_supplier(model, data_supplier_name=None, **data_parameters):
    """
    Specify a data supplier out of available data suppliers.
    """
    # Validate input
    if data_supplier_name not in AVAILABLE_DATA_SUPPLIERS.keys():
        raise ValueError(f"When specifying a data supplier name {data_supplier_name}, please make sure data_supplier_"
                         f"name is available, is one of {[name for name in AVAILABLE_DATA_SUPPLIERS.keys()]}.")

    # Merge parameters
    data_parameters = load_and_merge_parameters(data_parameters,
                                                AVAILABLE_DATA_SUPPLIERS[data_supplier_name].parameter_path)
    LOGGER.info(f"Data parameters:\n{json.dumps(data_parameters, indent=4)}")

    bert_layer = model.layers[[layer.name.startswith('bert') for layer in model.layers].index(True)]
    return AVAILABLE_DATA_SUPPLIERS[data_supplier_name](bert_layer, **data_parameters)
