import json
import argparse
import logging
import traceback
import sys
import time
from flask import Flask, request, jsonify

from server.exceptions import BadRequest, InternalServerError
from server.parallel_support import Work, create_queues

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = Flask(__name__)


@app.route("/")
def main():
    """
    Fallback to point users in the right direction if they do not specify the correct application to call
    """
    text = 'Please go to /pipeline and run the model with the following json input structure:{"input_data": "..."}'
    return text


@app.route("/pipeline", methods=['POST'])
def call_pipeline():
    """
    This method calls the pipeline. The input is provided by a 'POST' method in the server call. The post-data should be
    a dict of the following structure:
    {"input_data": "..."[, "steps": "..."]},
    where the "input_data" is either a plain string (if post-correction is the first step)
    or a json-formated string for the other steps, and the "steps" denote which steps to execute in order, from
    the possibilities ["post_correction", "ner_bert", "ner_lists", "modernisation"]. If the optional "steps" key is not
    provided, ALL steps are executed in this order. If an empty list is provided, an error is raised.''
    :return: {"results": result containing the augmented data, "message": "Success"}. See the schema for details on the
    form of the result.
    """
    logger.info("Calling pipeline.")
    if request.method == 'POST':
        data = request.json
        # Validate input
        if data is None:
            raise BadRequest("Input data seems to be missing, please provide valid json as input.")
        if not isinstance(data, dict) or "input_data" not in data.keys():
            raise BadRequest('''
Input data format seems invalid, please use the structure 
{"input_data": "..."[, "steps": "..."]},
where the "input_data" is either a plain string (if post-correction is the first step)
or a json-formated string for the other steps, and the "steps" denote which steps to execute in order, from
the possibilities ["post_correction", "ner_bert", "ner_lists", "modernisation"]. If "steps" is not provided, 
ALL steps are executed in this order. If an empty list is provided, an error is raised.''')

        # Execute NER
        try:
            uuid = time.time()
            work = Work(uuid, **data)
            pre_queue.put_nowait(work)
            while True:
                work = done_queue.get()
                if work.uuid == uuid:
                    break
                done_queue.put(work)

        except Exception as _:
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            raise InternalServerError(error_trace)

        logger.info('Function call_pipeline() returned successfully')
        return {
            "results": work.data,
            "message": "Success"
        }


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    """
    Handle bad requests.
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(InternalServerError)
def handle_internal_server_error(error):
    """
    Handle internal errors.
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == "__main__":
    # Global variables ner and server
    # Note that INITIALIZING without BERT is not the same as calling the server but not requesting the BERT output
    logger.info(f'sys.agv: {sys.argv!r}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str, default='server/config_NA.json')
    parser.add_argument('--n_parallel', type=int, default=4)
    args, unknown = parser.parse_known_args()
    argument_dict = vars(args)
    with open(argument_dict["configfile"]) as f:
        config = json.load(f)

    pre_queue, bert_queue, post_queue, done_queue = create_queues(config, argument_dict["n_parallel"])

    app.run(host="0.0.0.0", port="5005")
