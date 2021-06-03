import logging
from multiprocessing import Process, Queue

from lib.pipeline import Pipeline
# from tests.mock_pipeline import Pipeline

class Work:
    def __init__(self, uuid, input_data, steps=None):
        self.uuid = uuid
        self.data = input_data
        if steps is None:
            self.steps = ["string_to_sentences", "post_correction", "ner_bert", "ner_lists", "modernisation"]
        else:
            self.steps = steps


def target_wrapper(q, done_q, conf, nr):
    logging.info(f'Starting worker {nr}')
    logging.info(f'q in target: {q}')
    pipeline = Pipeline(conf)
    # Read from the queue; this will be spawned as a separate Process
    while True:
        work = q.get()
        relevant_steps = tuple(step for step in work.steps if step in conf)
        # logging.info(f"Work starting on {nr}")
        work.data = pipeline(work.data, relevant_steps)
        # logging.info(f"Work finishing on {nr}")
        done_q.put(work)


def create_queues(conf, n_parallel):
    pre_q = Queue()
    bert_q = Queue()
    post_q = Queue()
    done_q = Queue()

    all_steps = {
        "pre": ("string_to_sentences", "post_correction"),
        "bert": ("ner_bert",),
        "post": ("ner_lists", "modernisation"),
    }

    confs = {
        queue: {k: v for k, v in conf.items() if k in steps}
        for queue, steps in all_steps.items()
    }

    Process(target=target_wrapper, args=(bert_q, post_q, confs['bert'], 0), daemon=True).start()
    for nr in range(n_parallel):
        # reader_proc() reads from pqueue as a separate process
        Process(target=target_wrapper, args=(pre_q, bert_q, confs['pre'], nr), daemon=True).start()
        Process(target=target_wrapper, args=(post_q, done_q, confs['post'], nr), daemon=True).start()
    return pre_q, bert_q, post_q, done_q
