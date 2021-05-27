"""
Given a set of words and two sets of labels, the f1 score is calculated.
"""

import logging

# Define logger
logger = logging.getLogger(__name__)


class F1(object):
    """
    Represents the different representations of the f1 score.
    """
    def __init__(self):
        logger.info("F1 test initialized")

    def __call__(self, labels_a, labels_b, categories=( ['B-PER', 'I-PER'], ['B-LOC', 'I-LOC'], ['B-ORG', 'I-ORG'],[ 'B-MISC', 'I-MISC'])):
        """
        Labels a are correct:
        """
        f1_per_cat = []
        label_keys = ['person', 'location', 'organisation', 'miscellaneous']

        for cat_nr, cat in enumerate(categories):
            false_negatives = 0
            false_positives = 0
            true_positives = 0
            for hit in cat:
                for a, c in zip(labels_a, labels_b):
                    b = c[label_keys[cat_nr]]
                    if a == b and a == hit:
                        true_positives += 1
                    elif a == hit and a != b:
                        false_negatives += 1
                    elif b == hit and a != b:
                        false_positives += 1

            precision = 0
            recall = 0
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_per_cat.append([f1, true_positives, false_positives, false_negatives])

        return f1_per_cat
