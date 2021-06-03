import os.path as op
import unittest
import json

import train_script

TEST_DATA_DIR = op.join(op.dirname(__file__), 'data')


class TestTraining(unittest.TestCase):
    def testTraining(self):
        with open(op.join(TEST_DATA_DIR, 'test_args.json')) as f:
            arguments = json.load(f)
        train_script.main(arguments)


if __name__ == "__main__":
    TestTraining().testTraining()
