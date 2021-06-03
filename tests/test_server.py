import unittest
import os
import os.path as op
import subprocess
import time
import requests
import os
from lib import schema


TEST_DATA_DIR = op.join(op.dirname(__file__), 'test_data')
TL_DIR = op.join(op.dirname(__file__), '..')
URL = "http://0.0.0.0:5005/"


class TestServer(unittest.TestCase):
    server_process = None
    test_string = """
Hallo Wereld!
Voor mij Adriaen van Renteregem geboren te Eindhoven
"""

    @classmethod
    def setUpClass(cls):
        print(f'CWD for subprocess wil be {TL_DIR}')
        # need to add TL_DIR to the Pythonpath on child_env
        my_env = os.environ.copy()
        print(f"current PYTHONPATH is {my_env.get('PYTHONPATH', '')}")
        my_env['PYTHONPATH'] = my_env.get('PYTHONPATH', '') + ':' + TL_DIR
        cls.server_process = subprocess.Popen(
            [
                'python3', 'server/fast_server.py',
                '--configfile', 'tests/test_data/server/config_test.json',
                '--n_parallel', '2'
            ],
            cwd=TL_DIR,
            env=my_env
        )
        wait_time = 50  # sleep while the process starts
        for t in range(wait_time):
            print('.', end='')
            time.sleep(1)
        print(cls.server_process.stdout)
        print(cls.server_process.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.server_process.terminate()
        cls.server_process.wait()

    def testWhole(self):
        server_response = requests.post(url=URL + 'pipeline', json={"input_data": self.test_string})
        full_json = server_response.json()
        self.assertEqual(full_json['message'], 'Success')
        json_data = full_json['results']
        schema.Validator(ignore=('ner_bert',))(json_data)  # skip for now as the schema requires BERT

    def testSplit(self):
        server_response = requests.post(url=URL + 'pipeline', json={
            "input_data": self.test_string,
            "steps": ["string_to_sentences", "post_correction", "modernisation"]
        })
        full_json = server_response.json()
        self.assertEqual(full_json['message'], 'Success')
        json_data = full_json['results']
        schema.Validator(ignore=('ner_bert', 'ner_lists'))(json_data)  # skip for now as the schema requires BERT

        server_response = requests.post(url=URL + 'pipeline', json={
            "input_data": json_data,
            "steps": ["ner_bert", "ner_lists"]
        })
        full_json = server_response.json()
        self.assertEqual(full_json['message'], 'Success')
        # schema.Validator()(json_data)  # skip for now as the schema requires BERT


if __name__ == "__main__":
    TestServer.setUpClass()
    ts = TestServer()
    ts.testWhole()
    ts.tearDownClass()
