import unittest
import json
import os.path as op
import glob

TEST_DATA_DIR = op.join(op.dirname(__file__), 'test_data')


class TestTools(unittest.TestCase):
    def __init__(self, method_name='runTest', path='.'):
        super().__init__(methodName=method_name)
        self.path = path
        with open(op.join(TEST_DATA_DIR, self.path, f'{self._testMethodName}_in.json')) as file:
            self.input = json.load(file)
        self.out = None
        self.ref = None

    def compare_output(self):
        with open(op.join(TEST_DATA_DIR, self.path, f'{self._testMethodName}_ref.json')) as file:
            self.ref = json.load(file)
        try:
            for ref_sentence, out_sentence in zip(self.ref, self.out):
                self.assertEqual(ref_sentence, out_sentence)
        except (AssertionError, FileNotFoundError) as error:
            with open(op.join(TEST_DATA_DIR, self.path, f'_{self._testMethodName}_out.json'), 'w') as file:
                json.dump(self.out, file, indent=4, sort_keys=False)
            raise error


class TestTestTools(TestTools):
    def test_good_weather(self):
        self.out = [[{**self.input[0][0], **{'breeze': 'lovely'}}]]
        self.compare_output()

    def test_bad_weather(self):
        with self.assertRaises(AssertionError):
            self.out = [[{**self.input[0][0], **{'breeze': 'horrendous'}}]]
            self.compare_output()


class TestLocalDirectoryStructure(unittest.TestCase):
    # skip this test when there is no constants.DATA_DIR, i.e. on the server
    ignore_dirs = ['ner_bert', 'server']

    local_data_dir = op.join(op.dirname(__file__), '..', 'pipeline_data')

    @unittest.skipUnless(op.isdir(local_data_dir), "Skip on bitbucket")
    def test_structure_equal(self):
        test_dirs = [d for d in glob.glob(op.join(TEST_DATA_DIR, '*')) if op.isdir(d)]
        local_dirs = [d for d in glob.glob(op.join(self.local_data_dir, '*')) if op.isdir(d)]
        test_dir_names = [op.basename(d) for d in test_dirs]
        local_dir_names = [op.basename(d) for d in local_dirs]
        for ignore_d in self.ignore_dirs:
            if ignore_d in local_dir_names:
                idx = local_dir_names.index(ignore_d)
                del local_dirs[idx]
                del local_dir_names[idx]
            if ignore_d in test_dir_names:
                idx = test_dir_names.index(ignore_d)
                del test_dirs[idx]
                del test_dir_names[idx]
        self.assertEqual(test_dir_names, local_dir_names)
        for test_d, local_d in zip(test_dirs, local_dirs):
            test_files = [op.basename(f) for f in glob.glob(op.join(test_d, '*'))]
            local_files = [op.basename(f) for f in glob.glob(op.join(local_d, '*'))]
            self.assertEqual(test_files, local_files)


if __name__ == '__main__':
    TestLocalDirectoryStructure().test_structure_equal()
