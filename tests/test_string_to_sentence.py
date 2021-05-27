"""
Module to test the "ners" module
"""

import unittest

from lib.string_to_sentences.string_to_sentences import StringToSentences


class TestCall(unittest.TestCase):
    """
    Test the call function.
    """

    def test_correct_format(self):
        """
        Simple test case, tests correct format.
        """
        # Given
        string_to_sentences = StringToSentences()
        text = "hello world"
        essential_keys = ["word", "begin_char", "end_char"]

        # When
        result = string_to_sentences.__call__(text)

        # Then
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all([isinstance(s, list) and len(s) > 0 for s in result]))
        self.assertTrue(all([isinstance(w, dict) for s in result for w in s]))
        self.assertTrue(all([all([x in w.keys() for x in essential_keys]) for s in result for w in s]))

    def test_empty(self):
        """
        Empty test case.
        """
        # Given
        string_to_sentences = StringToSentences()
        text = ""
        expected_result = []

        # When
        result = string_to_sentences.__call__(text)

        # Then
        self.assertEqual(result, expected_result)

    def test_simple_result(self):
        """
        Test simple result
        """
        # Given
        string_to_sentences = StringToSentences()
        text = "hello world"
        expected_result = [["hello", " ", "world"]]

        # When
        result = string_to_sentences.__call__(text)

        # Then
        self.assertEqual([[w["word"] for w in s] for s in result], expected_result)

    def test_chars_not_overlap(self):
        """
        Test chars not overlap.
        """
        # Given
        string_to_sentences = StringToSentences()
        text = """
        Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum
        sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies
        nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel,
        aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum
        felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate
        eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante,
        dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum.
        Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui.
        """

        # When
        result = string_to_sentences.__call__(text)
        base = [False]*len(text)
        begin_end_chars = [(w["begin_char"], w["end_char"])for s in result for w in s]

        # Then
        for begin, end in begin_end_chars:
            n = end - begin
            self.assertGreater(n, 0)
            self.assertFalse(all(base[begin:end]))
            base[begin:end] = [True]*n
