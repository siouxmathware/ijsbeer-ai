""" SpellChecker Module; simple, intuitive spell checker based on the post by
    Peter Norvig. See: https://norvig.com/spell-correct.html """
from __future__ import absolute_import, division, unicode_literals

import os
import json
import string
from collections import Counter

from spellchecker.utils import load_file, write_file, _parse_into_words, ENSURE_UNICODE
from spellchecker import WordFrequency


class SpellChecker(object):
    """ The SpellChecker class encapsulates the basics needed to accomplish a
        simple spell checking algorithm. It is based on the work by
        Peter Norvig (https://norvig.com/spell-correct.html)

        Args:
            language (str): The language of the dictionary to load or None \
            for no dictionary. Supported languages are `en`, `es`, `de`, fr` \
            and `pt`. Defaults to `en`
            local_dictionary (str): The path to a locally stored word \
            frequency dictionary; if provided, no language will be loaded
            distance (int): The edit distance to use. Defaults to 2.
            case_sensitive (bool): Flag to use a case sensitive dictionary or \
            not, only available when not using a language dictionary.
        Note:
            Using a case sensitive dictionary can be slow to correct words."""

    __slots__ = ["_distance", "_word_frequency", "_start_set", "_end_set",
                 "_tokenizer", "_case_sensitive", "_start_length"]

    def __init__(
        self,
        language="en",
        local_dictionary=None,
        distance=2,
        tokenizer=None,
        case_sensitive=False,
        start_length=3
    ):
        self._distance = None
        self.distance = distance  # use the setter value check

        self._tokenizer = _parse_into_words
        if tokenizer is not None:
            self._tokenizer = tokenizer

        self._case_sensitive = case_sensitive if not language else False
        self._word_frequency = WordFrequency(self._tokenizer, self._case_sensitive)

        if local_dictionary:
            self._word_frequency.load_dictionary(local_dictionary)
        elif language:
            filename = "{}.json.gz".format(language.lower())
            here = os.path.dirname(__file__)
            full_filename = os.path.join(here, "resources", filename)
            if not os.path.exists(full_filename):
                msg = (
                    "The provided dictionary language ({}) does not " "exist!"
                ).format(language.lower())
                raise ValueError(msg)
            self._word_frequency.load_dictionary(full_filename)

        self._start_length = start_length
        self._start_set = self._fill_start_set()
        self._end_set = self._fill_end_set()
        pass

    def _fill_start_set(self):
        return set(
            word[:length]
            for word in self._word_frequency.dictionary
            for length in range(1, self._start_length + 1)
        ).union('')

    def _fill_end_set(self):
        return set(
            word[-length:]
            for word in self._word_frequency.dictionary
            for length in range(1, self._start_length + 1)
        ).union('')

    def __contains__(self, key):
        """ setup easier known checks """
        key = ENSURE_UNICODE(key)
        return key in self._word_frequency

    def __getitem__(self, key):
        """ setup easier frequency checks """
        key = ENSURE_UNICODE(key)
        return self._word_frequency[key]

    @property
    def word_frequency(self):
        """ WordFrequency: An encapsulation of the word frequency `dictionary`

            Note:
                Not settable """
        return self._word_frequency

    @property
    def distance(self):
        """ int: The maximum edit distance to calculate

            Note:
                Valid values are 1 or 2; if an invalid value is passed, \
                defaults to 2 """
        return self._distance

    @distance.setter
    def distance(self, val):
        """ set the distance parameter """
        tmp = 2
        try:
            int(val)
            if 0 < val <= 2:
                tmp = val
        except (ValueError, TypeError):
            pass
        self._distance = tmp

    def split_words(self, text):
        """ Split text into individual `words` using either a simple whitespace
            regex or the passed in tokenizer

            Args:
                text (str): The text to split into individual words
            Returns:
                list(str): A listing of all words in the provided text """
        # word = ENSURE_UNICODE(word).lower() if not self._case_sensitive else ENSURE_UNICODE(word)
        # text = ENSURE_UNICODE(text)
        return self._tokenizer(text)

    def export(self, filepath, encoding="utf-8", gzipped=True):
        """ Export the word frequency list for import in the future

             Args:
                filepath (str): The filepath to the exported dictionary
                encoding (str): The encoding of the resulting output
                gzipped (bool): Whether to gzip the dictionary or not """
        data = json.dumps(self.word_frequency.dictionary, sort_keys=True)
        write_file(filepath, encoding, gzipped, data)

    def word_probability(self, word, total_words=None):
        """ Calculate the probability of the `word` being the desired, correct
            word

            Args:
                word (str): The word for which the word probability is calculated
                total_words (int): The total number of words to use in the
                    calculation; use the default for using the whole word
                    frequency
            Returns:
                float: The probability that the word is the correct word """
        if total_words is None:
            total_words = self._word_frequency.total_words
        # we do not want to waste time ensuring things are unicode
        # word = ENSURE_UNICODE(word)
        return self._word_frequency.dictionary[word] / total_words

    def correction(self, word):
        """ The most probable correct spelling for the word

            Args:
                word (str): The word to correct
            Returns:
                str: The most likely candidate """
        # we do not want to waste time ensuring things are unicode
        # word = ENSURE_UNICODE(word)
        candidates = list(self.candidates(word))
        return max(sorted(candidates), key=self.word_probability)

    def candidates(self, word):
        """ Generate possible spelling corrections for the provided word up to
            an edit distance of two, if and only when needed

            Args:
                word (str): The word for which to calculate candidate spellings
            Returns:
                set: The set of words that are possible candidates """
        # we do not want to waste time ensuring things are unicode
        # word = ENSURE_UNICODE(word)
        if self._known_single(word):  # short-cut if word is correct already
            return {word}

        if not self._check_if_should_check(word):
            return {word}

        # get edit distance 1...
        res = [x for x in self.edit_distance_1(word)]
        tmp = self.known(res)
        if tmp:
            return tmp
        # if still not found, use the edit distance 1 to calc edit distance 2
        if self._distance == 2:
            tmp = self.known([x for x in self.__edit_distance_alt(res)])
            if tmp:
                return tmp
        return {word}

    def known(self, words):
        """ The subset of `words` that appear in the dictionary of words

            Args:
                words (list): List of words to determine which are in the
                    corpus
            Returns:
                set: The set of those words from the input that are in the
                    corpus
        """
        # we do not want to waste time ensuring things are unicode
        # words = [ENSURE_UNICODE(w) for w in words]
        tmp = [w if self._case_sensitive else w.lower() for w in words]
        return set(
            w
            for w in tmp
            if w in self._word_frequency.dictionary and self._check_if_should_check(w)
        )

    def _known_single(self, word):
        return word in self._word_frequency.dictionary and self._check_if_should_check(word)

    def unknown(self, words):
        """ The subset of `words` that do not appear in the dictionary

            Args:
                words (list): List of words to determine which are not in the
                    corpus
            Returns:
                set: The set of those words from the input that are not in
                    the corpus
        """
        # we do not want to waste time ensuring things are unicode
        # words = [ENSURE_UNICODE(w) for w in words]
        tmp = [
            w if self._case_sensitive else w.lower()
            for w in words
            if self._check_if_should_check(w)
        ]
        return set(w for w in tmp if w not in self._word_frequency.dictionary)

    def edit_distance_1(self, word):
        """ Compute all strings that are one edit away from `word` using only
            the letters in the corpus

            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance one from the \
                provided word """
        # we do not want to waste time ensuring things are unicode
        # word = ENSURE_UNICODE(word).lower() if not self._case_sensitive else ENSURE_UNICODE(word)
        word = word.lower() if not self._case_sensitive else word
        if self._check_if_should_check(word) is False:
            return {word}
        letters = self._word_frequency.letters
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        # we can remove any splits for which L does not occur as a start in our dict?
        # only really possible if self._distance == 1, but will do it anyway for now
        assert(self.distance == 1), 'Code hack implemented that does not work for d > 1'
        splits = [
            (L, R)
            for L, R in splits
            # R[2:] is the part that will always be the end of the combinations,
            # as for transposes R[0] and R[1] are switched, these must not necessarily be in the known ends
            if L[:self._start_length] in self._start_set and R[2:][-self._start_length:] in self._end_set
        ]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edit_distance_2(self, word):
        """ Compute all strings that are two edits away from `word` using only
            the letters in the corpus

            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance two from the \
                provided word """
        # word = ENSURE_UNICODE(word).lower() if not self._case_sensitive else ENSURE_UNICODE(word)
        word = word.lower() if not self._case_sensitive else word
        return [
            e2 for e1 in self.edit_distance_1(word) for e2 in self.edit_distance_1(e1)
        ]

    def __edit_distance_alt(self, words):
        """ Compute all strings that are 1 edits away from all the words using
            only the letters in the corpus

            Args:
                words (list): The words for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance two from the \
                provided words """
        # we do not want to waste time ensuring things are unicode
        # words = [ENSURE_UNICODE(w) for w in words]
        tmp = [
            w if self._case_sensitive else w.lower()
            for w in words
            if self._check_if_should_check(w)
        ]
        return [e2 for e1 in tmp for e2 in self.edit_distance_1(e1)]

    def _check_if_should_check(self, word):
        if len(word) == 1 and word in string.punctuation:
            return False
        if len(word) > self._word_frequency.longest_word_length + 3:  # magic number for removal of up to 2 letters.
            return False
        try:  # check if it is a number (int, float, etc)
            float(word)
            return False
        except ValueError:
            pass

        return True
