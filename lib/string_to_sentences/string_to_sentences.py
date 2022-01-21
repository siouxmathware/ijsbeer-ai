import logging
import re
from typing import List, Pattern, Dict

from lib.string_to_sentences.replacer import Replacer

logger = logging.getLogger(__name__)


class StringToSentences:
    """
    Converts a given string to a list of lists of dicts, see
    :meth:`__call__<lib.string_to_sentences.string_to_sentences.StringToSentences.__call__>` for final result form.
    """
    replacer = Replacer(from_key='word', to_key='word')
    split_rule = re.compile('([ \n])')
    last_word_regex = re.compile(r"(^[^ ]+[a-z])([.!?]$)")
    ner_pattern = re.compile("(^[^a-zA-Z0-9:;.,?!\"’'`]$)")

    def __init__(self, **_):
        """
        :param _: Unused kwarg argument, included for symmetry with the other pipeline steps.
        """
        self.compiled_line_word_split_patterns = self.compile_line_word_split_patterns()
        self.double_newline_pattern = [re.compile(p) for p in ["^\n$", "^\n$"]]
        self.sentence_end_pattern = [re.compile(p) for p in [self.last_word_regex, "^ $", "^[A-Z]"]]

    def __call__(self, text):
        """
        :param text: A string representing the historic text. Words are considered to be separated, broadly speaking, by
            either spaces and/or newlines. A double newline is interpreted to separate sentences. Other methods are also
            used to separate sentences based on punctuation.
        :return: a list of lists of words object with format example given below.

        .. code-block:: python

            out = {"word": "Weerld!", "begin_char": 7, "end_char": 14, "ner": True}



        Here begin/end char represents the original place inside text, can be used for locating this word in the
        original document.
        The "ner" key tells if this word should be used for named entity recognition.
        """
        words = self.get_list_of_dict_word_and_chars(text)
        words = self.line_word_splits(words)
        words = self.set_ner_flag(words)
        sentences = self.split_in_sentences(words)

        return sentences

    @staticmethod
    def get_list_of_dict_word_and_chars(text):
        """
        Split the string into words where a word can be a normal word, but also a linebreak, space or any other special
        character.
        """
        words = StringToSentences.split_rule.split(text)
        word_lengths = [len(word) for word in words]
        end_chars = []
        end_char = 0
        for word_length in word_lengths:
            end_char += word_length
            end_chars.append(end_char)
        begin_chars = [0] + end_chars[:-1]

        # Combine to dict adding ner info to it
        word_list_with_labels = [{
            "word": word,
            "begin_char": begin_char,
            "end_char": end_char,
            "ner": True
        } for word, begin_char, end_char in zip(words, begin_chars, end_chars) if word]

        return word_list_with_labels

    def split_in_sentences(self, words):
        """
        Deal with the line breaks.
        """
        index = []
        sentences = []

        # Generate index where to split on
        index += self._return_matches(words, self.double_newline_pattern, start=1)
        # Example: world. Hello but not J. Jacobs
        index += self._return_matches(words, self.sentence_end_pattern, start=2)

        # Add beginning/end, sort index and remove double indices
        index = list(set([0] + index + [len(words)]))
        index.sort()

        # Retrieve the sentences
        for a, b in zip(index[:-1], index[1:]):
            sentence = self._make_sentence(words[a:b])
            sentences.append(sentence)

        # Remove empty sentences
        # TODO: is this really a possibility?
        sentences = [sentence for sentence in sentences if len(sentence) > 0]

        return sentences

    def _make_sentence(self, words):
        end = self.replacer.replace_words(
            words[-2:],
            [self.last_word_regex],
            self.replacer.splittor(self.last_word_regex)
        )
        return words[:-2] + end

    def set_ner_flag(self, words):
        """
        Set ner to False for words that match the NER (anti)pattern
        """
        def replacement_function(match_words):
            for word in match_words:
                word['ner'] = False
            return match_words

        words = self.replacer.replace_words(words, [self.ner_pattern], replacement_function)

        return words

    @staticmethod
    def compile_line_word_split_patterns():
        patterns = [
            ["^(.+[a-zA-Z])$", "^( )$", "^(\n$)", "^([„-][a-zA-Z].+$)"],
            ["^(.+[a-zA-Z])$", "^(\n)$", "^([„-][a-zA-Z]{1}.+$)"],
            ["^(.+[a-zA-Z][„-])$", "^(\n)$", "(^ $)", "^([„-]?[a-zA-Z].+$)"],
            ["^(.+[a-zA-Z][„-])$", "^(\n)$", "(^[„-]?[a-zA-Z].+$)"]
        ]
        return [[re.compile(p) for p in pattern] for pattern in patterns]

    def line_word_splits(self, words):
        """
        Deal with detectable word splits caused by line breaks. For applying ner and also do post correction
        or modernisation it is better to look at the whole word.
        """
        # Example: aan \n„bod => aanbod
        for compiled in self.compiled_line_word_split_patterns:
            words = self.replacer.replace_words(words,
                                                compiled,
                                                self.replacer.joinor())
        return words

    @staticmethod
    def _return_matches(words: List[Dict], pattern: List[Pattern], start=0):
        """
        Generic method to return the index of when a sublist matches a pattern.
        """
        n = len(pattern)
        matches = []
        for i in range(len(words) - n + 1):
            words_sub = words[i:i + n]
            if all([p.search(w['word']) for p, w in zip(pattern, words_sub)]):
                matches.append(i + start)
        return matches


if __name__ == "__main__":
    sts = StringToSentences()
    txt = """
ook is van
batavie gekomen ten
anker de Chialoup d’ Tal„
„meije met haar hoog Ed:
apporte missive gedateerd
8: stantij
"""[1:-1]  # chop of first and last \n
    result = sts(txt)
    print('\n'.join(''.join(w['word'] for w in s) for s in result))
