import re
from typing import List, Dict, Union, Callable, Pattern


class Replacer:
    """
    Handles several types of regex-based on operations on one or multiple words.
    """
    def __init__(self, from_key: str, to_key: str):
        """
        :param from_key: The key of the word dict that is the input for the methods, e.g. "word"
        :param to_key: The key of the word dict that is to contain the output of the methods, e.g. "word"
        """
        self.from_key = from_key
        self.to_key = to_key

    def return_matches(self, words: List[Dict], patterns: List[Union[str, Pattern]], start: int = 0):
        """
        :param words: List of words through which to search for the pattern
        :param patterns: The patterns to search a subsequence of the words is matched against the sequence of patters:
            each word must match the appropriate pattern
        :param start: Which element of the patterns is considered to be the start of the match.
        :return: Indices of matches, offset by the start parameter.
        """
        n = len(patterns)
        matches = []
        for i in range(len(words) - n):
            words_sub = words[i:i + n]
            if all([re.search(p, w[self.from_key]) for p, w in zip(patterns, words_sub)]):
                matches.append(i + start)
        return matches

    def get_replacement_inputs(self, words_sub: List[Dict], patterns: List[Pattern]):
        """
        :param words_sub: The words that are found to be a match to the patterns.
        :param patterns: The patterns to which the words were matched.
        :return: Those parts of the words_sub that match the group(s) in the corresponding regex patterns.
        """
        replacement_words = []
        for word, pattern in zip(words_sub, patterns):
            match = pattern.match(word[self.from_key])
            new_form = ''.join(match.groups()) if match.groups() else ''
            new_word = word.copy()
            new_word[self.from_key] = new_form
            replacement_words.append(new_word)
        return replacement_words

    def replace_words(self, words: List[Dict], patterns: List[Pattern],
                      replace_function: Callable, recursive: bool = False):
        """
        :param words: The list of words in which to search for patterns and apply replacements.
        :param patterns: The patterns to search.
        :param replace_function: The replace_function is called on the words found to match the patterns and the output
            is placed in the list of words in their place.
        :param recursive: Whether to apply the replacement also to words that have already been replaced. If False
            (default), the search will skip to the end of the replaced words after a replacement.
        :return:
        """
        n = len(patterns)
        i = 0
        while i < len(words) - n + 1:
            if all(
                    p.search(w[self.from_key]) for p, w in zip(patterns, words[i:i + n])
            ):
                words_sub = words[i:i + n]
                replacements = replace_function(self.get_replacement_inputs(words_sub, patterns))
                for j, word_sub in enumerate(words_sub):
                    word = words.pop(i)
                    assert word == word_sub
                for replacement in replacements[::-1]:
                    words.insert(i, replacement)

                if not recursive:
                    i += len(replacements) - 1

            i += 1
        return words

    def joinor(self):
        """
        :return: Function (N.B. the method returns a function) that will take a series of words and join them into a
            single word. To be used as an input for
            ``Replacer.replace_words(..., replacement_function=Replacer.joinor(), ...)``
        """
        def replacement_function(match_words):
            new_form = ''.join(w[self.from_key] for w in match_words)
            new_word = match_words[0].copy()
            new_word[self.to_key] = new_form
            new_word['begin_char'] = match_words[0]['begin_char']
            new_word['end_char'] = match_words[-1]['end_char']
            return [new_word]
        return replacement_function

    def splittor(self, regex: Pattern):
        """
        :param regex: The regex that splits a word into multiple words. Regex groups indicate what should become new
            words.
        :return: Function (N.B. the method returns a function) that will take a single word and split it into several
            different words. To be used as an input for
            ``Replacer.replace_words(..., replacement_function=Replacer.splittor(), ...)``
        """
        def take_slice(orig_word, reg):
            if reg[0] == 0:  # the first word keeps the other fields
                word = orig_word.copy()
            else:  # subsequent words are empty
                word = {key: '' for key, val in orig_word.items()}
                word['begin_char'] = None
                word['end_char'] = None
                word['ner'] = orig_word['ner']
            word[self.to_key] = orig_word[self.from_key][reg[0]:reg[1]]
            word['begin_char'] = orig_word['begin_char'] + reg[0]
            # count the end_char from the end in case something was removed in the middle
            # this happens in line-break words, and might happen otherwise too.
            word['end_char'] = orig_word['end_char'] + reg[1] - len(orig_word[self.from_key])
            return word

        def replacement_function(match_words):
            original_word = match_words[0]
            match = regex.match(original_word[self.from_key])
            if match:
                # N.B. regs[0] is the whole match (this makes the regex groups correspond to the usual \1, \2 etc.)
                words = [take_slice(original_word, reg) for reg in match.regs[1:]]
                return words
            else:
                return match_words
        return replacement_function
