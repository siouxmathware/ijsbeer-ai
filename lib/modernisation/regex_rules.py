import logging
import re
import os.path as op
from typing import List

from lib import constants

LOGGER = logging.getLogger(__name__)
RULE_SEP = '@'


class RegexRules:
    """
    Deals with modernising historical Dutch words based on regular expressions.
    All regular expressions are to be divided in three categories:

    1. Line-break regular expressions that define how words can be divided over multiple lines in the historical text.

    2. Direct word-to-word translations.

    3. Further regular expressions on parts of words, these are all heuristically found.

    The first and third are both applied by calls to the
    :meth:`regex_subs <lib.modernisation.regex_rules.RegexRules.regex_subs>` method. The second is handled by the
    :meth:`dict_lookup <lib.modernisation.regex_rules.RegexRules.dict_lookup>` method.
    """
    tree_regexes = {
        'initial_letter': {
            'regex_form': re.compile(r'^\^[a-z0-9].+'),
            'key_from_regex_fun': lambda r: r[1],
            'key_from_word_fun': lambda w: w[0]
        },
        None: {
            'regex_form': re.compile(r'.+'),
            'key_from_regex_fun': lambda r: None,
            'key_from_word_fun': lambda w: None
        }
    }

    def __init__(self):
        """
        Loads the line-break rules, dictionaries and other rules from various sources.
        """
        self.dict_per_word = {}
        self.conflicts = {}
        self.rules_per_word_per_file = {
            'line_breaks': {},
            'abbreviations': {},
            'orthographic_rules': {},
            'volunteer_corpus': {},
            'volunteer_corpus_2021-02-23': {},
        }
        for val in self.rules_per_word_per_file.values():
            for tree in self.tree_regexes.keys():
                val[tree] = {}
        self._get_rules()
        self.call_counts = {}

    def dict_lookup(self, word_form_lowercase: str):
        """
        :param word_form_lowercase: Lower case form of the word to modernise
        :return: A tuple, the first element indicated the found replacement (if any), the second whether a replacement
            was found (True) or not (False).
        """
        if word_form_lowercase in self.dict_per_word:
            return self.dict_per_word[word_form_lowercase], True
        else:
            return word_form_lowercase, False

    def regex_subs(self, word_form_lowercase: str, filename: str):
        """
        :param word_form_lowercase: Lower case form of the word to modernise
        :param filename: The filename for which to apply regexes.
        :return: A tuple, the first element indicated the result of applying the regular expressions, the second whether
            the regex changed the word-form (True) or not (False).
        """
        regex_word_form = word_form_lowercase

        if word_form_lowercase:
            # To avoid all kinds of things breaking if the word_form_lowercase is empty
            for tree, leaf in self.tree_regexes.items():
                key = leaf['key_from_word_fun'](word_form_lowercase)
                if key in self.rules_per_word_per_file[filename][tree]:
                    for rule, replacement in self.rules_per_word_per_file[filename][tree][key]:
                        regex_word_form = rule.sub(replacement, regex_word_form)
                        self.call_counts[filename] = self.call_counts.get(filename, 0) + 1
        return regex_word_form, regex_word_form != word_form_lowercase

    def _get_rules(self):
        """
        :return: None, but modifies the object's dict_per_word, conflicts and rules_per_word_per_file dictionaries.
        """
        for file in self.rules_per_word_per_file:
            with open(op.join(constants.DATA_DIR, 'modernisation', f'{file}.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                self._get_rules_for_file(lines, file)
        for word, conflicts in self.conflicts.items():
            LOGGER.warning(f"Found multiple conflicting replacements for {word}: {', '.join(conflicts)}")

    def _get_rules_for_file(self, lines: List[str], filename: str):
        """
        :param lines: The lines of a particular file
        :param filename: The name of the file, used only to place rules on the correct key in rules_per_word_per_file.
        :return: None, but modifies the object's dict_per_word, conflicts and rules_per_word_per_file dictionaries.
        """
        for line in lines:
            if len(line) > 0 and RULE_SEP in line:
                rule, replacement = self._split_line(line)
                if self._is_word_rule(rule):
                    word = rule[1:-1]  # trim the ^ and $
                    if word in self.dict_per_word and self.dict_per_word[word] != replacement:
                        self.conflicts[word] = self.conflicts.get(word, [self.dict_per_word[word]])
                        self.conflicts[word].append(replacement)
                        self.dict_per_word[word] = word
                    else:
                        self.dict_per_word[word] = replacement
                else:
                    self._add_rule_to_per_file_dict(filename, rule, replacement)

    def _add_rule_to_per_file_dict(self, filename, rule, replacement):
        tree = self._get_tree_for_rule(rule)
        key = self.tree_regexes[tree]['key_from_regex_fun'](rule)
        regex_list = self.rules_per_word_per_file[filename][tree].get(key, [])
        regex_list.append(tuple([re.compile(rule), replacement]))
        self.rules_per_word_per_file[filename][tree][key] = regex_list

    def _get_tree_for_rule(self, rule):
        for tree, leaf in self.tree_regexes.items():
            if leaf['regex_form'].match(rule):
                return tree
            return None

    def _get_trees_for_word(self, word):
        for tree, leaf in self.tree_regexes.items():
            key = leaf['key_from_word_fun'](word)
            yield tree, key

    @staticmethod
    def _is_word_rule(rule):
        if re.match(r'\^[a-z0-9 ]+\$', rule):
            # return true if the rule is simply a literal word, with no special characters
            # Note that it is not a problem if we are too strict here, but it IS if we are too lax
            return True
        else:
            return False

    @staticmethod
    def _split_line(line):
        split = line.split(RULE_SEP)
        assert len(split) == 2, f"Line {line} does not split to two elements on splitting with {RULE_SEP}"
        rule = split[0]
        replacement = split[1]
        return rule, replacement
