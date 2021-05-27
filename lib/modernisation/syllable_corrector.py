import json
import os.path as op

from lib.modernisation.syllable_tokenizer import SyllableTokenizer
from lib import constants


class SyllableCorrector:
    """
    Modernisation strategy by substituting syllables and combinations thereof by their modernised equivalents. The
    default replacements used are given by the human-readable
    :download:`switches.json </../../pipeline_data/modernisation/switches.json>`.
    The splitting into syllables is handled by the
    :class:`SyllableTokenizer<lib.modernisation.syllable_tokenizer.SyllableTokenizer>` class.
    """
    def __init__(self, file_name=None):
        """
        :param file_name: The name of the switches file, defaults to `switches.json` the path is given relative to the
            `constants.DATA_DIR` directory.
        """
        file_name = file_name if file_name is not None else 'switches.json'
        switches_file = op.join(constants.DATA_DIR, 'modernisation', file_name)
        self.tokenizer = SyllableTokenizer()
        with open(switches_file) as f:
            self.switches = {
                tuple(hist.split(".")): tuple(mod.split("."))
                for hist, mod in json.load(f).items()
            }
        self.tree = self._create_tree()

    def __call__(self, word_form_lowercase, verbose=False):
        """
        :param word_form_lowercase: The lower case word-form to be modernised.
        :param verbose:  Flag whether to output extra info for debugging (True) or not (False).
        :return: A tuple, the first element indicated the result of applying the syllabel corrector, the second whether
            the regex changed the word-form (True) or not (False).
        """
        touched = False
        tags = self.tokenizer.encode(word_form_lowercase)
        translated_tags = []
        while tags:
            if verbose:
                print()
                print(translated_tags + ['-'] + tags)
                print()
            tag = tags[0]
            if tag not in self.tree:
                translated_tags.append(tag)
                tags.pop(0)
            else:
                res = self._translate_part(tags, self.tree[tag], 1)
                if res:
                    touched = True
                    replacement, nr_replacement = res
                else:
                    replacement, nr_replacement = [tag], 1
                if verbose:
                    print('a:', end=' ')
                    print(tags[:nr_replacement])
                    print('b:', end=' ')
                    print(replacement)
                    print()
                tags = tags[nr_replacement:]
                translated_tags += replacement
        translated_word = self.tokenizer.decode(translated_tags)
        return translated_word, touched

    def _add_leaves(self, tree, hist_t, mod_t, depth):
        first = hist_t[0]
        branch = tree.get(first, {})
        if len(hist_t) > 1:
            branch = self._add_leaves(branch, hist_t[1:], mod_t, depth + 1)
        else:
            branch = {None: mod_t}
        tree[first] = branch
        return tree

    def _create_tree(self):
        tree = {}
        for hist_t, mod_t in self.switches.items():
            first = hist_t[0]
            branch = tree.get(first, {})
            if len(hist_t) > 1:
                branch = self._add_leaves(branch, hist_t[1:], mod_t, 0)
            else:
                branch = {None: mod_t}
            tree[first] = branch
        return tree

    def _translate_part(self, tags, branch, nr_hist):
        #     print('O', branch)
        if len(tags) > 1 and tags[1] in branch:
            res = self._translate_part(tags[1:], branch[tags[1]], nr_hist + 1)
            if res:
                part, nr_hist = res
                #             print('A', part)
                return part, nr_hist
            elif None in branch:
                return branch[None], nr_hist
            #             print('B', branch.get(None, False))
            else:
                return False
        elif None in branch:
            return branch[None], nr_hist
        else:
            return False
