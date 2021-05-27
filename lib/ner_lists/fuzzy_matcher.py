import itertools
from difflib import SequenceMatcher
from heapq import nlargest
from typing import List, Dict

from lib.ner_lists.entity import Entity


class FuzzyMatcher:
    """
    Class to match words with list elements
    """
    def __init__(self, cutoff_score: float):
        """
        :param cutoff_score: Score below which to disregard matches.
        """
        self.cutoff_score = cutoff_score

    def score(self, word_group, word_list, cutoff_score=None):
        """
        :param cutoff_score: threshold for % of characters that should match between word_group and an el. of word_list
        :param word_group: a string, which is a concatenated list of words that are to be matched
        :param word_list: a list of (space-less concatenated) words that are compared to the word_group elements might
        be permutations of others, e.g. fortbatavia and bataviafort
        :return: a number, 0<=number<=1. It is the percentage of characters that match in the best match (from the list)
        """

        if cutoff_score is None:
            cutoff_score = self.cutoff_score

        word_group = ''.join(word_group).lower()

        results = self.get_close_matches_scores(word_group, word_list, 1, cutoff_score)
        score = 0
        match = None
        if len(results) > 0:
            score = results[0][0]
            match = results[0][1]
            if score == 1:
                # add a small score to longer words to prefer longer words (lexicographically)
                score = 1 + 0.001 * len(word_group)
        return score, match

    @staticmethod
    def get_close_matches_scores(word_group, possibilities, n, cutoff):
        """
        :param word_group: a string, might be multiple words concatenated without spaces.
        :param possibilities: list of strings. word_group will be compared to all strings in this list.
        :param n: positive integer. the top-n matches will be returned.
        :param cutoff: percentage of characters that should match before something is considered a match
        :return: either an empty list -> no match/hit
                or a size 2 tuple with score and word_group string
        """
        if not n > 0:
            raise ValueError("n must be > 0: %r" % (n,))
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
        matches = []
        s = SequenceMatcher()
        s.set_seq2(word_group)
        for possibility in possibilities:
            # TODO: there's probably a speed  gain if we replace the cutoff below with the "best score so far"
            s.set_seq1(possibility)
            if s.real_quick_ratio() >= cutoff and \
                    s.quick_ratio() >= cutoff:
                ratio = s.ratio()
                if ratio >= cutoff:
                    matches.append((ratio, possibility))

        # Move the best scorers to head of list
        best_match = nlargest(n, matches)
        # Strip scores for the best n matches
        return best_match


class FuzzyTreeMatcher(FuzzyMatcher):
    """
    Finds matches given a "tree", where a "tree" is a list of "searchables" grouped by a key element of the searchable.
    The key element is searched for first, after which different ordereings of the match are considered.
    """
    def __init__(self, tree: Dict[str, Dict], cutoff_score: float):
        """
        :param tree: The tree through which to search for entities, each key points to a list of "searchables"
            containing that key as one of the elements.
        :param cutoff_score: Score below which to disregard matches.
        """
        self.tree = tree
        super().__init__(cutoff_score)

    def find_matches(self, i_sentence: int, sentence: List[Dict], n_approx: int = 2):
        """
        :param i_sentence: An index of the sentence, used to match it to the correct place later on.
        :param sentence: List of words to match againse the :const:`tree`
        :param n_approx: The approximate expected number of words in a hit. It influences the threshold for the first
        hit. Setting it higher results in a higher threshold, causing faster performance but possible missed hits.
        Setting it too low results in poor performance.
        :return: *Yields* entities as they are found
        """
        first_word_cutoff_score = self.cutoff_score * (n_approx - 1) / n_approx
        for i, word in enumerate(sentence):
            first_score, first_match = self.score([word], list(self.tree.keys()), first_word_cutoff_score)
            if first_score >= first_word_cutoff_score:
                for entity in self.yield_matches(i_sentence, sentence, i, first_match):
                    yield entity

    def yield_matches(self, i_sentence: int, sentence: List[Dict], i: int, entity_key: str):
        """
        :param i_sentence: string index
        :param sentence: list of words
        :param i: index of word in sentence (which matches the key of an element in the tree-dictionary)
        :param entity_key: The tree is essentially a large dictionary. entity_key is the key of the most likely entity
        :return: *Yields* entities as they are found
        """
        for entity_words, canonicals in self.tree[entity_key].items():
            n = len(entity_words)
            possibilities = list(''.join(perm) for perm in itertools.permutations(entity_words))
            for offset in range(0,  n):
                i_offset = i - offset
                if i_offset < 0:
                    continue
                word_group = sentence[i_offset:i_offset + n]
                score, match = self.score(word_group, possibilities)
                if match:
                    for canonical in canonicals:
                        yield Entity(sentence=i_sentence,
                                     begin=i_offset,
                                     end=i_offset + n,
                                     score=score,
                                     searchable=' '.join(canonical["searchable"]),
                                     # TODO: see FindEntitiesGivenDirectList
                                     canonical_form=canonical["canonical_form"],
                                     extra_attributes=canonical["extra_attributes"]
                                     )


class FuzzyListMatcher(FuzzyMatcher):
    """
    Finds matches given a "list", a list of "searchables" grouped by a key element of the searchable.
    The key element is searched for first, after which different ordereings of the match are considered.
    """
    def __init__(self, fuzzy_list: Dict[str, Dict], cutoff_score: float):
        """
        :param fuzzy_list: The list through which to search for entities
        :param cutoff_score: Score below which to disregard matches.
        """
        self.fuzzy_list = fuzzy_list
        super().__init__(cutoff_score)

    def find_matches(self, i_sentence: int, sentence: List[Dict], n_approx: int = 2):
        """
        :param i_sentence: An index of the sentence, used to match it to the correct place later on.
        :param sentence: List of words to match againse the :const:`tree`
        :param n_approx: The approximate expected number of words in a hit. It influences the threshold for the first
        hit. Setting it higher results in a higher threshold, causing faster performance but possible missed hits.
        Setting it too low results in poor performance.
        :return: *Yields* entities as they are found
        """
        first_word_cutoff_score = self.cutoff_score * (n_approx - 1) / n_approx
        for i, word in enumerate(sentence):
            first_score, first_match = self.score(
                [word], [searchable[0] for searchable in self.fuzzy_list.keys()], first_word_cutoff_score)
            if first_score >= first_word_cutoff_score:
                for entity in self.yield_matches(i_sentence, sentence, i):
                    yield entity

    def yield_matches(self, i_sentence: int, sentence: List[Dict], i: int):
        """

        :param i_sentence: string index
        :param sentence: list of words
        :param i: index of word in sentence (which matches the key of an element in the tree-dictionary)
        """

        for searchable, canonicals in self.fuzzy_list.items():
            n = len(searchable)
            possibilities = [''.join(searchable)]
            word_group = sentence[i:i + n]
            score, match = self.score(word_group, possibilities)
            if match:
                for canonical in canonicals:
                    yield Entity(sentence=i_sentence,
                                 begin=i,
                                 end=i+n,
                                 score=score,
                                 searchable=' '.join(canonical["searchable"]),
                                 # TODO: see FindEntitiesGivenDirectList
                                 canonical_form=canonical["canonical_form"],
                                 extra_attributes=canonical["extra_attributes"]
                                 )
