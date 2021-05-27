from abc import ABCMeta, abstractmethod


class GetWords(metaclass=ABCMeta):
    """
    Abstract base class for the GetWordsNer and GetWordsBert classes
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """"""
        pass

    @abstractmethod
    def __call__(self, sentences):
        """"""
        pass

    @classmethod
    def from_string(cls, getter_name: str):
        """
        :param getter_name: Name of the derived class, either "NER" or "BERT"
        :return: The uninstantiated class of the desired kind.
        """
        if getter_name == 'NER':
            return GetWordsNer
        elif getter_name == 'BERT':
            return GetWordsBert
        else:
            raise ValueError(f"The specified gett_name={getter_name} is not available, pick either NER or BERT.")


class GetWordsNer(GetWords):
    def __init__(self, **kwargs):
        """
        :param kwargs: any additional key-word arguments are passed to GetWords
        """
        super().__init__(**kwargs)

    def __call__(self, sentences):
        """
        :return: A list of lists, where the inner lists contain all words that have word['ner'] == True for a sentence.
        """
        return [[w for w in s if w['ner']] for s in sentences]


class GetWordsBert(GetWords):
    def __init__(self, entity_type, **kwargs):
        """
        :param entity_type: The type of entity (e.g. location or person) which should be searched.
        :param kwargs: any additional key-word arguments are passed to GetWords
        """
        self.entity_type = entity_type
        super().__init__(**kwargs)

    def __call__(self, sentences):
        """
        :param sentences: list of lists. Each element is a dict that represents a word. The intermediate level lists
            represent a sentence each. spaces and comma's are still considered words at this level
        :return: found_entities. A list of lists. Each intermediate list represents a BERT-found entity of type
            `self.type_of_list` (e.g. a location)
        """

        words_bert_results = []
        for sentence in sentences:
            location = []
            for word in sentence:
                if word['ner']:
                    label = word['labels']['BERT'][self.entity_type]['bio']
                    if label == "B":
                        if location:
                            words_bert_results.append(location)
                        location = [word]
                    elif label == "I":
                        if len(location) > 0:
                            location.append(word)
            if len(location) > 0:
                words_bert_results.append(location)
        return words_bert_results
