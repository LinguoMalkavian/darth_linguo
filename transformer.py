import tree_handler
import re

# TODO find a good dictionary for spanish
# Load the dictionary
diccionario = ["rojo", "azul", "roja", "azules"]


class corruptor ():
    def __init__(self, kind):
        self.kind = kind
        pass

    def test_possible(self, sentence):
        pass

    def transform(self, sentence):
        pass

# Subject Remover


class SubjectRemover(corruptor):
    def test_possible(self, sentence_tree):
        pass

    def transform(self, sentence_tree):
        pass

# Verb Remover


class VerbRemover(corruptor):
    def test_possible(self, sentence):
        pass

    def transform(self, sentence):
        pass


# Verbal inflection corruption
class VerbalInflCorruptor(corruptor):
    def test_possible(self, sentence):
        pass

    def transform(self, sentence):
        pass


# Adjectival inflection corruption
class AdjInflCorruptor(corruptor):
    inf_ADJ_regex = re.compile("([A-Za-záéíóúñÑÁÉÍÓÚ]+)(a|o|as|os|es)$")

    def test_possible(self, sentence):
        token_list = sentence.tokens
        for token in token_list:
            if token.pos == tree_handler.adj_TAG:
                if AdjInflCorruptor.inf_ADJ_regex.matches(token.text):
                    return True
        return False

    def transform(self, sentence):
        token_list = sentence.tokens
        for token in token_list:
            if token.pos == tree_handler.adj_TAG:
                if AdjInflCorruptor.inf_ADJ_regex.matches(token.text):
                    match = AdjInflCorruptor.inf_ADJ_regex.match(token.text)
                    root = match.group(1)
                    infl = match.group(2)
                    pos_inf = ["a", "o", "as", "os", "es"]
                    pos_inf.remove(infl)
                    for new_inf in pos_inf:
                        if root + new_inf in diccionario:
                            new_adj = root + new_inf
                            before = sentence.text[:token.beg]
                            after = sentence.text[token.end:]
                            newText = before + new_adj + after
                            return newText
        return -1
