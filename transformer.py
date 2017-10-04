import tree_handler
import re
import random

# Load the dictionary
lemma_list = []
lemma_file = open("lemario_snowball.txt", "r")
for line in lemma_file.readlines():
    lemma_list.append(line.strip())
# Load the inflected verb form list
verb_list = []
verb_file = open("verbos-espanol-conjugaciones.txt")
for line in lemma_file.readlines():
    lemma_list.append(line.strip())


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
class VerbInflCorruptor(corruptor):
    main_verb_tag = re.compile("vm[is][cfips]000")
    verb_infl_regex = re.compile("([A-Za-záéíóúñÑÁÉÍÓÚ]+)(o|s|e|)$")

    def test_possible(self, sentence):
        token_list = sentence.tokens
        for token in token_list:
            if token.pos == tree_handler.adj_TAG:
                if VerbInflCorruptor.inf_ADJ_regex.matches(token.text):
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
                        if root + new_inf in lemma_list:
                            new_adj = root + new_inf
                            before = sentence.text[:token.beg]
                            after = sentence.text[token.end:]
                            newText = before + new_adj + after
                            return newText
        return -1


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
        # List to store all codified possible transforms
        possib = []
        token_list = sentence.tokens
        for token in token_list:
            if token.pos == tree_handler.adj_TAG:
                if AdjInflCorruptor.inf_ADJ_regex.matches(token.text):
                    match = AdjInflCorruptor.inf_ADJ_regex.match(token.text)
                    root = match.group(1)
                    infl = match.group(2)
                    pos_inf = ["a", "o", "as", "os", "es"]
                    pos_inf.remove(infl)
                    random.shuffle(pos_inf)
                    for new_inf in pos_inf:
                        new_word = root + new_inf
                        if new_word in lemma_list:
                            rep = [new_word, token.beg, token.end]
                            possib.append(rep)
        if len(possib) != 0:
            choice = random.choose(possib)
            new_adj = choice[0]
            before = sentence.text[:choice[1]]
            after = sentence.text[choice[2]:]
            newText = before + new_adj + after
            return newText
        else:
            return -1
