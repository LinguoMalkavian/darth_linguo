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
    verb_list.append(line.strip())


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
    # Version with all main verbs
    # main_verb_tag = re.compile("vm[is][cfips]000")
    # Version with only the three most common tenses
    main_verb_tag = re.compile("vm[i][ips]000")

    root_str = "([A-Za-záéíóúñÑÁÉÍÓÚ]+)"

    # Morphological model of the present indicative
    pres_ind_tag = "vmip000"
    pres_infl = "(o|es|és|ás|e|emos|éis|en|as|a|amos|áis|an|imos|ís)$"
    verb_pr_ind_regex = re.compile(root_str + pres_infl)

    # Morphological model of regular past imprefect of the indicative
    imp_ind_tag = "vmii000"
    imperf_infl = "((aba|ía)(|s|mos|is|n))$"
    verb_imp_ind_regex = re.compile(root_str + imperf_infl)

    # Morphological model of regular simple past of the indicative
    perf_ind_tag = "vmis000"
    perf_infl = "(é|aste|ó|amos|asteis|aron|í|iste|ió|imos|isteis|ieron)$"
    verb_ps_ind_regex = re.compile(root_str + perf_infl)

    def test_possible(self, sentence):
        token_list = sentence.tokens
        for token in token_list:
            if token.pos == tree_handler.VerbInflCorruptor.main_verb_tag:
                return True
        return False

    def transform(self, sentence):
        token_list = sentence.tokens
        possib = []
        for token in token_list:
            if token.pos == tree_handler.pres_ind_tag:
                # Assemble the appropriate replacements
                # Verb is in present indicative
                if token.pos == VerbInflCorruptor.pres_ind_tag:
                    pos_inf = ["o", "es", "és", "ás", "e", "emos", "éis", "en"
                               , "as", "a", "amos", "áis", "an", "imos", "ís"]
                    v_match = VerbInflCorruptor.verb_pr_ind_regex.match(token)
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    if infl[0] == "a" or infl[0] == "á":
                        for i in pos_inf:
                            if i[0] != "a" and i[0] != "o" and i[0] != "á":
                                pos_inf.remove(i)
                    else:
                        for inf in pos_inf:
                            if inf[0] == "a" or inf[0] == "á":
                                pos_inf.remove(inf)
                # Verb is in imperfect indicative
                elif token.pos == VerbInflCorruptor.imp_ind_tag:
                    v_match = VerbInflCorruptor.verb_imp_ind_regex.match(token)
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    mid = v_match.group(3)
                    pos_inf = [mid, mid + "s", mid + "mos",
                               mid+"is", mid + "n"]
                # Verb is in past simple
                elif token.pos == VerbInflCorruptor.perf_ind_tag:
                    pos_inf = ["é", "aste", "ó", "amos", "asteis", "aron", "í",
                               "iste", "ió", "imos", "isteis", "ieron"]
                    v_match = VerbInflCorruptor.verb_ps_ind_regex.match(token)
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    if infl[0] == "a" or infl[0] == "á":
                        for i in pos_inf:
                            if i[0] != "a" and i[0] != "é" and i[0] != "á":
                                pos_inf.remove(i)
                    else:
                        for inf in pos_inf:
                            if inf[0] == "a" or inf[0] == "é":
                                pos_inf.remove(inf)

                pos_inf.remove(infl)
                for new_inf in pos_inf:
                    new_verb = root + new_inf
                    if new_verb in verb_list:
                        rep = [new_verb, token.beg, token.end]
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
