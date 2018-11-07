from spacy import symbols
import re
from collections import defaultdict
import random

# Load the dictionary
lemma_list = []
lemma_file = open("lemario_snowball.txt", "r")
for line in lemma_file.readlines():
    lemma_list.append(line.strip())
# Load the inflected verb form list
verb_list = []
verb_file = open("verbos-espanol-conjugaciones.txt")
for line in verb_file.readlines():
    verb_list.append(line.strip())

# Constants to keep tags organized
pos_adj = "ADJ"
pos_verb = "VERB"
pos_prep = "Prep"
dep_root = "ROOT"
mood_indicative = "Ind"
tense_present = "Pres"
tense_imperfect = "Imp"
tense_past = "Past"
finite_verb = "Fin"


class corruptor ():
    def __init__(self, kind):
        self.kind = kind
        pass

    def test_possible(self, sentence):
        pass

    def transform(self, sentence):
        pass

# Subject Remover


class PrepRemover(corruptor):
    def test_possible(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if token_features["AdpType"] == pos_prep:
                return True
        return False

    def transform(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if token_features["AdpType"] == pos_prep:
                prepo = token.text
                prepo_reg = re.compile("( ?)"+prepo)
                newtext = prepo_reg.sub("", sentence.text)
                newtext = newtext[0].capitalize() + newtext[1:]
                return newtext
        return -1


# Verb Remover
# Removes the main verb in a sentence
class VerbRemover(corruptor):
    # This transformation will only be attempted in sentences with verbal roots
    def test_possible(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if self.is_main_conjugated_verb(token, token_features):
                return True
        return False
    # This method takes the sentence, removes the main verb and returns it
    # Input sentence must be a spacy sentence with dependency parsing

    def transform(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if self.is_main_conjugated_verb(token, token_features):
                main_verb = token.text
                main_verb_reg = re.compile("( ?)"+main_verb)
                newtext = main_verb_reg.sub("", sentence.text)
                newtext = newtext[0].capitalize() + newtext[1:]
                return newtext
        return -1

    def is_main_conjugated_verb(self, token, features):
        if token.dep_ == dep_root \
          and features["POS"] == pos_verb \
          and features["VerbForm"] == finite_verb:
            return True
        return False


# Verbal inflection corruption
class VerbInflCorruptor(corruptor):
    # Version with all main verbs
    # main_verb_tag = re.compile("vm[is][cfips]000")
    # Version with only the three most common tenses

    root_str = "([A-Za-záéíóúñÑÁÉÍÓÚ]+)"

    # Morphological model of the present indicative

    pres_infl = "(o|es|és|ás|e|emos|éis|en|as|a|amos|áis|an|imos|ís)$"
    verb_pr_ind_regex = re.compile(root_str + pres_infl)

    # Morphological model of regular past imprefect of the indicative

    imperf_infl = "((aba|ía)(|s|mos|is|n))$"
    verb_imp_ind_regex = re.compile(root_str + imperf_infl)

    # Morphological model of regular simple past of the indicative

    perf_infl = "(é|aste|ó|amos|asteis|aron|í|iste|ió|imos|isteis|ieron)$"
    verb_ps_ind_regex = re.compile(root_str + perf_infl)

    def test_possible(self, sentence):
        for token in sentence:
            features = extract_token_features(token)
            if features["POS"] == pos_verb and features["VerbForm"] == 'Fin':
                return True
        return False

    def transform(self, sentence):
        # TODO deal with possible correct sentences when 1st and 3rd are
        # interchanged (look at the subject?)
        possib = []
        for token in sentence:
            # Assemble the appropriate replacements
            # Verb is in present indicative
            pos_inf = []
            token_info = extract_token_features(token)
            infl = ""
            if token_info["Mood"] == mood_indicative \
                    and token_info["Tense"] == tense_present:
                pos_inf = ["o", "es", "és", "ás", "e", "emos", "éis", "en",
                           "as", "a", "amos", "áis", "an", "imos", "ís"]
                v_match = VerbInflCorruptor.verb_pr_ind_regex.match(token.text)
                if v_match:
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
            elif token_info["Mood"] == mood_indicative\
                    and token_info["Tense"] == tense_imperfect:
                v_match = VerbInflCorruptor.verb_imp_ind_regex.match(
                                                                   token.text)
                if v_match:
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    mid = v_match.group(3)
                    pos_inf = [mid, mid + "s", mid + "mos",
                               mid+"is", mid + "n"]
            # Verb is in past simple
            # elif token.pos == VerbInflCorruptor.perf_ind_tag:
            # modify when there are more kinds of verbs
            else:
                pos_inf = ["é", "aste", "ó", "amos", "asteis", "aron", "í",
                           "iste", "ió", "imos", "isteis", "ieron"]
                v_match = VerbInflCorruptor.verb_ps_ind_regex.match(token.text)
                if v_match:
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

            if infl in pos_inf:
                pos_inf.remove(infl)
            if v_match:
                for new_inf in pos_inf:
                    new_verb = root + new_inf
                    if new_verb in verb_list:
                        rep = [new_verb, token.text]
                        possib.append(rep)
        if len(possib) != 0:
            choice = random.choice(possib)
            newText = sentence.text.replace(choice[1], choice[0])
            return newText
        else:
            return -1


# Adjectival inflection corruption
class AdjInflCorruptor(corruptor):

    inf_ADJ_regex = re.compile("([A-Za-záéíóúñÑÁÉÍÓÚ]+)(a|o|as|os|es)$")

    def test_possible(self, sentence):
        for token in sentence:
            features = extract_token_features(token)
            if features["POS"] == pos_adj:
                if AdjInflCorruptor.inf_ADJ_regex.match(token.text):
                    return True
        return False

    def transform(self, sentence):
        # List to store all codified possible transforms
        possib = []
        for token in sentence:
            features = extract_token_features(token)
            if features["POS"] == pos_adj:
                match = AdjInflCorruptor.inf_ADJ_regex.match(token.text)
                if match:
                    root = match.group(1)
                    infl = match.group(2)
                    pos_inf = ["a", "o", "as", "os", "es"]
                    pos_inf.remove(infl)
                    random.shuffle(pos_inf)
                    for new_inf in pos_inf:
                        new_word = root + new_inf
                        if new_word in lemma_list:
                            rep = [new_word, token.text]
                            possib.append(rep)
        if len(possib) != 0:
            choice = random.choice(possib)
            newText = sentence.text.replace(choice[1], choice[0])
            return newText
        else:
            return -1


def extract_token_features(token):
    feat_dict = defaultdict(str)
    feat_dict["POS"] = token.pos_
    try:
        pos_string = token.tag_.split("__")[1]
    except:
        return feat_dict
    pair_strs = pos_string.split("|")
    for pair_str in pair_strs:
        pair = pair_str.split("=")
        if len(pair) == 2:
            feat_dict[pair[0]] = pair[1]
    return feat_dict
