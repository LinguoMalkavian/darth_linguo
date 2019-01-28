import re
from collections import defaultdict
import random



class corruptor ():
    # Constants to keep tags organized
    POS_ADJ = "ADJ"
    POS_VERB = "VERB"
    POS_PREP = "Prep"
    DEP_ROOT = "ROOT"
    MOOD_INDICATIVE = "Ind"
    TENSE_PRESENT = "Pres"
    TENSE_IMPERFECT = "Imp"
    TENSE_PAST = "Past"
    FINITE_VERB = "Fin"

    def __init__(self, kind, nlp_module):
        self.kind = kind
        self.nlp_module = nlp_module

    def test_possible(self, sentence):
        pass

    def transform(self, sentence):
        pass

    def is_word(self, word_text):
        """Check if a word form is in the vocabulary.
        """
        word_id = self.nlp_module.vocab.strings(word_text)
        if word_id in self.nlp_module.vocab:
            return True
        return False
# Subject Remover


class PrepRemover(corruptor):

    def test_possible(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if token_features["AdpType"] == self.POS_PREP:
                return True
        return False

    def transform(self, sentence):
        for token in sentence:
            token_features = extract_token_features(token)
            if token_features["AdpType"] == self.POS_PREP:
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
        if token.dep_ == self.DEP_ROOT \
          and features["POS"] == self.POS_VERB \
          and features["VerbForm"] == self.FINITE_VERB:
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
            if features["POS"] == self.POS_VERB and features["VerbForm"] == 'Fin':
                return True
        return False

    def transform(self, sentence):
        # TODO deal with possible correct sentences when 1st and 3rd are
        # interchanged (look at the subject?)
        possib = []
        for token in sentence:
            # Assemble the appropriate replacements
            # Verb is in present indicative
            possible_infl = []
            token_info = extract_token_features(token)
            infl = ""
            if token_info["Mood"] == self.MOOD_INDICATIVE \
                    and token_info["Tense"] == self.TENSE_PRESENT:
                possible_infl = ["o", "es", "és", "ás", "e", "emos", "éis", "en",
                           "as", "a", "amos", "áis", "an", "imos", "ís"]
                v_match = VerbInflCorruptor.verb_pr_ind_regex.match(token.text)
                if v_match:
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    if infl[0] == "a" or infl[0] == "á":
                        for i in possible_infl:
                            if i[0] != "a" and i[0] != "o" and i[0] != "á":
                                possible_infl.remove(i)
                    else:
                        for inf in possible_infl:
                            if inf[0] == "a" or inf[0] == "á":
                                possible_infl.remove(inf)
            # Verb is in imperfect indicative
            elif token_info["Mood"] == self.MOOD_INDICATIVE \
                    and token_info["Tense"] == self.TENSE_IMPERFECT:
                v_match = VerbInflCorruptor.verb_imp_ind_regex.match(
                                                                   token.text)
                if v_match:
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    mid = v_match.group(3)
                    possible_infl = [mid, mid + "s", mid + "mos",
                               mid+"is", mid + "n"]
            # Verb is in past simple
            # elif token.pos == VerbInflCorruptor.perf_ind_tag:
            # modify when there are more kinds of verbs
            else:
                possible_infl = ["é", "aste", "ó", "amos", "asteis", "aron", "í",
                           "iste", "ió", "imos", "isteis", "ieron"]
                v_match = VerbInflCorruptor.verb_ps_ind_regex.match(token.text)
                if v_match:
                    root = v_match.group(1)
                    infl = v_match.group(2)
                    if infl[0] == "a" or infl[0] == "á":
                        for i in possible_infl:
                            if i[0] != "a" and i[0] != "é" and i[0] != "á":
                                possible_infl.remove(i)
                    else:
                        for inf in possible_infl:
                            if inf[0] == "a" or inf[0] == "é":
                                possible_infl.remove(inf)

            if infl in possible_infl:
                possible_infl.remove(infl)
            if v_match:
                for new_inf in possible_infl:
                    new_verb = root + new_inf
                    if self.is_word(new_verb):
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
            if features["POS"] == self.POS_ADJ:
                if AdjInflCorruptor.inf_ADJ_regex.match(token.text):
                    return True
        return False

    def transform(self, sentence):
        # List to store all codified possible transforms
        possib = []
        for token in sentence:
            features = extract_token_features(token)
            if features["POS"] == self.POS_ADJ:
                match = AdjInflCorruptor.inf_ADJ_regex.match(token.text)
                if match:
                    root = match.group(1)
                    infl = match.group(2)
                    possible_infl = ["a", "o", "as", "os", "es"]
                    possible_infl.remove(infl)
                    random.shuffle(possible_infl)
                    for new_inf in possible_infl:
                        new_word = root + new_inf
                        if self.is_word(new_word):
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
