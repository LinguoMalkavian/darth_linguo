import re
from collections import defaultdict
import random


class corruptor ():
    # Constants to keep tags organized
    POS_ADJ = "ADJ"
    POS_VERB = "VERB"
    POS_NOUN = "NOUN"
    POS_PREP = "Prep"
    DEP_MOD = "amod"
    DEP_ROOT = "ROOT"
    DEP_SUBJ = "nsubj"
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
        word_id = self.nlp_module.vocab.strings[word_text]
        if word_id in self.nlp_module.vocab:
            return True
        return False


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
        for index, token in enumerate(sentence):
            if self.is_main_conjugated_verb(token):
                return index
        return None
    # This method takes the sentence, removes the main verb and returns it
    # Input sentence must be a spacy sentence with dependency parsing

    def transform(self, sentence, index):
        token = sentence[index]
        main_verb = token.text
        main_verb_reg = re.compile("( ?)"+main_verb)
        newtext = main_verb_reg.sub("", sentence.text).strip()
        newtext = newtext[0].capitalize() + newtext[1:]
        return newtext

    def is_main_conjugated_verb(self, token):
        features = extract_token_features(token)
        if (token.dep_ == self.DEP_ROOT
                and features["POS"] == self.POS_VERB
                and features["VerbForm"] == self.FINITE_VERB):
            return True
        return False


# Verbal inflection corruptio n
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

    def __init__(self, kind, nlp_module):
        super().__init__(kind, nlp_module)
        self.SUPORTED_TENSES = ["-".join([self.MOOD_INDICATIVE,
                                          self.TENSE_PAST]),
                                "-".join([self.MOOD_INDICATIVE,
                                         self.TENSE_PRESENT]),
                                "-".join([self.MOOD_INDICATIVE,
                                         self.TENSE_IMPERFECT])]

    def test_possible(self, sentence):
        indices = []
        for index, token in enumerate(sentence):
            if (self.is_final_verb_w_subj(token)):
                indices.append(index)
        return indices

    def is_final_verb_w_subj(self, token):
        features = extract_token_features(token)
        tense = "-".join([features["Mood"], features["Tense"]])
        if (features["POS"] == self.POS_VERB
                and features["VerbForm"] == self.FINITE_VERB
                and tense in self.SUPORTED_TENSES):
            # Verb is the right kind and the tense is supported
            # Check whether it has a subject by looking at it's children
            for child in token.children:
                if child.dep_ == self.DEP_SUBJ:
                    return True
        return False

    def transform(self, sentence, target_ind=None):
        # TODO deal with possible correct sentences when 1st and 3rd are
        # interchanged (look at the subject?)
        if target_ind is None:
            target_ind = self.test_possible(sentence)
        possib = []
        for index in target_ind:
            # Assemble the appropriate replacements
            token = sentence[index]
            possible_infl = []
            token_info = extract_token_features(token)
            infl = ""
            # Find the tense
            if (token_info["Mood"] == self.MOOD_INDICATIVE
                    and token_info["Tense"] == self.TENSE_PRESENT):
                # Verb is in present indicative
                possible_infl = ["o", "es", "és", "ás", "e", "emos", "éis",
                                 "en", "as", "a", "amos", "áis", "an", "imos",
                                 "ís"]
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
            # modify when there are more kinds of verbs
            else:
                possible_infl = ["é", "aste", "ó", "amos", "asteis", "aron",
                                 "í", "iste", "ió", "imos", "isteis", "ieron"]
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
            return None


# Adjectival inflection corruption
class AdjInflCorruptor(corruptor):

    inf_ADJ_regex = re.compile("([A-Za-záéíóúñÑÁÉÍÓÚ]+)(a|o|as|os|es)$")

    def test_possible(self, sentence):
        targets = []
        for index, token in enumerate(sentence):
            if self.is_modifying_adj(token):
                targets.append(index)
        return targets

    def transform(self, sentence, targets):
        # List to store all codified possible transformations
        possib = []
        for index in targets:
            token = sentence[index]
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
            return None

    def is_modifying_adj(self, token):
        features = extract_token_features(token)
        if (features["POS"] == self.POS_ADJ
                and token.head.pos_ == self.POS_NOUN
                and token.dep_ == self.DEP_MOD
                and AdjInflCorruptor.inf_ADJ_regex.match(token.text)):
                return True


def extract_token_features(token):
    feat_dict = defaultdict(str)
    feat_dict["POS"] = token.pos_
    try:
        features_string = token.tag_.split("__")[1]
    except IndexError:
        return feat_dict
    pair_strs = features_string.split("|")
    for pair_str in pair_strs:
        pair = pair_str.split("=")
        if len(pair) == 2:
            feat_dict[pair[0]] = pair[1]
    return feat_dict


def select_corruption(possibilities, counts):
    if possibilities:
        min_value = float("inf")
        arg_min = None
        for cor_type in possibilities:
            if counts[cor_type] < min_value:
                min_value = counts[cor_type]
                arg_min = cor_type
        selection = possibilities.pop(arg_min)
        return (arg_min, selection)
    else:
        return None
