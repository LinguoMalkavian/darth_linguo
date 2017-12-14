# This program gets data ready for processing by LinguoV1
import random
import math
import sys
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from string import capwords


def main():
    # Handles importing data
    # Manual seed for consistency
    random.seed(42)

    # Read in a corpus file
    input_corpus_path = sys.argv[1]
    corpus_name = input_corpus_path.split("/")[-1]
    real_text = read_in_file(input_corpus_path)

    # Separate the data between training and testing
    proportion_train = 0.8
    cutoff = math.floor(len(real_text)*proportion_train)
    random.shuffle(real_text)
    real_train, real_test = real_text[:cutoff], real_text[cutoff:]

    # Process the input sentences (for tokenization, tokenizer sucks otherwise)
    parsed_real_train = [word_tokenize(sentence) for sentence in real_train]
    parsed_real_test = [word_tokenize(sentence) for sentence in real_test]

    # Extract the statististical info needed to generate unigram word salad

    # Calculate average sentence length
    lengths = [len(sent) for sent in parsed_real_train]
    avg_sent_length = np.mean(lengths)
    length_sd = np.std(lengths)
    counts = defaultdict(int)
    total = 0.0
    for sentence in parsed_real_train:
        for token in sentence:
            if token.text != ".":
                counts[token.text.lower()] += 1
                total += 1

    # TODO: implement a version where low frequency words are replaced by pos
    # Switch happaxes for the UNK token
    hapaxes = []
    counts["#unk"] = 0
    for key in counts:
        if counts[key] == 1:
            counts["#unk"] += 1
            hapaxes.append(key)

    for hapax in hapaxes:
        counts.pop(hapax)

    vocabulary = []
    probdist = []
    for key in counts:
        vocabulary.append(key)
        probdist.append(counts[key])

    # Get the sentences represented as lists of words
    tokenized_real_train = token_replacement(parsed_real_train, hapaxes)
    tokenized_real_test = token_replacement(parsed_real_test, hapaxes)

    # get a list of word salads the same length as the real test data
    word_salads_train = [generateWS(vocabulary,
                                    probdist,
                                    avg_sent_length,
                                    length_sd) for _ in range(len(
                                                        tokenized_real_train))]
    word_salads_test = [generateWS(vocabulary,
                                   probdist,
                                   avg_sent_length,
                                   length_sd) for _ in range(len(
                                                       tokenized_real_test))]

    # Consolidate training data
    labeled_sentences_train = [[sentence, 1] for sentence in tokenized_real_train]
    labeled_sentences_train += [[sentence, 0] for sentence in word_salads_train]
    random.shuffle(labeled_sentences_train)

    # Consolidate test data
    labeled_sentences_test = [[sentence, 1] for sentence in tokenized_real_test]
    labeled_sentences_test += [[sentence, 0] for sentence in word_salads_test]
    random.shuffle(labeled_sentences_test)

    # Define the vocabulary and word ids
    vocabulary.append(".")
    word_to_ix = {}
    for word in vocabulary:
        word_to_ix[word] = len(word_to_ix)

    # Saving the Corpus
    training_corpus_fn = "Data/" + corpus_name + ".VSUnigram.labeled.training"
    testing_corpus_fn = "Data/" + corpus_name + ".VSUnigram.labeled.testing"

    save_corpus(labeled_sentences_train, training_corpus_fn)
    save_corpus(labeled_sentences_test, testing_corpus_fn)

    print("You now have {} train instances and {} test instancess:".format(
        len(labeled_sentences_train), len(labeled_sentences_test)))

# END OF THE UNREVISED Code


def token_replacement(tokenized_sentences, hapaxes):
    # Takes a list of sentences that have been tokenized
    # Words specified in hapaxes are replaced by UNK
    # TODO: implement a version that replaces words by their tag instead
    processed = []
    for sentence in tokenized_sentences:
        this_sentence = []
        for token in sentence:
            if token.text.lower() in hapaxes:
                this_sentence.append("#unk")
            else:
                this_sentence.append(token.text)
        processed.append(this_sentence)
    return processed


def save_corpus(data, filename):
    out_file = open(filename, "w")
    for instance in data:
        words = " ".join(instance[0])
        label = str(instance[1])
        out = words + "|" + label + "\n"
        out_file.write(out)
    out_file.close()


# Method that reads in a corpus file and filters out interogatives and
# exclamatives, returns a list of sentence strings
def read_in_file(file_path):
    in_file = open(file_path, "r")
    numlines = 0
    inter_excl = 0
    answer = []
    for line in in_file.readlines():
        # Keep only sentences, those have a period at the end
        # (is support for ? and ! needed??)
        if line.strip() != "":
            if line.strip()[-1] == ".":
                answer.append(line.strip())
            elif line.strip()[-1] == "?" or line.strip()[-1] == "!":
                inter_excl += 1
        numlines += 1
    print('''Full corpus has {} sentences
    {} were dumped
     among which {} interogatives or exclamatives'''.format(
                                len(answer),
                                numlines-len(answer),
                                inter_excl))
    return answer


# Receives the vocabulary and probability distribution and generates a single
# sentence drawn from the unigram distribution
def generateWS(vocab, probdist, avg_length, sd):
    # Method to generate one word salad sentence usin unigram distribution
    # Vocab is a list of vocabulary words
    # probdist contains the probabilities of vocabulary words in same order
    # avg_length is the average length of sentences
    # sd is the standar deviation for the legths of sentences

    # Draw the length
    length = math.floor(random.gauss(avg_length, sd))
    if length < 6:
        length = 6
    # Draw the words
    draw = np.choice(vocab, length, probdist).tolist()
    # Assemble the sentence
    # Capitalize the first word in the sentence
    sentence = [capwords(draw.pop(0))]
    while draw:
        next_word = draw.pop(0)
        # Special case for punctuation that needs to be closed
        if next_word in ["(", "«"]:
            try:
                sentence.append(next_word)
                sentence.append(draw.pop(0))
                closing = ""
                if next_word == "(":
                    closing = ")"
                elif next_word == "«":
                    closing = "»"
                draw.insert(random.randint(0, len(draw)), closing)
            except IndexError:
                break
        elif next_word not in [")", "»"]:
            sentence.append(next_word)
    sentence.append(".")
    return sentence
