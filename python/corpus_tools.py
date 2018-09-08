import csv
from nltk import word_tokenize
import random
import os


def getDataPath(corpusName):
    basePath = "/".join(os.getcwd().split("/")[:-1])
    dataPath = basePath + "/Data/" + corpusName + "/" + corpusName
    return dataPath


def load_raw_grammatical_corpus(input_corpus_filename, minlength=7):
    """Load a corpus of line separated sentences.

    input_corpus_filename -- the path to the file with the sentences
    minlength -- the minimum length of sentences to be kept (in tokens)"""
    input_corpus_path = input_corpus_filename
    in_file = open(input_corpus_path, "r")
    numlines = 0
    inter_excl = 0
    tokenized_sentences = []
    for line in in_file.readlines():
        # Keep only sentences, those have a period at the end
        if line.strip() != "":
            if line.strip()[-1] == ".":
                tokenized = word_tokenize(line)
                tokenized.append("<eos>")
                if len(tokenized) >= minlength:
                    tokenized_sentences.append(tokenized)
            elif line.strip()[-1] == "?" or line.strip()[-1] == "!":
                inter_excl += 1
        numlines += 1
    n_sent = len(tokenized_sentences)
    print('''Full corpus has {} sentences,
    \t {} were dumped,
    among which {} interogatives or exclamatives.'''.format(n_sent,
                                                            numlines-n_sent,
                                                            inter_excl))
    random.shuffle(tokenized_sentences)
    return tokenized_sentences


def save_tokenized_corpus(filename, sentences):
    with open(filename, "w") as outfile:
        for tokenList in sentences:
            sentString = " ".join(tokenList)+"\n"
            outfile.write(sentString)


def load_tokenized_corpus(filename):
    """Load a file with sentences that are already tokenized

    Tokens are separated by spaces, sentences by newlines"""

    sentences = []
    with open(filename, "r") as inFile:
        for line in inFile.readlines():
            sentences.append(line.strip().split(" "))
    return sentences


def token_replacement(sentences, word2id):
    """Replace out of vocabulary items

    Takes a list of sentences, each of which is a list of tokens (str)
    Words not in word2id are replaced by <unk>, everything is lower Cased."""

    print("Cleaning corpus")
    cleaned = []
    for sentence in sentences:
        this_sentence = []
        for token in sentence:
            if token.lower() in word2id:
                this_sentence.append(token.lower())
            else:
                this_sentence.append("<unk>")
        cleaned.append(this_sentence)
    return cleaned


def saveWord2Id(word2id, filenamePrefix):
    """Save the word id dictionary to file"""

    filename = filenamePrefix+"-word2id"
    with open(filename, "w") as outFile:
        writer = csv.writer(outFile)
        writer.writerows(word2id.items())
    print("Data saved to " + filename)


def loadWord2Id(corpusPath):
    """Load a word id dictionary"""
    filename = corpusPath+"-word2id"
    word2Id = {}
    with open(filename, "r") as inFile:
        reader = csv.reader(inFile)
        for row in reader:
            word2Id[row[0]] = row[1]
    return word2Id
