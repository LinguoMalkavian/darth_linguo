import csv
from nltk import word_tokenize
import random
import datetime
import math
import os


def getDataPath(corpusName):
    """Gives the full path prefix for data includes begining of fileName)"""
    dataPath = getDataFolderPath(corpusName) + corpusName
    return dataPath


def getModelPrefix(corpusName):
    """Gives the path to save models including the begining of file name."""
    modelFolder = getModelDir(corpusName)
    modelPrefix = modelFolder + "/" + corpusName
    return modelPrefix


def getModelDir(corpusName):
    """Gives the path to the Models directory"""
    basePath = "/".join(os.getcwd().split("/")[:-1])
    modelFolder = basePath + "/Models/" + corpusName
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)
    return modelFolder


def getDataFolderPath(corpusName):
    """Gives the path of the data folder"""
    basePath = "/".join(os.getcwd().split("/")[:-1])
    dataPath = basePath + "/Data/" + corpusName + "/"
    return dataPath


def load_raw_grammatical_corpus(input_corpus_filename,
                                minlength=7,
                                maxlength=45):
    """Load a corpus of line separated sentences.

    input_corpus_filename -- the path to the file with the sentences
    minlength -- the minimum length of sentences to be kept (in tokens)"""
    input_corpus_path = input_corpus_filename
    in_file = open(input_corpus_path, "r")
    numlines = 0
    inter_excl = 0
    tooShortCount = 0
    tooLongCount = 0
    tokenized_sentences = []
    for line in in_file.readlines():
        # Keep only sentences, those have a period at the end
        if line.strip() != "":
            if line.strip()[-1] == ".":
                tokenized = word_tokenize(line)
                tokenized.append("<eos>")
                if len(tokenized) >= minlength and len(tokenized) <= maxlength:
                    tokenized_sentences.append(tokenized)
                else:
                    if len(tokenized) < minlength:
                        tooShortCount += 1
                    else:
                        tooLongCount += 1
            elif line.strip()[-1] == "?" or line.strip()[-1] == "!":
                inter_excl += 1
        numlines += 1
    n_sent = len(tokenized_sentences)
    print('''Full corpus has {full} sentences,
    \t {dumped} were dumped,
    among which {inter} interogatives or exclamatives.
    {tooL} sentences had more than {max} tokens
    {tooS} had fewer than {min}'''.format(full=n_sent,
                                          dumped=numlines-n_sent,
                                          inter=inter_excl,
                                          tooL=tooLongCount,
                                          tooS=tooShortCount,
                                          max=maxlength,
                                          min=minlength))
    random.shuffle(tokenized_sentences)
    return tokenized_sentences


def saveErrors(falseNegatives, falsePositives, corpus_name, noiseName):
    """Saves false negatives and false positives to their own files"""

    dataFolder = getDataFolderPath(corpus_name)
    date = datetime.datetime.now()
    folderName = dataFolder + "errors/"
    if not os.path.isdir(folderName):
        os.mkdir(folderName)
    baseName = (folderName + corpus_name +
                "-VS" + noiseName + "gram-" + str(date))
    falseNeg_fn = baseName + "-false_negatives"
    falsePos_fn = baseName + "-false_positives"
    save_tokenized_corpus(falseNeg_fn, falseNegatives)
    save_tokenized_corpus(falsePos_fn, falsePositives)


def save_tokenized_corpus(filename, sentences):
    """Saves an already tokenized corpus to file"""
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
            word2Id[row[0]] = int(row[1])
    return word2Id


def labelAndShuffleItems(positiveItems, negativeItems):
    """Produces a single list of test instances with labels"""

    labeledPositive = [(sentence, 1) for sentence in positiveItems]
    labeledNegative = [(sentence, 0) for sentence in negativeItems]
    labeled = labeledPositive + labeledNegative
    random.shuffle(labeled)
    return labeled


def getLabeledData(pathGram, pathNoise):
    """Provides a corpus of labeled instances

    Takes paths for the positive and negative data,
    loads the files, labels the instances and returns
    them as a list of pairs (sentence,label)"""

    gram = load_tokenized_corpus(pathGram)
    noise = load_tokenized_corpus(pathNoise)
    labeled = labelAndShuffleItems(gram, noise)
    return labeled


def saveResults(resultStr, corpusPath):
    """Appends results to a single result file per corpus"""

    results_fn = corpusPath+"-results"
    with open(results_fn, "a+") as resultsFile:
        resultsFile.write(resultStr)


def makeResultsString(results):
    """Produces string version of results dictionary"""

    response = ""
    date = datetime.datetime.now()
    header = "Experiment: grammatical V.S {} noise\n".format(
                                                results["noise-type"])
    response += header
    timestr = "\tcarried out on {}:\n".format(str(date))
    response += timestr
    for key in results:
        if key != "noise-type":
            response += "{}:{}\n".format(key, results[key])
    return response


def seconds_to_hms(secondsin):
    hours = math.floor(secondsin/3600)
    remain = secondsin % 3600
    minutes = math.floor(remain/60)
    seconds = math.floor(remain % 60)
    answer = "{h} hours, {m} minutes and {s} seconds".format(h=hours,
                                                             m=minutes,
                                                             s=seconds)
    return answer


def splitCorpus(corpus, splits):
    """Receives a list of names of splits with the proportion of data,
    the splits must add to 1 (or less) and be all positive."""

    total = sum(splits.values())
    if total > 1:
        raise ValueError("The proportions of the pieces of the corpus must sum to 1 or less")

    response = {}
    sliceStart = 0
    sliceEnd = 0
    for sub_corpus_name in splits:
        prop = splits[sub_corpus_name]
        if prop <= 0 or prop >= 1:
            # if a proportion is invalid raise an exception
            error_message = """Each proportion must be a positive float less than 1,
            proportion for {} is {} """.format(sub_corpus_name, prop)
            raise ValueError(error_message)
        sliceStart = sliceEnd
        sliceEnd += math.floor(len(corpus)*prop)
        response[sub_corpus_name] = corpus[sliceStart:sliceEnd]

    return response


def saveCorpora(baseFilename, corpora):
    for corpus_name in corpora:
        filePath = baseFilename + "-" + corpus_name
        with open(filePath, "w") as outfile:
            for sentence in corpora[corpus_name]:
                line = " ".join(sentence)+"\n"
                outfile.write(line)
        print("{} sentences of {} saved to {}".format(len(corpora[corpus_name]),
                                                      corpus_name,
                                                      filePath))
