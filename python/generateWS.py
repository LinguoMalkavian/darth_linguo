import sys
import os
from tqdm import tqdm
from collections import defaultdict
import corpus_tools
from numpy.random import choice


def get_vocabulary(sentences,
                   hap_threshold=1):
    counts = defaultdict(int)
    total_tokens = 0
    print("Establishing vocabulary")
    for sentence in sentences:
        for token in sentence:
            counts[token.lower()] += 1
            total_tokens += 1
    hapaxes = []
    counts["<unk>"] = 0
    # Identify hapaxes, count them for smoothing
    for key in counts:
        if counts[key] <= hap_threshold:
            counts["<unk>"] += counts[key]
            hapaxes.append(key)
    # Remove them from the count
    for hapax in hapaxes:
        counts.pop(hapax)
    # Consolidate vocabulary and word ids
    vocabulary = []
    unidist = []
    for key in counts:
        vocabulary.append(key)
        unidist.append(counts[key])
    # Define word ids.
    word2id = {}
    for word in vocabulary:
        word2id[word] = len(word2id)
    # Print Report.
    report = '''The corpus has a total of {} tokens, with
    {} kept types and {} hapaxes'''.format(total_tokens, len(word2id),
                                           len(hapaxes))
    print(report)
    return word2id


def extract_ngram_freq(corpus, order):
    """Get the n-gram frequencies of the corpus"""
    n_frequencies = defaultdict(lambda: defaultdict(int))
    for original in corpus:
        # Make a copy so original remains unchanged
        sentence = original[:]
        for _ in range(order-1):
            sentence.insert(0, "#")
        for ini in range(len(sentence) - order + 1):
            prefix = " ".join(sentence[ini:ini+order-1])
            target = sentence[ini+order-1]
            n_frequencies[prefix][target] += 1
    return n_frequencies


def generateWordSalad(n_frequencies, order, minlength=7):
    """Generate a single word salad of specified order.

    n_frequencies -- a nested dictionary with:
        first key: strings of order-1 space separated tokens (prefix)
        second key: a string with one token (target)
        values:relative frequency of the continuation given the prefix.
    order -- an integer, the order of ngrams being used"""

    # The sentence is a list of tokens, start it by padding it with #
    sentence = ["#"]*(order-1)
    while sentence[-1] != "<eos>":
        prefix = " ".join(sentence[-(order-1):])
        options, freqs = zip(*n_frequencies[prefix].items())
        total = sum(freqs)
        probs = [freq/total for freq in freqs]
        word = choice(options, p=probs)
        if len(sentence) < minlength and word in [".", "<eos>"]:
            # Only allow the sentence to end if minimum has been reached
            if len(options) == 1:
                # There are no valid continuations, scrape the sentence
                sentence = ["#"]*(order-1)
                continue
            else:
                while word in [".", "<eos>"]:
                    word = choice(options, p=probs)
        sentence.append(word)
    return sentence[order-1:]


def saveWordSaladCorpus(salads, filenamePrefix, tag):
    """Store a corpus in a file"""

    filename = filenamePrefix + "-{}-gramWS".format(tag)
    corpus_tools.save_tokenized_corpus(filename, salads)
    print("Word salads of order {} saved to:\n {}".format(tag, filename))


def generateMultipleOrders(corpusName, orders=[2, 3, 4, 5, 6]):
    """Run whole pipeline to generate ws of several orders and the mix"""

    basePath = "/".join(os.getcwd().split("/")[:-1])
    dataPath = basePath + "/Data/" + corpusName + "/" + corpusName
    corpus_fn = dataPath + "-pretrain"

    # Load file with grammatical data.
    tokenized_sents = corpus_tools.load_raw_grammatical_corpus(corpus_fn)

    # Extract and save vocabulary.
    word2id = get_vocabulary(tokenized_sents, 2)
    corpus_tools.saveWord2Id(word2id, dataPath)

    preprocessed_sentences = corpus_tools.token_replacement(tokenized_sents,
                                                            word2id)
    numSalads = len(preprocessed_sentences)
    allWordSalads = {}
    allWordSalads["mix"] = []
    for order in orders:
        # Get n gram frequencies for chosen n
        n_gramFreqs = extract_ngram_freq(preprocessed_sentences, order)
        allWordSalads[order] = generateSingleOrder(n_gramFreqs,
                                                   order, numSalads)
        allWordSalads["mix"] += allWordSalads[order][:numSalads//len(orders)]
        saveWordSaladCorpus(allWordSalads[order], dataPath, order)
    # Generate also a mix
    saveWordSaladCorpus(allWordSalads["mix"], dataPath, "mix")

    return allWordSalads


def generateSingleOrder(frequencies, order, quantity):
    """Generate a number of word salads of a given order

    Prints a progress bar"""

    print("Generating word salads of order {}".format(order))
    salads = []
    for _ in tqdm(range(quantity)):
        salad = generateWordSalad(frequencies, order)
        salads.append(salad)

    return salads


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a corpus name")
    else:
        corpusName = sys.argv[1]
        all_WS = generateMultipleOrders(corpusName)
