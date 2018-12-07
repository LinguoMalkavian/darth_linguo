from tqdm import tqdm
from collections import defaultdict
import corpus_tools
from numpy.random import choice
import argparse


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


def generateWordSalad(n_frequencies, order, minlength=7, maxlength=45):
    """Generate a single word salad of specified order.

    n_frequencies -- a nested dictionary with:
        first key: strings of order-1 space separated tokens (prefix)
        second key: a string with one token (target)
        values:relative frequency of the continuation given the prefix.
    order -- an integer, the order of ngrams being used"""

    # The sentence is a list of tokens, start it by padding it with #
    n_pads = order-1
    sentence = ["#"]*(n_pads)
    while sentence[-1] != "<eos>":
        prefix = " ".join(sentence[-(order-1):])
        options, freqs = zip(*n_frequencies[prefix].items())
        total = sum(freqs)
        probs = [freq/total for freq in freqs]
        word = choice(options, p=probs)
        sentence.append(word)
        if ((sentence[-1] == "<eos>" and len(sentence) - n_pads < minlength)
           or len(sentence) - n_pads >= maxlength):
            sentence = ["#"]*(n_pads)

    return sentence[n_pads:]


def saveWordSaladCorpus(salads, filenamePrefix,
                        order, suffix=""):
    """Store a corpus in a file"""

    g_label = 0
    ug_type = "WS"
    if suffix != "":
        suffix = "-" + suffix

    filename = filenamePrefix + "-{}-gramWS".format(order) + suffix
    corpus_tools.save_uniform_labeled_corpus(filename,
                                             salads,
                                             g_label,
                                             ug_type)
    print("Word salads of order {} saved to:\n {}".format(order, filename))


def generateMultipleOrders(corpus_fn,
                           outfile_path,
                           orders=[2, 3, 4, 5, 6],
                           numSalads=None,
                           corpType="train"):
    """Load a tokenized corpus, train LMS and  generate ws
     of stated orders as well as a  the mix"""

    # Load file with grammatical data.
    tokenized_sents = corpus_tools.load_tokenized_corpus(corpus_fn)

    if numSalads is None:
        numSalads = len(tokenized_sents)

    allWordSalads = {}
    allWordSalads["mix"] = []
    for order in orders:
        # Get n gram frequencies for chosen n
        n_gramFreqs = extract_ngram_freq(tokenized_sents, order)
        allWordSalads[order] = generateSingleOrder(n_gramFreqs,
                                                   order, numSalads)
        allWordSalads["mix"] += allWordSalads[order][:numSalads//len(orders)]
        saveWordSaladCorpus(allWordSalads[order],
                            outfile_path,
                            order,
                            suffix=corpType)
    # Generate also a mix
    saveWordSaladCorpus(allWordSalads["mix"],
                        outfile_path,
                        "mix",
                        suffix=corpType)

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

    desc = "Generate Word salads based on a training corpus for the LM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('in_file', type=str,
                        help="The name of the file to extract the frequencies")
    parser.add_argument('--named_corpus',
                        help="If file is in the data folder name of corpus")
    parser.add_argument("--outfile", default=None,
                        help="""If specified saves WS with this prefix""")
    parser.add_argument("--numsalads", "-n", type=int,
                        help="N of WS, by def same as sentences in input")
    parser.add_argument("--orders", nargs='*', default=[2, 3, 4, 5, 6],
                        type=int,
                        help="Orders of WS to be generated")
    parser.add_argument("--mix", action='store_true',
                        help="Whether to produced the mixed WS file")
    parser.add_argument("--suf", default="train",
                        help="The suffix for the files (e.g. train, val)")

    args = parser.parse_args()

    if args.named_corpus:
        filename = corpus_tools.getDataFolderPath(args.named_corpus)\
            + args.in_file
    else:
        filename = args.in_file

    if args.outfile is not None:
        outfilePrefix = "/".join(filename.split("/")[:-1]) + "/" + args.outfile
    else:
        outfilePrefix = filename

    generateMultipleOrders(filename,
                           outfilePrefix,
                           orders=args.orders,
                           numSalads=args.numsalads,
                           corpType=args.suf)
