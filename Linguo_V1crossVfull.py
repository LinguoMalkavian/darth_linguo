
# Standard pytorch imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# NLTK modules
from nltk.tokenize.moses import MosesTokenizer

# other utilities
import numpy as np
from numpy.random import choice
from collections import defaultdict
import math
from string import capwords
from time import time


# In[2]:


# Methods for importing, preprocessing and generating data


def load_grammatical_corpus(input_corpus_filename):
    input_corpus_path = "Data/"+input_corpus_filename
    in_file = open(input_corpus_path, "r")
    real_text = []
    numlines = 0
    inter_excl = 0
    for line in in_file.readlines():
        # Keep only sentences, those have a period at the end
        if line.strip() != "":
            if line.strip()[-1] == ".":
                real_text.append(line.strip())
            elif line.strip()[-1] == "?" or line.strip()[-1] == "!":
                inter_excl += 1
        numlines += 1

    n_sent = len(real_text)
    print('''Full corpus has {} sentences,
    \t {} were dumped,
    among which {} interogatives or exclamatives'''.format(n_sent,
                                                           numlines-n_sent,
                                                           inter_excl))

    random.shuffle(real_text)
    # Process the input sentences (for tokenization, tokenizer sucks otherwise)
    tokenizer = MosesTokenizer()
    tokenized = [tokenizer.tokenize(sentence) for sentence in real_text]
    return tokenized


# Method to extract:
# The word2idx, a dictionary from vocabulary words to unique integers
# The hapaxes, a list with words with count less than the threshold
# Vocabulary and probdist are also generated for the unigram case
def get_vocabulary(sentences, hap_threshold):
    counts = defaultdict(int)
    total = 0.0
    for sentence in sentences:
        for token in sentence:
            if token != ".":
                counts[token.lower()] += 1
                total += 1
    hapaxes = []
    counts["#unk"] = 0
    # Identify hapaxes, count them for smoothing
    for key in counts:
        if counts[key] <= hap_threshold:
            counts["#unk"] += 1
            hapaxes.append(key)
    # Remove them from the count
    for hapax in hapaxes:
        counts.pop(hapax)
    # Consolidate vocabulary and word ids
    vocabulary = []
    probdist = []
    for key in counts:
        vocabulary.append(key)
        probdist.append(counts[key])

    # Define the vocabulary and word ids
    vocabulary.append(".")
    word_to_ix = {}
    for word in vocabulary:
        word_to_ix[word] = len(word_to_ix)
    return word_to_ix, hapaxes, vocabulary, probdist, counts


# Method to extract n-gram frequencies from the corpus
# as well as length statistics
# Corpus is a list of sentences, each sentence represented by a list of tokens
def extract_ngram_freq(corpus, order):
    n_frequencies = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        for _ in range(order-1):
            sentence.insert(0, "#")
        for ini in range(len(sentence) - order):
            prefix = " ".join(sentence[ini:ini+order-1])
            target = sentence[ini+order-1]
            n_frequencies[prefix][target] += 1
    return n_frequencies


# Method to replace hapaxes by the unk token in the corpus
def token_replacement(sentences, hapaxes):
    # Takes a list of tokenized sentences
    # Returns a list of sentences, each of which is a list of words (str)
    # Words specified in hapaxes are replaced by UNK
    cleaned = []
    for sentence in sentences:
        this_sentence = []
        for token in sentence:
            if token.lower() in hapaxes:
                this_sentence.append("#unk")
            else:
                this_sentence.append(token)
        cleaned.append(this_sentence)
    return cleaned


def generateWSuni(vocab, probdist, avg_length, sd):
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
    draw = choice(vocab, length, probdist).tolist()
    # Assemble the sentence
    sentence = [capwords(draw.pop(0))]
    while draw:
        next_word = draw.pop(0)
        # special case for punctuation that needs to be closed
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


def generateWSNgram(n_frequencies, avg_length, sd, order, unicounts):
    # Method to generate one word salad sentence usin unigram distribution
    # Vocab is a list of vocabulary words
    # probdist contains the probabilities of vocabulary words in same order
    # avg_length is the average length of sentences
    # sd is the standar deviation for the legths of sentences

    # Draw the length
    length = math.floor(random.gauss(avg_length, sd))
    if length < 5:
        length = 5

    sentence = ["#"]*(order-1)
    for i in range(length+order-1):
        prefix = " ".join(sentence[-(order-1):])
        try:
            vocab, freqs = zip(*n_frequencies[prefix].items())
            word = choice(vocab, 1, freqs)[0]
            sentence.append(word)
        except:
            vocab, freqs = zip(*unicounts.items())
            word = choice(vocab, 1, freqs)[0]
            sentence.append(word)
    sentence.append(".")
    return sentence


# Now we define the Neural network


class Linguo(nn.Module):
    def __init__(self, embedding_dim, vocab_size, lstm_dim, hidden_dim):
        super(Linguo, self).__init__()
        # Store the hidden layer dimension
        self.hidden_dim = hidden_dim
        # Define word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Define LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # Define hidden linear layer
        self.hidden2dec = nn.Linear(hidden_dim, 2)
        # Define the hidden state
        self.hstate = self.init_hstate()

    def forward(self, inputsentence):
        self.hstate = self.init_hstate()
        embeds = self.word_embeddings(inputsentence)
        lstm_out, self.hstate = self.lstm(embeds.view(len(inputsentence),
                                                      1,
                                                      -1), self.hstate)
        decision_lin = self.hidden2dec(lstm_out[-1])
        # print(decision_lin)
        decision_fin = F.log_softmax(decision_lin)
        return decision_fin

    def init_hstate(self):
        var1 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        var2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hidden_state = (var1, var2)
        return hidden_state


def prepare_input(word_to_ix, sentence):
    idxs = []
    for word in sentence:
        if word in word_to_ix:
            idxs.append(word_to_ix[word.lower()])
        else:
            idxs.append(word_to_ix["#unk"])
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Training time! Cue Eye of the Tiger
def train_model(train_data,
                embed_dim,
                lstm_dim,
                hidden_dim,
                word_to_ix,
                epochs,
                learning_rate):
    voc_size = len(word_to_ix)
    # Initialize model
    linguo = Linguo(embed_dim, voc_size, lstm_dim, hidden_dim)
    optimizer = optim.SGD(linguo.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()

    for i in range(epochs):
        epoch_loss = 0
        random.shuffle(train_data)
        for data, label in train_data:
            # Restart gradient
            linguo.zero_grad()
            # Run model
            in_sentence = prepare_input(word_to_ix, data)
            target = autograd.Variable(torch.LongTensor([label]))
            prediction = linguo(in_sentence)
            # Calculate loss and backpropagate

            # Squared Loss
            # loss = torch.pow(target-prediction.view(1),2)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            # for parameter in linguo.parameters():
            #   parameter.data.sub_(parameter.grad.data*learning_rate)
            epoch_loss += loss.data[0]
        print("\t Epoch{}:{}".format(i, epoch_loss))
    return linguo


# Testing, testing
def test_model(test_data, model, word2id):
    correct = 0.0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for testcase in test_data:
        target = testcase[1]
        prepared_inputs = prepare_input(word2id, testcase[0])
        prediction_vec = model(prepared_inputs).view(2)
        if prediction_vec.data[0] > prediction_vec.data[1]:
            prediction = 0
        else:
            prediction = 1
        if prediction == testcase[1]:
            correct += 1
            if target == 1:
                tp += 1
            else:
                tn += 1
        else:
            if target == 1:
                fn += 1
            else:
                fp += 1

    # Compile results
    accuracy = correct/len(test_data)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    fmeasure = 2*tp / (2*tp+fp+fn)
    results = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "fmeasure": fmeasure,
               "tp": tp,
               "tn": tn,
               "fp": fp,
               "fn": fn}
    return results


def average_results(result_list):
    total = len(result_list)
    averaged = defaultdict(float)
    for report in result_list:
        for item in report:
            averaged[item] += report[item]
    for item in averaged:
        averaged[item] = averaged[item]/total
    return averaged


# This cell runs the full experiment with cross validation

def main():
    # Modify parameters here
    corpus_name = "euro.mini"
    max_ngram = 6
    hap_thresh = 1
    folds = 10
    train_proportion = 0.8
    embed_dim = 32
    lstm_dim = 32
    hidden_dim = 32
    epochs = 5
    learning_rate = 0.1

    # Load and preprocess the grammatical part of the corpus
    t1 = time()

    corpus = load_grammatical_corpus(corpus_name)
    word2id, hapaxes, vocab, probdist, ucounts = get_vocabulary(corpus,
                                                                hap_thresh)
    prepro_gram = token_replacement(corpus, hapaxes)
    message = "Your corpus has {sent} grammatical sentences".format(
                                                        sent=len(prepro_gram))
    print(message)
    message = "Grammatical corpus loaded in {:.3f} seconds".format(time()-t1)

    # Get sentence length statistics
    lengths = [len(sent) for sent in prepro_gram]
    avg_sent_length = np.mean(lengths)
    length_sd = np.std(lengths)

    full_results = []
    # Run for each n, with x-fold cross validation
    for n in range(1, max_ngram+1):
        t2 = time()
        # Generate the word salads
        message = "Generating word salads of order {}...".format(n)
        t1 = time()
        nsal = len(prepro_gram)
        if n == 1:
            word_salads = [generateWSuni(vocab,
                                         probdist,
                                         avg_sent_length,
                                         length_sd)
                           for _ in range(nsal)]
        else:
            n_freqs = extract_ngram_freq(prepro_gram, n)
            word_salads = [generateWSNgram(n_freqs,
                                           avg_sent_length,
                                           length_sd,
                                           n,
                                           ucounts)
                           for _ in range(nsal)]

        labeled_g = [[sentence, 1] for sentence in prepro_gram]
        labeled_ws = [[sentence, 0] for sentence in word_salads]

        message = "Word salad data has been generated for order {}".format(n)
        print(message)
        te = time() - t1
        message = "\t{} word salads generated in {:.3f} seconds".format(nsal,
                                                                        te)
        print(message)
        message = "Starting experiment on {}-grams ...".format(n)

        result_list = []
        # Iterate over the number of folds
        for fold in range(folds):
            t1 = time()
            message = "Starting training on fold {} for {}-grams...".format(
                                                                    fold+1, n)
            print(message)
            # Shuffle and split data
            random.shuffle(labeled_g)
            random.shuffle(labeled_ws)
            cutoff = math.floor(train_proportion * len(labeled_g))
            train_g, test_g = labeled_g[:cutoff], labeled_g[cutoff:]
            train_ws, test_ws = labeled_ws[:cutoff], labeled_ws[cutoff:]

            train_data = train_g + train_ws
            random.shuffle(train_data)

            test_data = test_g + test_ws
            random.shuffle(test_data)

            # Train the Model
            model = train_model(train_data,
                                embed_dim,
                                lstm_dim,
                                hidden_dim,
                                word2id,
                                epochs,
                                learning_rate)
            te = time()-t1
            message = '''Training finished in {:.4f} seconds.
                Starting testing...'''.format(te)
            print(message)
            print("...")
            t1 = time()
            # Test the Model
            fold_results = test_model(test_data, model, word2id)
            result_list.append(fold_results)
            te = time()-t1
            message = "Testing finished in {} seconds".format(te)
            print(message)
            message = "\Accuracy is {}".format(fold_results['accuracy'])
            print(message)

        order_results = average_results(result_list)
        te2 = time() - t2
        message = "Results are in for {}-grams".format(n)
        print(message)
        message = "\tFinished {} folds in {:.4f} s".format(folds, te2)
        print(message)
        message = "\tAverage accuracy is:{}".format(order_results["accuracy"])
        print(message)
        message = "\tAverage F measure is:{}".format(order_results["fmeasure"])
        print(message)
        order_results["order=n"]
        full_results.append(order_results)

    for resul in full_results:
        print(resul)


main()
