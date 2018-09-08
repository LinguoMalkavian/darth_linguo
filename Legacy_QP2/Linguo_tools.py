
# Standard pytorch imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm

# NLTK modules
from nltk import word_tokenize

# other utilities
import numpy as np
from numpy.random import choice
from collections import defaultdict
import math
from string import capwords
from time import time
import sys


class Experimenter():

    def __init__(self,
                 corpus_name=None,
                 max_ngram=5,
                 hap_thresh=1,
                 folds=10,
                 train_proportion=0.8,
                 embed_dim=32,
                 lstm_dim=32,
                 hidden_dim=64,
                 epochs=3,
                 learning_rate=0.1,
                 use_gpu=False):
        # Modify parameters here
        self.corpus_name = corpus_name
        self.max_ngram = max_ngram
        self.hap_thresh = hap_thresh
        self.folds = folds
        self.train_proportion = train_proportion
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.word2id = {}

    def preprocessCorpus(self):

        t1 = time()
        corpus = self.load_raw_grammatical_corpus(self.corpus_name)
        self.ucounts, self.hapaxes = self.get_vocabulary(corpus)
        # This now holds the preprocessed grammatical data
        prepro_gram = self.token_replacement(corpus, self.hapaxes)
        message = "Your corpus has {sent} grammatical sentences".format(
                                                        sent=len(prepro_gram))
        print(message)
        timestr = seconds_to_hms(time()-t1)
        message = "Grammatical corpus loaded in {} seconds".format(timestr)

        lengths = [len(sent) for sent in prepro_gram]
        self.avg_sent_length = np.mean(lengths)
        self.length_sd = np.std(lengths)

        return prepro_gram


    def generateWSData(self, n):

        t1 = time()
        nsal = len(self.prepro_gram)
        if n == 1:
            word_salads = [self.generateWSuni(self.ucounts,
                                              self.avg_sent_length,
                                              self.length_sd)
                           for _ in range(nsal)]
        else:
            n_freqs = self.extract_ngram_freq(self.prepro_gram, n)
            word_salads = [self.generateWSNgram(n_freqs, n)
                           for _ in range(nsal)]

        labeled_ws = [[sentence, 0] for sentence in word_salads]

        message = "Word salad data has been generated for order {}".format(n)
        print(message)
        te = seconds_to_hms(time() - t1)
        message = "\t{} word salads generated in {} seconds".format(nsal, te)
        print(message)
        message = "Starting experiment on {}-grams ...".format(n)
        return labeled_ws

    def load_raw_grammatical_corpus(self, input_corpus_filename, minlength=7):
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
                        tokenized_sentences.append(line.strip())
                elif line.strip()[-1] == "?" or line.strip()[-1] == "!":
                    inter_excl += 1
            numlines += 1

        n_sent = len(tokenized_sentences)
        print('''Full corpus has {} sentences,
        \t {} were dumped,
        with {} interogatives or exclamatives.'''.format(n_sent,
                                                         numlines-n_sent,
                                                         inter_excl))

        random.shuffle(tokenized_sentences)
        # tokenizer = MosesTokenizer()
        # tokenized = [tokenizer.tokenize(sentence) for sentence in real_text]
        return tokenized_sentences

    # Method to extract:
    # The word2idx, a dictionary from vocabulary words to unique integers
    # The hapaxes, a list with words with count less than the threshold
    # Vocabulary and probdist are also generated for the unigram case
    def get_vocabulary(self, sentences):
        hap_threshold = self.hap_thresh
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
        self.vocabulary = []
        self.probdist = []
        for key in counts:
            self.vocabulary.append(key)
            self.probdist.append(counts[key])

        # Define the vocabulary and word ids
        self.vocabulary.append(".")
        self.word2id = {}
        for word in self.vocabulary:
            self.word2id[word] = len(self.word2id)
        return counts, hapaxes

    # Method to extract n-gram frequencies from the corpus
    # as well as length statistics
    # Corpus is a list of sentences,
    # each sentence represented by a list of tokens
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
    def token_replacement(self, sentences, hapaxes):
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

    def generateWSuni(self, probdist, avg_length, sd):
        # Method to generate one word salad sentence usin unigram distribution
        # Vocab is a list of vocabulary words
        # probdist contains the probabilities of vocabulary words in same order
        # avg_length is the average length of sentences
        # sd is the standar deviation for the legths of sentences

        # Draw the length
        length = math.floor(random.gauss(avg_length, sd))
        while length < 6:
            length = math.floor(random.gauss(avg_length, sd))
        # Draw the words
        draw = choice(self.vocabulary, length, probdist).tolist()
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

    def generateWSNgram(self, n_frequencies, order):
        # Method to generate one word salad sentence usin unigram distribution
        # Vocab is a list of vocabulary words
        # probdist contains the probabilities of vocabulary words in same order
        # avg_length is the average length of sentences
        # sd is the standar deviation for the legths of sentences
        unicounts = self.ucounts
        message = "Generating word salads of order {}...".format(order)
        # Draw the length
        print(message)
        length = math.floor(random.gauss(self.avg_length, self.length_sd))
        while length < 6:
            length = math.floor(random.gauss(avg_length, sd))

        sentence = ["#"]*(order-1)
        for i in range(length+order-1):
            prefix = " ".join(sentence[-(order-1):])
            try:
                self.vocabulary, freqs = zip(*n_frequencies[prefix].items())
                word = choice(self.vocabulary, 1, freqs)[0]
                sentence.append(word)
            except KeyError:
                self.vocabulary, freqs = zip(*unicounts.items())
                word = choice(self.vocabulary, 1, freqs)[0]
                sentence.append(word)
        sentence.append(".")
        return sentence

    # Training time! Cue Eye of the Tiger
    def train_model(self, train_data):
        voc_size = len(self.word2id)
        # Initialize model
        if self.use_gpu:
            linguo = Linguo(self.embed_dim,
                            voc_size,
                            self.lstm_dim,
                            self.hidden_dim, use_gpu=True).cuda()
            optimizer = optim.SGD(linguo.parameters(), lr=self.learning_rate)
            loss_function = nn.NLLLoss().cuda()
        else:
            linguo = Linguo(self.embed_dim,
                            voc_size,
                            self.lstm_dim,
                            self.hidden_dim, use_gpu=False)
            optimizer = optim.SGD(linguo.parameters(), lr=self.learning_rate)
            loss_function = nn.NLLLoss()

        for i in range(self.epochs):
            print("Epoch {}".format(i+1))
            epoch_loss = 0
            random.shuffle(train_data)
            for data, label in tqdm(train_data):
                # Restart gradient
                linguo.zero_grad()
                # Run model
                if self.use_gpu:
                    in_sentence = self.prepare_input(data).cuda()
                    target = autograd.Variable(torch.LongTensor([label])
                                               ).cuda()
                else:
                    in_sentence = self.prepare_input(data)
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
    def test_model(self, test_data, model):
        correct = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        print("Begining test")
        for testcase in tqdm(test_data):
            target = testcase[1]
            prepared_inputs = self.prepare_input(testcase[0])
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

    def average_results(self, result_list):
        total = len(result_list)
        averaged = defaultdict(float)
        for report in result_list:
            for item in report:
                averaged[item] += report[item]
        for item in averaged:
            averaged[item] = averaged[item]/total
        return averaged

    def prepare_input(self, sentence):
        idxs = []
        for word in sentence:
            if word in self.word2id:
                idxs.append(self.word2id[word.lower()])
            else:
                idxs.append(self.word2id["#unk"])
        if self.use_gpu:
            tensor = torch.LongTensor(idxs).cuda()
            return autograd.Variable(tensor).cuda()
        else:
            tensor = torch.LongTensor(idxs)
            return autograd.Variable(tensor)

    def load_corpora_from_file(self, gram_corpus_fn, ws_corpus_fn):
        with open(ws_corpus_fn) as ws_file:
            labeled_ws = []
            vocab = set()
            for line in ws_file.readlines():
                tokens = []
                for token in line.split():
                    vocab.add(token.lower())
                    tokens.append(token.lower())
                instance = [tokens[:], 0]
                labeled_ws.append(instance)

        with open(gram_corpus_fn) as gram_file:
            labeled_gram = []
            for line in gram_file.readlines():
                tokens = []
                for token in line.split():
                    tokens.append(token.lower())
                instance = [tokens[:], 1]
                labeled_gram.append(instance)
        self.vocabulary = vocab
        for word in self.vocabulary:
            self.word2id[word] = len(self.word2id)
        return labeled_gram, labeled_ws

    # Methods for specific experiments ---------------------------

    # Runs cross validation experiment with the settings of the esperimenter
    # for all n-gram orders, with individual data generation at each orders
    # Extremely inefficient
    def crossv_run(self):
        talpha = time()
        # Load and preprocess the grammatical part of the corpus
        self.prepro_gram = self.preprocessCorpus()

        full_results = []
        # Run for each n, with x-fold cross validation
        for n in range(1, self.max_ngram+1):
            full_results.append(self.run_crossv_experiment(n))
        for resul in full_results:
            print(resul)
        ttotal = time() - talpha
        timestr = seconds_to_hms(ttotal)
        message = "The whole process took{}".format(timestr)
        print(message)

    # Runs Cross-validation for a specific n-gram orders
    # Does data generation, extremely ineficient
    def run_crossv_experiment(self, n):
        t2 = time()
        # Generate the word salads
        labeled_gramatical = [[sentence, 1] for sentence in self.prepro_gram]
        labeled_ws = self.generateWSData(n)
        cutoff = math.floor(self.train_proportion * len(labeled_gramatical))
        # Iterate over the number of folds
        result_list = []
        for fold in range(self.folds):
            t1 = time()
            message = "Starting training on fold {} for {}-grams...".format(
                                                                    fold+1, n)
            print(message)
            # Shuffle and split data
            random.shuffle(labeled_gramatical)
            random.shuffle(labeled_ws)

            train_g = labeled_gramatical[:cutoff]
            test_g = labeled_gramatical[cutoff:]
            train_ws, test_ws = labeled_ws[:cutoff], labeled_ws[cutoff:]

            train_data = train_g + train_ws
            random.shuffle(train_data)

            test_data = test_g + test_ws
            random.shuffle(test_data)

            # Train the Model
            model = self.train_model(train_data)
            te = seconds_to_hms(time() - t1)
            message = '''Training finished in {}.
                Starting testing...'''.format(te)
            print(message)
            print("...")
            t1 = time()
            # Test the Model
            fold_results = self.test_model(test_data, model)
            result_list.append(fold_results)
            te = seconds_to_hms(time() - t1)
            message = "Testing finished in {} seconds".format(te)
            print(message)
            message = "\tAccuracy is {}".format(fold_results['accuracy'])
            print(message)

        order_results = self.average_results(result_list)
        te2 = time() - t2
        message = "Results are in for {}-grams".format(n)
        print(message)
        message = "\tFinished {} folds in {:.4f} s".format(folds, te2)
        print(message)
        message = "\tAverage accuracy is:{}".format(order_results["accuracy"])
        print(message)
        message = "\tAverage F measure is:{}".format(order_results["fmeasure"])
        print(message)
        order_results["order"] = n

        return order_results


class Linguo(nn.Module):
    def __init__(self, embedding_dim, vocab_size, lstm_dim, hidden_dim, use_gpu):
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
        self.use_gpu = use_gpu
        self.hstate = self.init_hstate(use_gpu)

    def forward(self, inputsentence):
        self.hstate = self.init_hstate(self.use_gpu)
        embeds = self.word_embeddings(inputsentence)
        lstm_out, self.hstate = self.lstm(embeds.view(len(inputsentence),
                                                      1,
                                                      -1), self.hstate)
        decision_lin = self.hidden2dec(lstm_out[-1])
        # print(decision_lin)
        decision_fin = F.log_softmax(decision_lin)
        return decision_fin

    def init_hstate(self, use_gpu):
        if use_gpu:
            var1 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
            var2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
        else:
            var1 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            var2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hidden_state = (var1, var2)
        return hidden_state


def seconds_to_hms(secondsin):
    hours = math.floor(secondsin/3600)
    remain = secondsin % 3600
    minutes = math.floor(remain/60)
    seconds = math.floor(remain % 60)
    answer = "{h} hours, {m} minutes and {s} seconds".format(h=hours,
                                                             m=minutes,
                                                             s=seconds)
    return answer

if __name__ == "_main_":
    corpus_name = sys.argv[1]
    max_ngram = int(sys.argv[2])
    hap_thresh = int(sys.argv[3])
    folds = int(sys.argv[4])
    train_proportion = float(sys.argv[5])
    embed_dim = int(sys.argv[6])
    lstm_dim = int(sys.argv[7])
    hidden_dim = int(sys.argv[8])
    epochs = int(sys.argv[9])
    learning_rate = float(sys.argv[10])
    use_gpu = bool(sys.argv[11])
    exp = Experimenter(corpus_name, max_ngram, hap_thresh, folds,
                       train_proportion, embed_dim, lstm_dim, hidden_dim,
                       epochs, learning_rate, use_gpu)
    exp.crossv_run()
