
# Standard pytorch imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# NLTK modules
from nltk.tokenize.moses import MosesTokenizer
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

    def __init__(self, corpus_name, max_ngram, hap_thresh, folds,
                 train_proportion, embed_dim, lstm_dim, hidden_dim,
                 epochs, learning_rate):
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

    def run(self):
        talpha = time()
        # Load and preprocess the grammatical part of the corpus
        self.prepro_gram = preprocessCorpus(corpus_name, hap_thresh,)

        full_results = []
        # Run for each n, with x-fold cross validation
        for n in range(1, max_ngram+1):
            full_results.append(run_experiment(n))
        for resul in full_results:
            print(resul)
        ttotal = time() - talpha
        timestr = seconds_to_hms(ttotal)
        message = "The whole process on gpu took{}".format(timestr)
        print(message)

    def preprocessCorpus(self):

        t1 = time()
        corpus = load_grammatical_corpus(self.corpus_name)
        self.ucounts = get_vocabulary(corpus)
        # This now holds the preprocessed grammatical data
        prepro_gram = token_replacement(corpus, hapaxes)
        message = "Your corpus has {sent} grammatical sentences".format(
                                                            sent=len(prepro_gram))
        print(message)
        message = "Grammatical corpus loaded in {:.3f} seconds".format(time()-t1)



        lengths = [len(sent) for sent in prepro_gram]
        self.avg_sent_length = np.mean(lengths)
        self.length_sd = np.std(lengths)

        return prepro_gram, avg_sent_length, length_sd, probdist

    def run_experiment(self, n):
        t2 = time()
        # Generate the word salads
        labeled_gramatical = [[sentence, 1] for sentence in prepro_gram]
        labeled_ws = self.generateData(n)
        cutoff = math.floor(self.train_proportion * len(labeled_gramatical))
        # Iterate over the number of folds

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
        order_results["order"] = n

        return order_results

    def generateData(self, n):

        t1 = time()
        nsal = len(self.prepro_gram)
        if n == 1:
            word_salads = [generateWSuni(self.ucounts,
                                         self.avg_sent_length,
                                         self.length_sd) for _ in range(nsal)]
        else:
            n_freqs = extract_ngram_freq(prepro_gram, n)
            word_salads = [generateWSNgram(n_freqs, n) for _ in range(nsal)]

        labeled_ws = [[sentence, 0] for sentence in word_salads]

        message = "Word salad data has been generated for order {}".format(n)
        print(message)
        te = time() - t1
        message = "\t{} word salads generated in {:.3f} seconds".format(nsal,
                                                                        te)
        print(message)
        message = "Starting experiment on {}-grams ...".format(n)
        return labeled_ws

    def load_grammatical_corpus(self, input_corpus_filename):
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
        # tokenizer = MosesTokenizer()
        # tokenized = [tokenizer.tokenize(sentence) for sentence in real_text]
        tokenized = [word_tokenize(sentence) for sentence in real_text]
        return tokenized


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
        self.word_to_ix = {}
        for word in self.vocabulary:
            self.word_to_ix[word] = len(self.word_to_ix)
        return counts


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


    def generateWSuni( probdist, avg_length, sd):
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
        draw = choice(self.vocabulary length, probdist).tolist()
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

    def generateWSNgram(n_frequencies, avg_length, sd, order):
        # Method to generate one word salad sentence usin unigram distribution
        # Vocab is a list of vocabulary words
        # probdist contains the probabilities of vocabulary words in same order
        # avg_length is the average length of sentences
        # sd is the standar deviation for the legths of sentences
        unicounts = self.ucounts
        message = "Generating word salads of order {}...".format(n)
        # Draw the length
        length = math.floor(random.gauss(avg_length, sd))
        if length < 5:
            length = 5

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
        voc_size = len(self.word_to_ix)
        # Initialize model
        linguo = Linguo(self.embed_dim,
                        voc_size,
                        self.lstm_dim,
                        self.hidden_dim).cuda()
        optimizer = optim.SGD(linguo.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss().cuda()

        for i in range(epochs):
            epoch_loss = 0
            random.shuffle(train_data)
            for data, label in train_data:
                # Restart gradient
                linguo.zero_grad()
                # Run model
                in_sentence = prepare_input(word_to_ix, data).cuda()
                target = autograd.Variable(torch.LongTensor([label])).cuda()
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


    def seconds_to_hms(secondsin):
        hours = math.floor(secondsin / 3600)
        remain = secondsin % 3600
        minutes = math.floor(remain / 60)
        seconds = math.floor(remain % 60)
        answer = "{h} hours, {m} minutes and {s} seconds".format(h=hours,
                                                                 m=minutes,
                                                                 s=seconds)
        return answer

    def prepare_input(self, sentence):
        idxs = []
        for word in sentence:
            if word in self.word_to_ix:
                idxs.append(self.word_to_ix[word.lower()])
            else:
                idxs.append(self.word_to_ix["#unk"])
        tensor = torch.LongTensor(idxs).cuda()
        return autograd.Variable(tensor).cuda()


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
        var1 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
        var2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
        hidden_state = (var1, var2)
        return hidden_state





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
    exp = Experimenter(corpus_name, max_ngram, hap_thresh, folds,
                       train_proportion, embed_dim, lstm_dim, hidden_dim
                       epochs, learning_rate)
    exp.run()
