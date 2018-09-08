# Standard pytorch imports
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
# other utilities
from collections import defaultdict
import math
from time import time
import datetime
import sys
import corpus_tools


class Experimenter():

    def __init__(self,
                 corpus_name=None,
                 embed_dim=32,
                 lstm_dim=32,
                 hidden_dim=64,
                 epochs=3,
                 learning_rate=0.1,
                 use_gpu=False):
        # Modify parameters here
        self.corpus_name = corpus_name
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.word2id = {}

    def runSingleExperiment(self, corpusPath, noiseName):
        # Load the training corpus
        self.word2id = corpus_tools.loadWord2Id(corpusPath)
        gramTrain_fn = corpusPath + "-grammatical-train"
        noiseTrain_fn = corpusPath + "-" + noiseName + "-train"
        gramTrain = corpus_tools.load_tokenized_corpus(gramTrain_fn)
        labeledGramTrain = [(sentence, 1) for sentence in gramTrain]
        noiseTrain = corpus_tools.load_tokenized_corpus(noiseTrain_fn)
        labeledNoiseTrain = [(sentence, 0) for sentence in noiseTrain]
        labeledTrain = labeledGramTrain + labeledNoiseTrain
        random.shuffle(labeledTrain)
        # Train the Model
        model = self.train_model(labeledTrain)

        # Get test data
        gramTest_fn = corpusPath + "-grammatical-test"
        noiseTest_fn = corpusPath + "-" + noiseName + "-test"
        gramTest = corpus_tools.load_tokenized_corpus(gramTest_fn)
        labeledGramTest = [(sentence, 1) for sentence in gramTest]
        noiseTest = corpus_tools.load_tokenized_corpus(noiseTest_fn)
        labeledNoiseTest = [(sentence, 0) for sentence in noiseTest]
        labeledTest = labeledGramTest + labeledNoiseTest
        results = self.test_model(labeledTest, model)
        results["noise-type"] = noiseName
        resultStr = self.makeResultsString(results)
        print(resultStr)
        self.saveResults(resultStr, corpusPath)

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
        results = {"accuracy": accuracy,
                   "precision": precision,
                   "recall": recall,
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
                idxs.append(self.word2id["<unk>"])
        if self.use_gpu:
            tensor = torch.LongTensor(idxs).cuda()
            return autograd.Variable(tensor).cuda()
        else:
            tensor = torch.LongTensor(idxs)
            return autograd.Variable(tensor)

    def saveResults(self, resultStr, corpusPath):
        results_fn = corpusPath+"-results"
        with open(results_fn, "a+") as resultsFile:
            resultsFile.write(resultStr)

    def makeResultsString(self, results):
        response = ""
        date = datetime.datetime.now()
        header = "Experiment: grammatical V.S {} noise\n".format(
                                                    results["noise-type"])
        response += header
        timestr = "realized on {}:".format(str(date))
        response += timestr
        for key in results:
            if key != "noise-type":
                response += "{}:{}\n".format(key, results[key])
        return response

    # Methods for specific experiments ---------------------------

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
    def __init__(self, embedding_dim, vocab_size,
                 lstm_dim, hidden_dim, use_gpu):
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


if __name__ == "__main__":
    try:
        corpus_name = sys.argv[1]
        embed_dim = int(sys.argv[2])
        lstm_dim = int(sys.argv[3])
        hidden_dim = int(sys.argv[4])
        epochs = int(sys.argv[5])
        learning_rate = float(sys.argv[6])
        if sys.argv[7].lower() == "true":
            use_gpu = True
        else:
            use_gpu = False
        noiseName = sys.argv[8]
        corpusPath = corpus_tools.getDataPath(corpus_name)
    except IndexError:
        print("Arguments missing, please remember to provide the following:")
        print("In order:")
        print("Corpus name (corpus must be stored in the Data folder)")
        print("Embeding dimension (int)")
        print("LSTM dimension (int)")
        print("Hidden dimension (int)")
        print("Number of epochs (int)")
        print("Learning rate (float)")
        print("Use GPU (boolean)")
        print("Noise name(string), (what comes after the base name)")
    else:
        exp = Experimenter(corpusPath, embed_dim, lstm_dim, hidden_dim,
                           epochs, learning_rate, use_gpu)
        exp.runSingleExperiment()
