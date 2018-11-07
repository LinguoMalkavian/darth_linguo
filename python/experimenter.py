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
from datetime import datetime
from time import time
import sys
import corpus_tools
import os


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
        self.corpusPath = corpus_tools.getDataPath(corpus_name)
        self.word2id = corpus_tools.loadWord2Id(self.corpusPath)

    def runSingleExperiment(self, noiseName):
        """Runs one experiment, full with training and testing"""

        startingTime = time()
        # Load the training corpus

        gramTrain_fn = self.corpusPath + "-grammatical-train"
        noiseTrain_fn = self.corpusPath + "-" + noiseName + "-train"
        labeledTrain = corpus_tools.getLabeledData(gramTrain_fn, noiseTrain_fn)

        # Train the Model
        model = self.train_model(labeledTrain)
        self.saveModel(model, noiseName)
        trainDoneT = time()
        training_time = corpus_tools.seconds_to_hms(trainDoneT - startingTime)

        # Load test data
        gramTest_fn = self.corpusPath + "-grammatical-test"
        noiseTest_fn = self.corpusPath + "-" + noiseName + "-test"
        labeledTest = corpus_tools.getLabeledData(gramTest_fn, noiseTest_fn)

        # Run test
        results, falseNegs, falsePos = self.test_model(labeledTest, model)
        testing_time = corpus_tools.seconds_to_hms(time()-trainDoneT)

        # Print and Save results
        results["noise-type"] = noiseName
        results["Training time"] = training_time
        results["Testing time"] = testing_time
        resultStr = corpus_tools.makeResultsString(results)
        resultStr += self.getParametersString()
        print(resultStr)
        corpus_tools.saveResults(resultStr, self.corpusPath)
        corpus_tools.saveErrors(falseNegs, falsePos,
                                self.corpus_name, noiseName)

    def train_model(self, train_data):
        """Train a new model with the given data"""

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

    # Test the model
    def test_model(self, test_data, model):
        """ Runs a model on a list of labeled data.

        Returns a results dictionary with Accuracy, precision, and recall
        As well as a list of false positives and one of false negatives"""
        correct = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        print("Begining test")
        falseNegs = []
        falsePos = []
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
                    falseNegs.append(testcase[0])
                else:
                    fp += 1
                    falsePos.append(testcase[0])

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
        return results, falseNegs, falsePos

    def classifyInstance(self, model, instance):
        """Runs a single instance through the network"""

        prepared_inputs = self.prepare_input(instance)
        prediction_vec = model(prepared_inputs).view(2)
        if prediction_vec.data[0] > prediction_vec.data[1]:
            prediction = 0
        else:
            prediction = 1
        return prediction

    def saveModel(self, model, noiseName):
        """Saves a model to file"""
        path = corpus_tools.getModelPrefix(self.corpus_name)
        path += "VS"+noiseName+"_trained_{}".format(datetime.now())
        torch.save(model, path)
        print("Model has been saved to {}".format(path))

    def loadModel(self, noiseName=""):
        """Loads the most recently trained model"""
        modelDir = corpus_tools.getModelDir(self.corpus_name)
        availableModelPaths = os.listdir(modelDir)
        validModelPaths = [path for path in availableModelPaths
                           if path.find(noiseName) != -1]

        availableModelPaths.sort()
        if validModelPaths:
            modelPath = availableModelPaths[-1]
            model = torch.load(modelDir+"/"+modelPath)
            return model
        else:
            print("There are no trained models with desired noise")
            return None

    def average_results(self, result_list):
        total = len(result_list)
        averaged = defaultdict(float)
        for report in result_list:
            for item in report:
                averaged[item] += report[item]
        for item in averaged:
            averaged[item] = averaged[item]/total
        return averaged

    def getParametersString(self):
        """Gives a string with the experimental settings"""

        results = "\t"
        results += " Embedding:" + str(self.embed_dim)
        results += " LSTM: " + str(self.lstm_dim)
        results += " Hidden: " + str(self.hidden_dim)
        results += " Epochs: " + str(self.epochs)
        results += " learning_rate: " + str(self.learning_rate)
        results += " GPU: " + str(self.use_gpu) + "\n"

        return results

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
        exp = Experimenter(corpus_name, embed_dim, lstm_dim, hidden_dim,
                           epochs, learning_rate, use_gpu)
        exp.runSingleExperiment(noiseName)
