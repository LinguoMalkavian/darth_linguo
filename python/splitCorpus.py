import sys
import corpus_tools
import random
import math


def preprocess_file(filepath, isRaw, withValidation, trainProp, testProp):
    """Run the preprocessing on the specified file"""

    # Load the file contents
    if isRaw:
        sentences = corpus_tools.load_raw_grammatical_corpus(filepath)
    else:
        sentences = corpus_tools.load_tokenized_corpus(filepath)

    random.shuffle(sentences)

    if withValidation:
        firstThreshold = int(math.floor(trainProp * len(sentences)))
        secondThreshold = firstThreshold
        secondThreshold += int(math.floor(testProp * len(sentences)))
        train_sentences = sentences[:firstThreshold]
        test_sentences = sentences[firstThreshold:secondThreshold]
        validation_sentences = sentences[secondThreshold:]
        corpus_tools.save_tokenized_corpus(filepath+"-val",
                                           validation_sentences)
    else:
        firstThreshold = int(math.floor(trainProp * len(sentences)))
        train_sentences = sentences[:firstThreshold]
        test_sentences = sentences[firstThreshold:]
    corpus_tools.save_tokenized_corpus(filepath+"-train", train_sentences)
    corpus_tools.save_tokenized_corpus(filepath+"-test", test_sentences)


if __name__ == "__main__":
    try:
        filepath = sys.argv[1]
    except IndexError:
        print("Please provide a file name")
    else:
        isRaw = False
        withValidation = False
        if len(sys.argv) >= 3 and sys.argv[2].lower() == "true":
            isRaw = True
        if len(sys.argv) >= 4 and sys.argv[3].lower() == "true":
            withValidation = True
        if len(sys.argv) >= 5:
            trainProp = int(sys.argv[4])/100
        else:
            trainProp = 0.80
        if withValidation and len(sys.argv) >= 6:
            testProp = sys.argv[5]/100
        else:
            testProp = 0.2
        preprocess_file(filepath, isRaw, withValidation, trainProp, testProp)
