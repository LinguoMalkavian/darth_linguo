import corpus_tools
import tempfile
from allen_linguo import LinguoDatasetReader
import unittest
import os
import warnings


class TestCorpusHandling(unittest.TestCase):

    def test_corpus_paths(self):
        corpusName = "test"
        originalWD = os.getcwd()
        os.chdir("../Data/test")

        actual_data_folder = os.getcwd() + "/"
        actual_corpusPath = actual_data_folder + "test"

        os.chdir(originalWD)
        built_corpusPath = corpus_tools.getDataPath(corpusName)
        self.assertEqual(built_corpusPath, actual_corpusPath)
        built_data_folder = corpus_tools.getDataFolderPath(corpusName)
        self.assertEqual(actual_data_folder, built_data_folder)

    def test_data_reading(self):
        corpusName = "test"
        built_corpus_Path = corpus_tools.getDataPath(corpusName)
        filename = built_corpus_Path + "-base"
        sentences = corpus_tools.load_raw_grammatical_corpus(filename)
        self.assertEqual(len(sentences), 10)
        lengths = [len(x) for x in sentences]
        self.assertTrue(max(lengths) <= 45)
        self.assertTrue(min(lengths) >= 7)
        for sentence in sentences:
            self.assertEqual(sentence[-1], "<eos>")
            self.assertEqual(sentence[-2], ".")

    def test_unlabeled_corpus_saving(self):
        """Saves a dummy tokenized corpus to file and reads it"""

        original_corpus = [["Yo", "soy", "una", "oración", "gramatical", ",",
                           "regocíjense", "en", "mi", "glória", "."],
                           ["Yo", "ungrammatical", "es", "oración", ","
                            "tú", "presumido", "elitista", "."]]
        with tempfile.TemporaryDirectory() as temp_dir:
            fileName = temp_dir + "testfile"
            corpus_tools.save_tokenized_corpus(fileName, original_corpus)
            loaded_corpus = corpus_tools.load_tokenized_corpus(fileName)
            assert len(original_corpus) == len(loaded_corpus)
            for original_sent, loaded_sent in zip(original_corpus,
                                                  loaded_corpus):
                self.assertEqual(original_sent, loaded_sent)

    def test_labeled_corpus_saving(self):
        """Saves corpus to file as G or unG and loads with reader"""

        original_corpus = [["Yo", "soy", "una", "oración", "gramatical", ",",
                           "regocíjense", "en", "mi", "glória", "."],
                           ["Yo", "ungrammatical", "es", "oración", ","
                            "tú", "presumido", "elitista", "."]]
        reader = LinguoDatasetReader()

        with tempfile.TemporaryDirectory() as temp_dir:
            # first test the grammatical case
            fileName_asG = temp_dir + "testfile"
            corpus_tools.save_uniform_labeled_corpus(fileName_asG,
                                                     original_corpus,
                                                     g_label=1)
            loaded_asG = reader.read(fileName_asG)
            self.assertEqual(len(original_corpus), len(loaded_asG))
            for original_sent, loaded_sent in zip(original_corpus, loaded_asG):
                self.assertEqual(loaded_sent.fields["g_label"].label,
                                 "grammatical")
                self.assertEqual(loaded_sent.fields["ug_type"].label, "G")
                plain_loaded = [str(token) for
                                token in loaded_sent.fields["sentence"].tokens]
                self.assertEqual(plain_loaded, original_sent)
            # Now to test it for ungrammatical (with a valid ug_type)
            fileName_asUG = temp_dir + "testfileUG"
            corpus_tools.save_uniform_labeled_corpus(fileName_asUG,
                                                     original_corpus,
                                                     g_label=0, ug_type="WS")
            loaded_asUG = reader.read(fileName_asUG)
            self.assertEqual(len(original_corpus), len(loaded_asUG))
            for original_sent, loaded_sent in zip(original_corpus,
                                                  loaded_asUG):
                self.assertEqual(loaded_sent.fields["g_label"].label,
                                 "ungrammatical")
                self.assertEqual(loaded_sent.fields["ug_type"].label, "WS")
                plain_loaded = [str(token) for
                                token in loaded_sent.fields["sentence"].tokens]
                self.assertEqual(plain_loaded, original_sent)

    def test_corpus_labeling(self):
        """Loads corpus from file and saves it with labels, then reads it """
        corpusName = "test"
        built_corpus_Path = corpus_tools.getDataPath(corpusName)
        filename = built_corpus_Path + "-GT"
        reader = LinguoDatasetReader()
        with tempfile.TemporaryDirectory() as temp_dir:
            outpath = temp_dir + "-labeled"
            corpus_tools.labelCorpus(filename, outpath,
                                     g_label=0, ug_type="WS")
            original = corpus_tools.load_tokenized_corpus(filename)
            loaded = reader.read(outpath)
            for original_sent, loaded_sent in zip(original, loaded):
                self.assertEqual(loaded_sent.fields["g_label"].label,
                                 "ungrammatical")
                self.assertEqual(loaded_sent.fields["ug_type"].label, "WS")
                plain_loaded = [str(token) for
                                token in loaded_sent.fields["sentence"].tokens]
                self.assertEqual(plain_loaded, original_sent)


if __name__ == '__main__':
    unittest.main()
