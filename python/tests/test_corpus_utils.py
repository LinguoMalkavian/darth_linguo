import unittest
import os
print(os.getcwd())
import corpus_tools
import tempfile
from allen_linguo import LinguoDatasetReader

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

    def test_argument_parser(self):
        parser = corpus_tools.get_corpus_tools_argparser()
        # Test the split_corpus action
        # Test the functionality with all the positional arguments
        full_split = "split_corpus test_file --named_corpus test_corp " + \
            "--piece_names piece1 piece2 --piece_ratio 0.6 0.4 --tokenized" + \
            " --min_len 5 --max_len 100 --outfile out_test"
        args = parser.parse_args(full_split.split())
        self.assertEqual("test_file", args.in_file)
        self.assertEqual("test_corp", args.named_corpus)
        self.assertEqual(["piece1", "piece2"], args.piece_names)
        self.assertEqual([0.6, 0.4], args.piece_ratio)
        self.assertEqual(True, args.tokenized)
        self.assertEqual(5, args.min_len)
        self.assertEqual(100, args.max_len)
        self.assertEqual("out_test", args.outfile)
        self.assertEqual(corpus_tools.handle_splitCorpus, args.func)

        # Test with only the mandatory arguments
        nameonly_split = "split_corpus path/to/test/file"
        args = parser.parse_args(nameonly_split.split())
        self.assertEqual("path/to/test/file", args.in_file)
        self.assertEqual(None, args.named_corpus)
        self.assertEqual(["LM1", "LM2", "GT", "GV"], args.piece_names)
        self.assertEqual([0.25, 0.25, 0.4, 0.1], args.piece_ratio)
        self.assertEqual(7, args.min_len)
        self.assertEqual(45, args.max_len)
        self.assertEqual(None, args.outfile)
        self.assertEqual(corpus_tools.handle_splitCorpus, args.func)

        # ------- Test corpus labeling action
        full_label = "label_corpus test_file 0 --named_corpus test_corp" + \
            " --ungramType WS --outfile out_test"
        args = parser.parse_args(full_label.split())
        self.assertEqual("test_file", args.in_file)
        self.assertEqual("test_corp", args.named_corpus)
        self.assertEqual(0, args.gram_label)
        self.assertEqual("out_test", args.outfile)
        self.assertEqual("WS", args.ungramType)

        short_label = "label_corpus path/to/corpus 1"
        args = parser.parse_args(short_label.split())
        self.assertEqual("path/to/corpus", args.in_file)
        self.assertEqual(None, args.named_corpus)
        self.assertEqual(1, args.gram_label)
        self.assertEqual(None, args.outfile)
        self.assertEqual("G", args.ungramType)

    def test_corpus_splitting(self):

        filename = "tests/data/small_sample.txt"
        tokenized = corpus_tools.load_raw_grammatical_corpus(filename)
        splits = {"first": 0.25, "second": 0.25, "third": 0.5}
        corpora = corpus_tools.splitCorpus(tokenized, splits)
        self.assertEqual(len(corpora["first"]), 25)
        self.assertEqual(len(corpora["second"]), 25)
        self.assertEqual(len(corpora["third"]), 50)
        unshuffled = True
        for sentence in tokenized[:25]:
            if sentence not in corpora["first"]:
                unshuffled = False
        for sentence in tokenized[25:50]:
            if sentence not in corpora["first"]:
                unshuffled = False
        for sentence in tokenized[50:]:
            if sentence not in corpora["first"]:
                unshuffled = False
        self.assertFalse(unshuffled)


if __name__ == '__main__':
    unittest.main()
