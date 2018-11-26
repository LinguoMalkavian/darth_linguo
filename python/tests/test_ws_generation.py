from corpus_tools import load_raw_grammatical_corpus
import tempfile
from allen_linguo import LinguoDatasetReader
import unittest
import generateWS


class TestWSGeneration(unittest.TestCase):

    def setUp(self):
        self.tiny_corpus = [["I", "am", "Sam", ".", "<eos>"],
                            ["Sam", "I", "am", ".", "<eos>"]]

        filename = "tests/data/green_eggs.txt"
        self.small_corpus = load_raw_grammatical_corpus(filename,
                                                        minlength=6,
                                                        maxlength=11)
        self.freqs = {}
        for n in [2, 3, 4, 5, 6]:
            self.freqs[n] = generateWS.extract_ngram_freq(self.small_corpus, n)

    def test_ngram_freq_small(self):
        bigram_freqs = generateWS.extract_ngram_freq(self.tiny_corpus, 2)

        self.assertEqual(bigram_freqs["am"]["Sam"], 1)
        self.assertEqual(bigram_freqs["am"]["."], 1)
        self.assertEqual(bigram_freqs["."]["<eos>"], 2)
        self.assertEqual(bigram_freqs["I"]["am"], 2)
        self.assertEqual(bigram_freqs["Sam"]["I"], 1)
        self.assertEqual(bigram_freqs["Sam"]["."], 1)
        self.assertEqual(bigram_freqs["#"]["I"], 1)
        self.assertEqual(bigram_freqs["#"]["Sam"], 1)

        trigram_freqs = generateWS.extract_ngram_freq(self.tiny_corpus, 3)

        self.assertEqual(trigram_freqs["# #"]["I"], 1)
        self.assertEqual(trigram_freqs["# #"]["Sam"], 1)
        self.assertEqual(trigram_freqs["# I"]["am"], 1)
        self.assertEqual(trigram_freqs["# Sam"]["I"], 1)
        self.assertEqual(trigram_freqs["I am"]["Sam"], 1)
        self.assertEqual(trigram_freqs["I am"]["."], 1)
        self.assertEqual(trigram_freqs["Sam ."]["<eos>"], 1)
        self.assertEqual(trigram_freqs["I am"]["."], 1)
        self.assertEqual(trigram_freqs["I am"]["Sam"], 1)
        self.assertEqual(trigram_freqs["am ."]["<eos>"], 1)

    def test_ngram_freq_big(self):
        bigram_freqs = generateWS.extract_ngram_freq(self.small_corpus, 2)
        trigram_freqs = generateWS.extract_ngram_freq(self.small_corpus, 3)
        tetragram_freqs = generateWS.extract_ngram_freq(self.small_corpus, 4)
        for trigram in trigram_freqs:
            parts = trigram.split(" ")
            if parts[0] != "#":
                count = sum([trigram_freqs[trigram][end]
                             for end in trigram_freqs[trigram]])
                self.assertEqual(count,
                                 bigram_freqs[parts[0]][parts[1]])

        for tetragram in tetragram_freqs:
            parts = tetragram.split(" ")
            prefix = " ".join(parts[:-1])
            cont = parts[-1]
            if parts[0] != "#":
                count = sum([tetragram_freqs[tetragram][end]
                             for end in tetragram_freqs[tetragram]])
                self.assertEqual(count,
                                 trigram_freqs[prefix][cont])

    def test_generateWordSalad(self):
        """Test that individual word salads are being generated adequately"""
        for n in [2, 3, 4, 5, 6]:
            for _ in range(1000):
                word_salad = generateWS.generateWordSalad(self.freqs[n],
                                                          n,
                                                          minlength=5,
                                                          maxlength=10)
                self.assertNotEqual(word_salad[0], "#")
                self.assertEqual(word_salad[-1], "<eos>")
                self.assertTrue(len(word_salad) >= 3)
                self.assertTrue(len(word_salad) <= 10)
