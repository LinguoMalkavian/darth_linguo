import corpus_tools


def test_corpus_paths():
    corpusName = "test"
    actual_corpusPath ="/Users/pablo/Dropbox/workspace/darth_linguo/Data/test/test"
    built_corpusPath = corpus_tools.getDataPath(corpusName)
    assert built_corpusPath == actual_corpusPath
    actual_data_folder = "/Users/pablo/Dropbox/workspace/darth_linguo/Data/test/"
    built_data_folder = corpus_tools.getDataFolderPath(corpusName)
    assert actual_data_folder == built_data_folder


def test_data_reading():
    corpusName = "test"
    built_corpus_Path = corpus_tools.getDataPath(corpusName)
    filename = built_corpus_Path + "-base"
    sentences = corpus_tools.load_raw_grammatical_corpus(filename)
    assert len(sentences) == 10
    lengths = [len(x) for x in sentences]
    assert max(lengths) <= 45
    assert min(lengths) >= 7
    for sentence in sentences:
        assert sentence[-1] == "<eos>"
        assert sentence[-2] == "."
