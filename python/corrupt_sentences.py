import spacy
import transformer
import sys


def main():
    # Load the model
    print("Loading Spacy Spanish model")
    nlp_mod = spacy.load('es_core_news_sm')
    print("Model loaded")
    # Initialize the array of corruptors
    corruptortypes = ["prepRM", "verbRM", "verbInfl", "adjInfl"]
    corruptors = {}
    corruptors["prepRM"] = transformer.PrepRemover("prepRM")
    corruptors["verbRM"] = transformer.VerbRemover("verbRM")
    corruptors["verbInfl"] = transformer.VerbInflCorruptor("verbInfl")
    corruptors["adjInfl"] = transformer.AdjInflCorruptor("adjInfl")
    # Initialize counters for corrupted sentences
    corruptCount = {}
    uncorrupted_count = 0
    for typ in corruptortypes:
        corruptCount[typ] = 0

    # Load sentence generator
    in_corpus_filename = sys.argv[1]
    out_corpus_folder = in_corpus_filename + "."
    in_corpus_file = open(in_corpus_filename, "r")
    sentence_gen = sentence_generator(in_corpus_file, nlp_mod)

    # Create outfiles for each type of corrupted sentence
    outfiles = {}
    for kind in corruptortypes:
        outname = out_corpus_folder + "corrupted_by." + kind
        outfiles[kind] = open(outname, "w")

    processed_count = 0
    # Iterate parsed sentences and test for coruptibility
    print("Begining Corruption")
    for parsed_sentence in sentence_gen:
        posib_trans = []
        # Test for each corruptor
        for corr in corruptortypes:
            possibility = corruptors[corr].test_possible(parsed_sentence)
            if possibility:
                posib_trans.append(corr)

        # Choose corruptor that has the fewest sentences so far
        select = None
        selectCount = float("inf")
        for possib in posib_trans:
            if corruptCount[possib] < selectCount:
                select = possib
                selectCount = corruptCount[possib]

        if select:
            # Corrupt sentence
            corruptedVersion = corruptors[select].transform(parsed_sentence)
            if corruptedVersion != -1:
                # Save corrupted sentence to corresponding file
                outfiles[select].write(corruptedVersion)
                corruptCount[select] += 1
            else:
                uncorrupted_count += 1
        else:
            uncorrupted_count += 1
        processed_count += 1
        # Progress checker, for sanity
        if processed_count % 500 == 0:
            print("{} sentences processed".format(processed_count))
    # Close files
    for kind in outfiles:
        outfiles[kind].close()
    # Print summary to console
    total = 0
    for trans_type in corruptCount:
        print(trans_type + ":" + str(corruptCount[trans_type]))
        total += corruptCount[trans_type]
    print("Total: {0}".format(total))
    print("Incorruptible: {0}".format(uncorrupted_count))


def sentence_generator(in_file, nlp_mod):
    """"Lazyly reads sentences and pases them through the processing pipeline.

    in_file: the file object with a sentence per line
    nlp_mod: a loaded spacy nlp module
    """
    nextline = "start"
    while nextline:
        nextline = in_file.readline()
        sentence_obj = nlp_mod(nextline)
        yield sentence_obj


main()
