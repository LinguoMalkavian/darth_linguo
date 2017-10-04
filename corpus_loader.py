import tree_handler
import transformer
import sys


def main():

    # Initialize the array of corruptors
    corruptortypes = ["subjectRM", "verbRM", "verbInfl", "adjInfl"]
    corruptors = {}
    corruptors["subjectRM"] = transformer.SubjectRemover("subjectRM")
    corruptors["verbRM"] = transformer.VerbRemover("verbRM")
    corruptors["verbInfl"] = transformer.VerbInflCorruptor("verbInfl")
    corruptors["adjInfl"] = transformer.AdjInflCorruptor("adjInfl")
    # Initialize counters for corrupted sentences
    corruptCount = {}
    uncorrupted_count = 0
    for typ in corruptortypes:
        corruptCount[typ] = 0

    # Load parsed sentences
    in_corpus_filename = sys.argv[1]
    out_corpus_folder = in_corpus_filename + "/"
    in_corpus_file = open(in_corpus_filename, "r")

    # TODO handle the way to split the pased sentence file
    sentence_generator = tree_handler.sentence_obj_generator(in_corpus_file)

    # Create outfiles for each type of corrupted sentence
    outfiles = {}
    for kind in corruptortypes:
        outname = out_corpus_folder + "corrupted_by." + kind
        outfiles[kind] = open(outname, "w")

    # Iterate parsed sentences and test for coruptibility
    for parsed_sentence in sentence_generator:
        posib_trans = []
        # Test for each corruptor
        for corr in corruptortypes:
            possibility = corruptors[corr].test_possible(parsed_sentence)
            if possibility:
                posib_trans.append(corr)

        # Choose corruptor that has the fewest sentences so far
        select = None
        selectCount = int("inf")
        for possib in posib_trans:
            if corruptCount[possib] < selectCount:
                select = possib
                selectCount = corruptCount[possib]

        if select:
            # Corrupt sentence
            corruptedVersion = corruptors[select].transform(parsed_sentence)
            # Save corrupted sentence to corresponding file
            outfiles[select].write(corruptedVersion + "\n")
            corruptCount[select] += 1
        else:
            uncorrupted_count += 1
    # Print summary to console
    total = 0
    for trans_type in corruptCount:
        print(trans_type + ":" + corruptCount[trans_type])
        total += corruptCount[trans_type]
    print("Total:" + total)
    print("Incorruptible:" + uncorrupted_count)


main()
