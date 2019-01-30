import spacy
import corpus_tools
import corruption_tools
import os
from tqdm import tqdm

nlp_med = spacy.load('es_core_news_md')

# Initialize the array of corruptors
corruptortypes = ["verbRM", "verbInfl", "adjInfl"]
corruptors = {}
corruptors["verbRM"] = corruption_tools.VerbRemover("verbRM", nlp_med)
corruptors["verbInfl"] = corruption_tools.VerbInflCorruptor("verbInfl", nlp_med)
corruptors["adjInfl"] = corruption_tools.AdjInflCorruptor("adjInfl", nlp_med)
# Initialize counters for corrupted sentences
corruptCount = {}
uncorrupted_count = 0
for typ in corruptortypes:
    corruptCount[typ] = 0

# Load sentence generator
in_corpus_filename = sys.argv[1]
#in_corpus_filename = os.path.abspath("../data/exp1_mini/exp1_mini-CT")
out_corpus_folder = in_corpus_filename + "_"
in_corpus_file = open(in_corpus_filename, "r")
sentence_gen = corpus_tools.sentence_generator(in_corpus_file, nlp_med)

# Create outfiles for each type of corrupted sentence
outfiles = {}
for kind in corruptortypes:
    outname = out_corpus_folder + "corrupted-by_" + kind
    outfiles[kind] = open(outname, "w")

processed_count = 0
# Iterate parsed sentences and test for coruptibility
print("Begining Corruption")
for parsed_sentence in tqdm(sentence_gen):

    # Test for each corruptor, store the possible transformations
    possib_trans = {}
    for cor_type in corruptortypes:
        target = corruptors[cor_type].test_possible(parsed_sentence)
        if target is not None:
            possib_trans[cor_type] = target

    success = False
    while possib_trans and not success:
        # Choose the valid corruption with the fewest sentences
        kind, target = corruption_tools.select_corruption(possib_trans,
                                                          corruptCount)
        # Corrupt sentence
        corruptedVersion = corruptors[kind].transform(parsed_sentence, target)
        if corruptedVersion is not None:
            # Save corrupted sentence to corresponding file
            outfiles[kind].write(corruptedVersion + " <eos>\n")
            corruptCount[kind] += 1
            # Finish the while loop
            success = True
    if not success:
        uncorrupted_count += 1
    processed_count += 1

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
