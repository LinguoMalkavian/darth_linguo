import sys
import json
from collections import defaultdict
from tqdm import tqdm
import os

def calculate_pair_accuracy(prediction_filename):
    gram = []
    ugram = []
    accuracy_list = []
    correct_count = 0
    with open(prediction_filename,"r") as pred_file:
        for line in pred_file:
            item = json.loads(line)
            if item["gold_label"] == "grammatical":
                gram.append(item)
            else:
                ugram.append(item)
    assert len(gram)==len(ugram),f"Error in file {prediction_filename}, there are {len(gram)} grammatical versions and {len(ugram)} ungrammatical ones." 
    for original,corrupt in zip(gram,ugram):
        if (original["predicted_label"] == "grammatical"
                and corrupt["predicted_label"] == "ungrammatical"):
            correct_count += 1
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)

    return correct_count/len(accuracy_list), accuracy_list

if __name__ == "__main__":
    full_exp_name = sys.argv[1]
    root_folder = os.getcwd()
    results_folder = os.path.join(root_folder, "results", "pairtest", "classifier_results") 
    accuracies = []
    sample_nums = []
    for noise_type in ["VA", "AA", "RV"]:
        pred_file = os.path.join(results_folder, f"{full_exp_name}_{noise_type}_results")
        accuracy_num, res_list = calculate_pair_accuracy(pred_file) 
        accuracies.append(accuracy_num)
        sample_nums.append(len(res_list))
    total = sum([accuracy*num for accuracy,num in zip(accuracies,sample_nums)])/sum(sample_nums)
    response= f"{full_exp_name},{accuracies[0]},{accuracies[1]},{accuracies[2]},{total}"
    print(response)    
