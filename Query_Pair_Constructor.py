import pandas as pd
import numpy as np
import json
import argparse
from utils import  labeling_doc_QueryMethod, labeling_doc_QrelMethod
import nltk

###########################################################################

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description= " generate data labels for easy and hard queries")
    parser.add_argument("--input", type=str, default="DeepCT2/data/diamond.tsv", help="input path")
    parser.add_argument("--datapath", type=str, default="/home/abbas/weight-difficulty/DeepCT2/data/diamond.tsv", help="input path of data")
    parser.add_argument("--alpha", type=float, default=0.15, help="lower band difficulty treshold")
    parser.add_argument("--output", type=str, default="DeepCT2/data/", help="output path")
    parser.add_argument("--method", type=str, default="", help="output path")

    args = parser.parse_args()
    
    print("first part")
    # all_query_path = "/home/abbas/abbas_qpp/deepct_files/data_train/diamond.tsv"
    all_query = pd.read_csv(args.input, sep='\t', names=['qid', 'q_orig','MRR10_orig', 'q_var','MRR10_var'])
    # all_query = pd.read_csv(args.input, sep='\t', names=col_names)

    data_1_0_msmarco = all_query.loc[(all_query["MRR10_orig"] <= args.alpha)]

    doc_Diamond_easy = pd.read_csv(args.datapath + "/docs_easy.training.small.tsv", sep='\t', names=['qid', 'doc_text'])
    doc_Diamond_hard = pd.read_csv(args.datapath + "/docs_hard.training.small.tsv", sep='\t', names=['qid', 'doc_text'])


    print("second part")
    qid = data_1_0_msmarco.iloc[0]["qid"]
    q_easy = data_1_0_msmarco.iloc[0]["q_var"]
    q_hard = data_1_0_msmarco.iloc[0]["q_orig"]

    doc_easy = doc_Diamond_easy.loc[doc_Diamond_easy['qid'] == qid, "doc_text"].iloc[0]
    doc_hard = doc_Diamond_hard.loc[doc_Diamond_hard['qid'] == qid, "doc_text"].iloc[0]


    print("third part")
    data_lines_easy = []
    data_lines_hard = []

    i = 0
    hist = []
    for index in range(len(data_1_0_msmarco)):
        try:
            qid = data_1_0_msmarco.iloc[index]["qid"]
            q_easy = data_1_0_msmarco.iloc[index]["q_var"]
            q_hard = data_1_0_msmarco.iloc[index]["q_orig"]

            doc_easy = doc_Diamond_easy.loc[doc_Diamond_easy['qid'] == qid, "doc_text"].iloc[0]
            doc_hard = doc_Diamond_hard.loc[doc_Diamond_hard['qid'] == qid, "doc_text"].iloc[0]
            hist.append(len(doc_easy.split(" ")))
            hist.append(len(doc_hard.split(" ")))
            if args.method == "QueryMethod":
                labels_doc_easy, labels_doc_hard = labeling_doc_QueryMethod(doc_easy, doc_hard, q_easy, q_hard)
            elif args.method == "QrelMethod":
                labels_doc_easy, labels_doc_hard = labeling_doc_QrelMethod(doc_easy, doc_hard, q_easy, q_hard)
            # else:
            #     labels_doc_easy, labels_doc_hard = labeling_doc(doc_easy, doc_hard, q_easy, q_hard)

            json_data_line_easy = {"query": q_easy, "term_recall": labels_doc_easy, "doc": {"position": "1", "id": qid, "title": doc_easy}}
            json_data_line_hard = {"query": q_hard, "term_recall": labels_doc_hard, "doc": {"position": "1", "id": qid, "title": doc_hard}}

            data_lines_easy.append(json_data_line_easy)
            data_lines_hard.append(json_data_line_hard)
        except:
            pass
        i += 1
        if i%10000 == 0:
            print(i)
    # Create a list to store data lines

    print("forth part")


    # output_file_path_easy = "/home/abbas/weight-difficulty/DeepCT2/data/" + 'doc_label_pos_0.25.json'
    # output_file_path_hard = "/home/abbas/weight-difficulty/DeepCT2/data/" + 'doc_label_neg_0.25.json'
    
    output_file_path_easy = args.output + "_pos_"+ args.method + "_" + str(args.alpha)[0:4] +".json"
    output_file_path_hard = args.output + "_neg_"+ args.method + "_" + str(args.alpha)[0:4] +".json"
    
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return obj

    with open(output_file_path_easy, 'w') as json_file:
        for line in data_lines_easy:
            json.dump(line, json_file, default=convert_to_json_serializable)
            json_file.write('\n')

    with open(output_file_path_hard, 'w') as json_file:
        for line in data_lines_hard:
            json.dump(line, json_file, default=convert_to_json_serializable)
            json_file.write('\n')