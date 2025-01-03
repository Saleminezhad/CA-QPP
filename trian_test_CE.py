from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math
import os
import argparse
import pickle 
from scipy.stats import kendalltau,pearsonr


# set random seed and torch seed to 42
import torch
import random
import numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Train Cross Encoder Model")
    parser.add_argument("--model", type=str, default="BERTQPP/pklfiles/termfreq_015_ex/", help="input path")
    parser.add_argument("--epoch_num", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gpu_device", type=str, default="3", help="gpu device")
    parser.add_argument("--llm_model", type=str, default="deberta", help="LLM model")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_device

    
    input_path = "pklfiles/" + args.model
    q_map_dic_train_path = input_path + "/main_doc.pkl"
    with open(q_map_dic_train_path, 'rb') as f:
        q_map_dic_train=pickle.load(f)

    train_set=[]

    for key in q_map_dic_train:
        qtext=q_map_dic_train[key]["qtext_out"]
        firstdoctext=q_map_dic_train[key]["doc_text_out"]
        actual_map = q_map_dic_train[key] ["performance"]
        
        train_set.append( InputExample(texts=[qtext,firstdoctext],label=actual_map ))

    print("111")
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * args.epoch_num * 0.1) #10% of train data for warm-up
    if args.llm_model == "deberta":
        model_name='microsoft/deberta-v3-base'
        model = CrossEncoder(model_name, num_labels=1, max_length=512)
        print("222")

    elif args.llm_model == "bert":
        model_name='bert-base-uncased'
        model = CrossEncoder(model_name, num_labels=1)
    elif args.llm_model == "minilmcos":
        model_name='msmarco-MiniLM-L6-cos-v5'
        model = CrossEncoder(model_name, num_labels=1)
    elif args.llm_model == "minilmV3":
        model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'
        model = CrossEncoder(model_name, num_labels=1, max_length=512)
    else:
        print("Model not found")
        exit()

    model_path=f"models/{args.model}"
    
    # Train the model
    
    model.fit(train_dataloader=train_dataloader,
            epochs=args.epoch_num,
            warmup_steps=warmup_steps,
            output_path=model_path)
    model.save(model_path)



    #test 

    if not os.path.exists('results/'+ args.model +"/"):
        os.system("mkdir "+ 'results/'+ args.model +"/")
    print("333")

    years = ['2019', '2020' ,'hard']
    for year in years:

        with open(input_path + f"/{year}_doc.pkl", 'rb') as f:
        # with open(f"/mnt/data/abbas/BERTQPP/pklfiles/trec-dl-{year}_all-MiniLM-L6-v2_map1000.pkl", 'rb') as f:
            q_map_first_doc_test=pickle.load(f)

        sentences = []
        map_value_test=[]
        queries=[]
        for key in q_map_first_doc_test:
            sentences.append([q_map_first_doc_test[key]["qtext_out"],q_map_first_doc_test[key]["doc_text_out"]])
            # sentences.append([q_map_first_doc_test[key]["qtext"],q_map_first_doc_test[key]["doc_text"]])

            queries.append(key)
            
        print("444")
            
        if args.llm_model == "deberta":
            model = CrossEncoder(model_path, num_labels=1, max_length=512)
        elif args.llm_model == "bert":
            model = CrossEncoder(model_path, num_labels=1)
        elif args.llm_model == "minilmcos":
            model = CrossEncoder(model_path, num_labels=1)
        elif args.llm_model == "minilmV3":
            model = CrossEncoder(model_path, num_labels=1, max_length=512)
            
            
        # model = CrossEncoder("/mnt/data/abbas/BERTQPP/models/2_tuned_model-ce_microsoft/deberta-v3-base_e2_b16", num_labels=1)

        scores=model.predict(sentences)
        print("555")
        
        actual=[]
        predicted=[]
        out=open(f'results/{args.model}/{year}_doc','w')
        for i in range(len(sentences)):
            predicted.append(float(scores[i]))
            out.write(queries[i]+'\t'+str(predicted[i])+'\n')
        out.close()
