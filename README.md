# CA-QPP

# Query Performance Prediction via Query Variations

This repository contains the implementation and evaluation of our methodology for **Query Performance Prediction (QPP)** in information retrieval systems. Our approach leverages **query variations** to estimate the effectiveness of a retrieval method without reliance on relevance judgments.

---

## 📚 Methodology

### Problem Definition
The QPP task aims to estimate the retrieval effectiveness of a query \(q\) using a retrieval method \(R\), which produces a ranked list of documents \(D_q\). Our approach predicts the query performance (\( \hat{\mu}(q, C) \)) by analyzing contrasts between variations of the original query:
- **Promotive Variation**: Amplifies effective terms to simulate high performance.
- **Demotive Variation**: Amplifies ineffective terms to simulate low performance.

### Approach Highlights
1. **Classifying Query Terms**: Terms are classified as promotive, demotive, or neutral based on their impact on retrieval performance.
2. **Constructing Query Variations**: Generate promotive and demotive variations using term weighting.
3. **Performance Prediction**: A cross-encoder contrasts retrieval results from the variations to estimate query performance.

---

## 🛠️ Implementation Details

### Datasets
we train our model on:
- **MS MARCO V1 Passage Collection**

We evaluate on:
- **TREC DL 2019**
- **TREC DL 2020**
- **DL-Hard**

### Models
- Query Classfication: **BERT-base-uncased**
- Performance prediction: **Cross-encoder (MiniLM, BERT, DeBERTa)**

---

## 📊 Results

### Metrics
We evaluate QPP performance using correlation metrics:
- **Pearson’s  $\rho$**
- **Spearman’s  $\rho$**
- **Kendall’s  $\tau$**

Results will be shown on the experiment section of the paper.

---

## 💂️ Repository Structure

- `QueryClassification/data/`: Datasets and query pairs.
- `QueryClassification/output/`: models will be saved here.
- `QueryClassification/predictions/`: Datasets and query samples.

- `PerformancePrediction/pklfiles/`: Classified query sets will be stored here.
- `PerformancePrediction/models/`: trained Cross encoders will be saved here.
- `PerformancePrediction/results/`: Retrieval effectiveness values are stored here.

---

## 🚀 Query Term Classification

### Requirements

Install dependencies for the query classification section:
```bash
pip install -r requirements_QC.txt
```



### Constructing Query pairs

To create the training dataset for our model, we require pairs of queries alongside their first retrieved documents to assign labels to terms based on their retrieval effectiveness. The definition of promotive and demotive queries is determined by their performance metrics, such as MAP (Mean Average Precision). 


To compute term weight labels, employ the ` doc_train_term_label_generator.py ` script. Upon execution, each term in the documents will be assigned a label, with a default of zero if not applicable. The output files serves as the training data for the initial phase, where we classify terms. 


<hr>

```bash


export label_mthod=QrelMethod # QueryMethod or QrelMethod

python doc_train_term_label_generator.py --input QueryClassification/data/diamond.tsv \
    --alpha 0.25 \
    --datapath QueryClassification/data/ \
    --output QueryClassification/data/doc_label \
    --method $label_mthod

```

### Query Term Classification

Code: we adapted [DeepCT](https://github.com/AdeDZY/DeepCT) methodology, However we transfer code into tensorflow 2 so it can be used with updated gpus. modified code can be found in the ` QueryClassification ` folder.

for training run this code.
```bash 

python deepct_train_doc.py --data_dir_train QueryClassification/data/$DOC_PATH_DATA \
    --label_method $label_mthod \
    --max_seq_length_train 95 \
    --train_batch_size 16 \
    --num_train_epochs 12 \
    --output_dir_train  QueryClassification/outputs/MODEL_DOC$label_mthod/ \
    --gpu_device 0 \
    --alpha 0.25

```


OUTPUT_DIR: output folder for training. It will store the tokenized training file (train.tf_record) and the checkpoints (model.ckpt).

the classifed terms sets will be acquired by this code for both query and document.

```bash


python deepct_test_doc.py  --output_dir_train QueryClassification/outputs/MODEL_DOC$label_mthod/ \
    --predictions_dir_test  QueryClassification/predictions/MODEL_DOC$label_mthod/ \
    --max_seq_length_test 128 \
    --train_batch_size 16 \
    --num_train_epochs 12 \
    --gpu_device 0

python deepct_test.py --output_dir_train QueryClassification/outputs/MODEL_DOC$label_mthod/ \
    --predictions_dir_test QueryClassification/predictions/Query_MODEL_DOC$label_mthod/ \
    --max_seq_length_test 15 \
    --train_batch_size 16 \
    --num_train_epochs 8 \
    --gpu_device 0  


```

TEST_DATA_FILE: a tsv file ` (docid \t doc_content) ` of the entire collection that you want ot compute weight. Here, we use the [MS MARCO queries](https://drive.google.com/file/d/1kiwbqlwQDSzO2BZFpcNs5Bsa1RgbAoPo/view?usp=sharing) to compute thier term weights, therefore, these term weights will be considered as the input for the query performance prediction part.

$OUTPUT_DIR: output folder for testing. computed term weights will be stored here.


### Constructing Query Variations: 
Generate promotive and demotive variations using term weighting.longside the modifed query we include the performance of the query and feed it into our model.
```
python doc_weighted_term_frequency_bertqpp.py --input QueryClassification/predictions/$MODEL_DOC$label_mthod/ \
    --input_query QueryClassification/predictions/$MODEL_QUERY$label_mthod/ \
    --output $MODEL

```
---

## 🚀 Query Performance Prediction

### Requirements
Install dependencies for the performance prediction section:
```bash
pip install -r requirements_PP.txt
```

To train and test the model with your specific metric, use the `expansion_creat_train_test_pkl_files.py` script. This script facilitates learning the map@20 of BM25 retrieval on the MSMARCO training set. In our experiments, we utilized different models as cross encoder, and the trained model will be saved in the `BERTQPP/models/` directory.
```bash

export llm_model=deberta  # deberta pr bert or minilm
export MODEL=qpp_"$llm_model"_m"$m"_B"$B_SIZE"_E"$E_NUMBER

python expansion_creat_train_test_pkl_files.py --input_doc QueryClassification/predictions/$MODEL_DOC$label_mthod/ \
    --input_query QueryClassification/predictions/$MODEL_QUERY$label_mthod/ \
    --output $MODEL \
    --m 50 \
    --version V0


python trian_test_CE.py --model $MODEL \
    --epoch_num 2 \
    --batch_size 16 \
    --gpu_device 0 \
    --llm_model $llm_model 
```

 The results will be stored in the `results` directory, following the format: `QID\tPredicted_QPP_value`.

To evaluate the results, you can calculate the correlation between the actual performance of each query and the predicted QPP value.

---


## 👩‍💻 Contributors
- Abbas Saleminezhad

---

## 📧 Contact
For questions or collaborations, reach out to **[abbas.saleminezhad@gmail.com](mailto:your_email@example.com)**.
