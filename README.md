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
We evaluate on:
- **TREC DL 2019**
- **TREC DL 2020**
- **DL-Hard**

### Models
- Term weight regression: **BERT-base-uncased**
- Performance prediction: **Cross-encoder (MiniLM, BERT, DeBERTa)**

---

## 📊 Results

### Metrics
We evaluate QPP performance using correlation metrics:
- **Pearson’s \( \rho \)**
- **Spearman’s \( \rho \)**
- **Kendall’s \( \tau \)**

Results will be added after experiments.

---

## 💂️ Repository Structure

- `data/`: Datasets and query samples.
- `models/`: Pretrained and fine-tuned models.
- `src/`: Implementation of the methodology.
- `notebooks/`: Analysis and visualization scripts.
- `results/`: Performance evaluation outputs.

---

## 🚀 Usage

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### Training
Instructions for training the term weighting and performance prediction models will be added.

### Evaluation
Steps to evaluate the QPP approach on standard datasets will be included.

---

## 🔬 Experiments

Code for experiments and results will be added in upcoming updates.

---

## 💄 Citation
If you use this work, please cite our paper:
```
@article{your_paper_citation,
  title={Your Paper Title},
  author={Authors},
  journal={Conference/Journal},
  year={2025}
}
```

---

## 👩‍💻 Contributors
- Abbas Saleminezhad

---

## 📧 Contact
For questions or collaborations, reach out to **[abbas.saleminezhad@gmail.com](mailto:your_email@example.com)**.
