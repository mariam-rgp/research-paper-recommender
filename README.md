# Research Paper Recommendation System

A content-based research paper recommendation system that recommends relevant academic papers based on their titles and abstracts.

The project compares a traditional **TF-IDF** approach with **Transformer-based semantic embeddings** to evaluate how well each method captures similarity between research papers.

## Project Overview

Finding relevant research papers can be time-consuming when dealing with large collections of academic publications.

This project aims to build a recommendation system that retrieves papers similar to a given paper or text query.

Two approaches are implemented and compared:

1. **TF-IDF + Cosine Similarity**
2. **Transformer Embeddings + Cosine Similarity**

The system is evaluated using ranking-based information retrieval metrics, and **MLflow** is used to track and compare experiments.

## Dataset

The project uses an arXiv research paper dataset containing paper titles, abstracts, and subject terms.

After preprocessing and cleaning, the dataset contains approximately **41K research papers**.

The main text used for recommendation is constructed from the paper title and abstract.

> The original dataset is not included in the repository because of its size.

A small `test_papers.csv` file is included for evaluation and testing purposes.

## Project Pipeline

```text
Raw Research Papers
        ↓
Data Cleaning & Preprocessing
        ↓
Title + Abstract
        ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
TF-IDF              Transformer
Vectorization       Embeddings
 │                       │
 └───────────┬───────────┘
             ↓
      Cosine Similarity
             ↓
      Top-K Recommendations
             ↓
         Evaluation
             ↓
          MLflow
```

## Approaches

### 1. TF-IDF Baseline

TF-IDF is used as a traditional text representation technique.

Configuration:

* `max_features = 50,000`
* TF-IDF vectorization
* Cosine similarity for retrieving similar papers

The TF-IDF approach provides a strong baseline for comparing traditional lexical similarity with semantic embeddings.

### 2. Transformer Embeddings

Transformer-based embeddings are generated using a Sentence Transformer model.

Model:

`all-MiniLM-L6-v2`

Each research paper is represented as a **384-dimensional semantic embedding**.

Cosine similarity is then used to identify papers with similar semantic meaning.

## Evaluation

The two approaches are evaluated using ranking-based metrics:

* Precision@10
* Recall@10
* NDCG@10
* MRR@10
* MAP@10

### Results

| Metric       |   TF-IDF |  Transformer |
| ------------ | -------: | -----------: |
| Precision@10 |   0.9060 |   **0.9480** |
| Recall@10    | 0.000332 | **0.000349** |
| NDCG@10      |   0.8859 |   **0.8996** |
| MRR@10       |   0.9542 |   **0.9542** |
| MAP@10       |   0.9320 |   **0.9350** |

The results show that the Transformer-based representation achieved higher Precision@10, Recall@10, and NDCG@10 than the TF-IDF baseline on the evaluation setup used in this project.

## Experiment Tracking

**MLflow** is used to track experiments and compare the performance of the recommendation approaches.

Tracked information includes:

* Model/representation type
* Dataset size
* Evaluation metrics
* Experiment runs

The project uses MLflow to make model comparison and experiment tracking reproducible.

## Project Structure

```text
research-paper-recommender/
│
├── data/
│   └── test_papers.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tfidf_baseline.ipynb
│   ├── 03_transformer_embeddings.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_recommendation_system.ipynb
│
├── src/
│   └── mlflow_tracking.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mariam-rgp/research-paper-recommender.git
cd research-paper-recommender
```

Create and activate the Conda environment:

```bash
conda create -n paper_recommender python=3.11
conda activate paper_recommender
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in the following order:

```text
01_data_exploration.ipynb
        ↓
02_tfidf_baseline.ipynb
        ↓
03_transformer_embeddings.ipynb
        ↓
04_evaluation.ipynb
        ↓
05_recommendation_system.ipynb
```

The notebooks cover:

* Dataset exploration and preprocessing
* TF-IDF baseline
* Transformer-based embeddings
* Model evaluation
* Paper recommendation

## MLflow

To run the MLflow tracking script:

```bash
python src/mlflow_tracking.py
```

Then start the MLflow UI:

```bash
mlflow ui
```

The MLflow tracking files are kept locally and are excluded from GitHub.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Hugging Face
* Sentence Transformers
* TF-IDF
* Transformer Embeddings
* Cosine Similarity
* MLflow
* Jupyter Notebook
* Git
* GitHub

## Future Improvements

* Build a REST API for the recommendation system using FastAPI
* Add a web interface for interactive recommendations
* Experiment with additional embedding models
* Add multilingual and Arabic research paper support
* Explore hybrid recommendation approaches
* Extend the system toward LLM/RAG-based research assistance
* Deploy the recommendation service

## Author

**Mariam Ahmed**

Computer & Information Graduate | Data Science & NLP

GitHub: `https://github.com/mariam-rgp`
