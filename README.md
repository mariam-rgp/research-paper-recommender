# 📚 Research Paper Recommendation System

An end-to-end **Research Paper Recommendation System** that recommends similar research papers based on their textual content.

The project compares a traditional **TF-IDF** approach with **Transformer-based semantic embeddings**, evaluates their recommendation performance, and provides a complete application pipeline with **Translation, FastAPI, and Streamlit**.

**MLflow** is used for experiment tracking and comparison between the TF-IDF and Transformer approaches.

---

## 🚀 Project Pipeline

```text
TF-IDF
   ↓
Transformer
   ↓
Evaluation
   ↓
Recommendation
   ↓
Translation
   ↓
FastAPI
   ↓
Streamlit
   ↓
MLflow
   └── Experiment Tracking
          ├── TF-IDF
          └── Transformer
```

---

## 🎯 Project Overview

Finding relevant research papers can be difficult when dealing with a large collection of academic documents.

This project builds a **content-based recommendation system** that recommends research papers based on their textual similarity.

Two different text representation approaches are implemented and compared:

### 1. TF-IDF

A traditional lexical representation based on word importance.

### 2. Transformer Embeddings

A semantic representation generated using the pretrained:

```text
all-MiniLM-L6-v2
```

The two approaches are evaluated using ranking metrics, and the recommendation system uses similarity scores to retrieve the most relevant papers.

The final system also includes a **translation component**, a **FastAPI backend**, and a **Streamlit interface**.

---

# 🗂️ Dataset

The project uses research paper data from **arXiv Paper Abstracts**.

The dataset contains information such as:

* Paper titles
* Paper abstracts
* Terms/categories

### Data preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Removing duplicate papers
* Cleaning textual data
* Combining the paper title and abstract
* Creating text-based features
* Preparing the data for TF-IDF and Transformer-based representations

Large raw datasets and generated model artifacts are excluded from the GitHub repository.

A small `test_papers.csv` file is included for testing/evaluation purposes.

---

# 🔄 Methodology

## 1. TF-IDF Baseline

The first approach uses **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Each research paper is converted into a numerical vector based on the importance of its words across the dataset.

```text
Paper Title + Abstract
          ↓
     TF-IDF Vectorizer
          ↓
   Numerical Representation
          ↓
    Cosine Similarity
          ↓
    Similar Papers
```

The implementation uses:

```python
TfidfVectorizer(max_features=50000)
```

TF-IDF provides a traditional lexical baseline for comparison with the semantic Transformer approach.

---

# 2. Transformer Embeddings

The second approach uses Transformer-based sentence embeddings.

The project uses:

```text
all-MiniLM-L6-v2
```

Each paper is converted into a **384-dimensional embedding**.

```text
Paper Title + Abstract
          ↓
   Transformer Encoder
          ↓
  384-dimensional Vector
          ↓
    Cosine Similarity
          ↓
    Similar Papers
```

Unlike TF-IDF, Transformer embeddings represent semantic relationships between words and sentences, allowing the system to identify papers that are conceptually related even when they do not share exactly the same words.

---

# 3. Evaluation

The two recommendation approaches are evaluated using ranking metrics.

The evaluation includes:

* **Precision@10**
* **Recall@10**
* **NDCG@10**
* **MRR@10**
* **MAP@10**

### Evaluation Results

| Metric       |   TF-IDF | Transformer |
| ------------ | -------: | ----------: |
| Precision@10 |   0.9060 |      0.9480 |
| Recall@10    | 0.000332 |    0.000349 |
| NDCG@10      |   0.8859 |      0.8996 |
| MRR@10       |   0.9542 |           — |
| MAP@10       |   0.9320 |           — |

The evaluation notebook contains the detailed implementation and comparison of both approaches.

> The Transformer values for MRR@10 and MAP@10 are omitted here until their exact final experiment values are confirmed.

---

# 4. Recommendation System

After generating the text representations, the system calculates similarity between papers.

The recommendation workflow is:

```text
Input Paper / Query
        ↓
Text Representation
        ↓
Similarity Calculation
        ↓
Ranking
        ↓
Top-K Recommendations
```

**Cosine similarity** is used to measure the similarity between the input paper and the available research papers.

The system returns the most relevant papers based on their similarity scores.

---

# 5. Translation

A translation component is integrated into the project after the recommendation stage.

```text
Recommendation
      ↓
 Translation
      ↓
Translated Output
```

This allows the recommended paper information to be translated as part of the application workflow.

The translation functionality is integrated into the overall recommendation pipeline rather than being a separate standalone project.

---

# 🌐 Application Layer

## 6. FastAPI

**FastAPI** is used as the backend API layer for the application.

It connects the machine-learning/recommendation logic with the user interface.

```text
Client
  ↓
FastAPI
  ↓
Recommendation System
  ↓
Translation
  ↓
Response
```

The API provides an interface through which the recommendation functionality can be accessed programmatically.

---

# 7. Streamlit

**Streamlit** is used to build the interactive frontend.

The overall application flow is:

```text
Streamlit
    ↓
FastAPI
    ↓
Recommendation
    ↓
Translation
    ↓
Results
```

This provides a simple interface for interacting with the research paper recommendation system.

---

# 📊 MLflow

**MLflow** is used in this project for **experiment tracking**.

The main purpose of MLflow is to compare the two implemented recommendation approaches:

```text
                 MLflow
                    ↓
          Experiment Tracking
             ↙            ↘
         TF-IDF       Transformer
```

For each approach, the experiment tracks relevant parameters and evaluation metrics.

### Tracked Metrics

The comparison includes:

* Precision@10
* Recall@10
* NDCG@10
* MRR@10
* MAP@10

The MLflow experiment is:

```text
Research Paper Recommender
```

MLflow makes it easier to keep the experimental results of **TF-IDF and Transformer** organized and compare their performance.

> **Current MLflow scope:** Experiment Tracking and comparison between TF-IDF and Transformer. Model Registry, MLflow Autologging, and Hyperparameter Tuning are not part of the current implementation.

---

# 📁 Project Structure

```text
research-paper-recommender/
│
├── data/
│   ├── .gitkeep
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
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets, generated embeddings, TF-IDF matrices, and local MLflow files are excluded from version control.

---

# 🛠️ Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Sentence Transformers**
* **Hugging Face**
* **MLflow**
* **FastAPI**
* **Streamlit**
* **Git**
* **GitHub**

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mariam-rgp/research-paper-recommender.git
cd research-paper-recommender
```

Create the Conda environment:

```bash
conda create -n paper_recommender python=3.11
conda activate paper_recommender
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Project Workflow

The notebooks follow the development process:

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

The resulting recommendation functionality is then integrated with:

```text
Recommendation
      ↓
Translation
      ↓
FastAPI
      ↓
Streamlit
```

MLflow is used alongside the experimentation process to track the TF-IDF and Transformer experiments.

---

# 📈 Key Features

* 📄 Research paper recommendation
* 🔤 TF-IDF text representation
* 🤗 Transformer-based semantic embeddings
* 🔎 Cosine similarity
* 📊 Ranking-based evaluation
* 📈 Precision@K
* 📈 Recall@K
* 📈 NDCG@K
* 📈 MRR@K
* 📈 MAP@K
* 🌐 Translation support
* ⚡ FastAPI backend
* 🖥️ Streamlit frontend
* 📊 MLflow experiment tracking
* 🔬 TF-IDF vs Transformer comparison
* 🐙 Git/GitHub version control

---

# 🔮 Future Improvements

Possible future improvements include:

* Hybrid TF-IDF + Transformer recommendation
* Fine-tuning the embedding model on research-paper data
* More advanced semantic search
* Re-ranking using cross-encoder models
* Improved multilingual search
* User feedback and personalized recommendations
* Deployment of the FastAPI and Streamlit applications
* Automated testing and CI/CD
* Integration with larger-scale vector databases

---

# 👩‍💻 Author



GitHub: **mariam-rgp**
