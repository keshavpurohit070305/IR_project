# 🧭 WanderWise – AI Travel Destination Discovery Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IR Lab Practicals](https://img.shields.io/badge/IR%20Lab-Practical%206%20%26%207-1b4332.svg)]()

> *"Discover places that match your kind of adventure."*

**WanderWise** is a full-featured, AI-powered travel discovery engine built using modern Information Retrieval (IR) algorithms. Designed as a college laboratory project and live viva demonstration, it features mathematical implementations of:
- **Practical 6:** Vector Space Model using Term Frequency (TF), Inverse Document Frequency (IDF), TF-IDF weighting, and Cosine Similarity ranking.
- **Practical 7:** Query Expansion and Relevance Feedback using the genuine Rocchio Algorithm.

---

## 🚀 Live Streamlit Community Cloud Deployment

WanderWise is fully optimized and configured for 1-click deployment on **Streamlit Community Cloud**.

### Exact Deployment Configuration:

| Configuration Field | Required Value |
|---|---|
| **Repository** | `keshavpurohit070305/IR_project` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

### Step-by-Step Deployment Guide:
1. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account (`keshavpurohit070305`).
2. Click **"Create app"** in the top right corner.
3. In the repository dropdown, select **`keshavpurohit070305/IR_project`**.
4. Set the **Branch** to **`main`**.
5. Set the **Main file path** to **`app.py`**.
6. Click **"Deploy!"**.
7. Streamlit Cloud will install packages from `requirements.txt` and launch the web app.

---

## 🌟 Key Application Features

- **Natural Language Discovery:** Search by conversational travel vibes, moods, or specific activities (e.g., *"I want peaceful mountain destinations with trekking"* or *"Find affordable beach destinations in India"*).
- **Genuine TF-IDF Scoring:** Computes exact TF, IDF, and Cosine Similarity scores displayed as intuitive percentage matches.
- **Interactive Rocchio Feedback Lab:** Mark destinations as **Relevant (👍)** or **Non-Relevant (👎)**, tune hyperparameters ($\alpha, \beta, \gamma$), and observe how the query vector expands and re-ranks destinations in real-time.
- **Interactive Visualizations (Plotly):**
  - Query vector TF-IDF weight breakdown.
  - Rocchio vocabulary expansion (boosted terms vs. suppressed terms).
  - Slope/bump rank movement chart comparing initial vs refined positions.
  - Grouped cosine similarity shift charts.
- **Curated 66 Indian Destinations:** Spanning Himachal Pradesh, Uttarakhand, Ladakh, Goa, Kerala, Karnataka, Rajasthan, Maharashtra, Meghalaya, Sikkim, Arunachal Pradesh, Tamil Nadu, Madhya Pradesh, and Andaman Islands.
- **Multi-Faceted Filtering:** Refine by Geographic Region, Travel Type (Adventure, Beach, Heritage, Nature, Relaxation, Spiritual, Wildlife, Hill Station), Budget Category, Daily Budget range slider, and Best Season.
- **My Discoveries:** Personalized bucket list calculating combined estimated daily budgets and itineraries.
- **Dedicated Viva Section:** Interactive step-by-step matrix calculation and 25+ viva questions and answers.

---

## 📂 Project Architecture & File Hierarchy

```
IR_project/
├── app.py                      # Main Streamlit web application entrypoint (Root-level)
├── requirements.txt            # Dependency specification for Streamlit Cloud
├── README.md                   # Comprehensive project documentation
├── PROJECT_REPORT.md           # Formal academic project report
├── VIVA_QUESTIONS.md           # 25+ Viva Q&A master checklist
├── .gitignore                  # Git ignore rules
│
├── .streamlit/
│   └── config.toml             # Custom Streamlit UI color theme & server configuration
│
├── data/
│   └── destinations.csv        # Curated dataset of 66 Indian travel destinations
│
├── algorithms/
│   ├── __init__.py
│   ├── tfidf.py                # TF-IDF Vector Space Model engine
│   ├── cosine_similarity.py   # Cosine similarity calculation & ranking
│   └── rocchio.py              # Genuine Rocchio relevance feedback & query expansion
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Text cleaning, tokenization, stopwords, Porter Stemmer
│   └── helpers.py              # Data loaders, searchable doc builder, filters, formatting
│
├── components/
│   ├── __init__.py
│   ├── destination_cards.py    # Responsive destination cards & Like/Dislike triggers
│   ├── search_interface.py     # Natural language search & sidebar filter controls
│   └── visualizations.py       # Plotly IR charts (weights, shifts, rank comparison)
│
├── assets/
│   └── styles.css              # Custom CSS styling (Airbnb / Wanderlog aesthetic)
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Tests for tokenization, stopwords, stemming
    ├── test_tfidf.py           # Tests for TF-IDF vectors & cosine similarity
    └── test_rocchio.py         # Tests for Rocchio centroid, formula & rank updates
```

---

## 📐 Mathematical Formulation

### 1. Vector Space Model & TF-IDF (Practical 6)

Every destination document $d_j$ and user query $q$ is transformed into a vector in a $V$-dimensional space:

$$\vec{d}_j = (w_{1, j}, w_{2, j}, \dots, w_{V, j})$$

Where each term weight $w_{i, j}$ is computed as:

$$w_{i, j} = \text{TF}(t_i, d_j) \times \text{IDF}(t_i)$$

- **Term Frequency:** $\text{TF}(t_i, d_j) = \frac{f_{t_i, d_j}}{\sum_{t'} f_{t', d_j}}$
- **Inverse Document Frequency:** $\text{IDF}(t_i) = \ln\left(\frac{N + 1}{\text{DF}(t_i) + 1}\right) + 1$

The similarity between query vector $\vec{q}$ and document vector $\vec{d}_j$ is calculated via **Cosine Similarity**:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}_j) = \frac{\vec{q} \cdot \vec{d}_j}{\|\vec{q}\| \|\vec{d}_j\|} = \frac{\sum_{k=1}^V q_k d_{jk}}{\sqrt{\sum_{k=1}^V q_k^2} \sqrt{\sum_{k=1}^V d_{jk}^2}}$$

### 2. Rocchio Relevance Feedback (Practical 7)

The query vector is updated mathematically based on user feedback:

$$\vec{q}_{new} = \alpha \vec{q}_{original} + \frac{\beta}{|D_R|} \sum_{d \in D_R} \vec{d} - \frac{\gamma}{|D_{NR}|} \sum_{d \in D_{NR}} \vec{d}$$

- $D_R$: Set of documents marked **Relevant (Liked)**.
- $D_{NR}$: Set of documents marked **Non-Relevant (Disliked)**.
- $\alpha = 1.0$: Weight assigned to the original user query.
- $\beta = 0.75$: Weight pulling the query toward the relevant document centroid.
- $\gamma = 0.25$: Weight pushing the query away from the non-relevant centroid.
- **Non-negative Constraint:** $\vec{q}_{new} = \max(0, \vec{q}_{new})$ ensures term weights remain non-negative in the Vector Space Model.
- **Unit Normalization:** $\vec{q}_{new} = \frac{\vec{q}_{new}}{\|\vec{q}_{new}\|}$.

---

## 💻 Local Setup & Execution Guide (Windows / Mac / Linux)

### Prerequisites:
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone the repository:
```powershell
git clone https://github.com/keshavpurohit070305/IR_project.git
cd IR_project
```

### 2. Create and activate a virtual environment (optional but recommended):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install required dependencies:
```powershell
pip install -r requirements.txt
```

### 4. Run the automated Pytest test suite:
```powershell
pytest
```
*Expected output: `13 passed in ~6s`.*

### 5. Launch the Streamlit application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing Coverage

The project includes unit tests for all core algorithms:
- `test_clean_text`, `test_tokenize`, `test_stopwords_removal`, `test_porter_stemmer`, `test_preprocess_pipeline_symmetry`.
- `test_tfidf_engine_fit_transform`, `test_query_vector_transformation`, `test_empty_query_vector`, `test_cosine_similarity_ranking`.
- `test_rocchio_centroid`, `test_rocchio_query_vector_update`, `test_rocchio_rank_comparison`.

Run tests with:
```powershell
pytest -v
```

---

## 🎓 Academic Practical Mapping

| Practical | Topic | Implementation File | Verification Test |
|---|---|---|---|
| **Practical 6** | Vector Space Model (TF, IDF, TF-IDF, Cosine Similarity) | `algorithms/tfidf.py`, `algorithms/cosine_similarity.py` | `tests/test_tfidf.py` |
| **Practical 7** | Rocchio Relevance Feedback & Query Expansion | `algorithms/rocchio.py` | `tests/test_rocchio.py` |

---

## 👥 Authors & Academic Submission

- **Student Name:** Keshav Purohit
- **GitHub:** [@keshavpurohit070305](https://github.com/keshavpurohit070305)
- **Course:** Information Retrieval Laboratory
- **Project Title:** WanderWise – AI Travel Destination Discovery Engine
