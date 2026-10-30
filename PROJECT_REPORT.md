# ACADEMIC PROJECT REPORT
## WanderWise: AI-Powered Travel Destination Discovery Engine
### Information Retrieval Laboratory Project – Practicals 6 & 7

---

### Student Details
- **Student Name:** Keshav Purohit
- **GitHub Repository:** [keshavpurohit070305/IR_project](https://github.com/keshavpurohit070305/IR_project)
- **Course Title:** Information Retrieval (IR)
- **Academic Year:** 2026

---

## 1. ABSTRACT
Information Retrieval (IR) systems bridge the gap between unstructured human intent and large knowledge repositories. Traditional keyword matching frequently fails when users express subjective preferences, emotions, or complex travel goals (e.g. *"peaceful mountain retreats with scenic trekking and cafes"*). 

This project presents **WanderWise**, an interactive Information Retrieval web application designed to demonstrate two foundational IR concepts:
1. **Practical 6:** The **Vector Space Model (VSM)** with Term Frequency (TF), Inverse Document Frequency (IDF), TF-IDF weighting, and Cosine Similarity ranking.
2. **Practical 7:** **Relevance Feedback and Query Expansion** utilizing the mathematical **Rocchio Algorithm**.

Built on a curated dataset of 66 diverse Indian destinations, WanderWise provides real-time ranking, visual query-shift explanations, and interactive parameter tuning ($\alpha, \beta, \gamma$), offering an academic tool for laboratory evaluation and viva defense.

---

## 2. PROBLEM STATEMENT & OBJECTIVES
Standard search engines often rely either on rigid boolean filtering or opaque deep learning representations. For academic laboratory study, students must understand:
- How text documents are mapped into high-dimensional vector spaces.
- How term importance is modulated by document frequency.
- How user relevance feedback modifies the retrieval vector trajectory.

### Specific Objectives:
1. Construct an end-to-end preprocessing pipeline including tokenization, stopword removal, and the Porter Stemming algorithm.
2. Formulate the document-term matrix using Scikit-Learn `TfidfVectorizer` and compute true cosine similarity against user queries.
3. Formulate the Rocchio relevance feedback equation:
   $$\vec{q}_{new} = \alpha \vec{q}_{original} + \frac{\beta}{|D_R|} \sum_{d \in D_R} \vec{d} - \frac{\gamma}{|D_{NR}|} \sum_{d \in D_{NR}} \vec{d}$$
4. Implement genuine positive and negative centroid calculation and track rank promotion/demotion.
5. Create an intuitive, modern Streamlit user interface without paid external APIs.

---

## 3. SYSTEM ARCHITECTURE & WORKFLOW

```
+---------------------------------------------------------+
|                    USER INTERACTION                     |
|  Query: "peaceful hill station trekking" / Feedback     |
+---------------------------------------------------------+
                           |
                           v
+---------------------------------------------------------+
|                  PREPROCESSING PIPELINE                 |
|  Lowercasing -> Regex Clean -> Tokenize -> Stopwords    |
|  -> Porter Stemming -> Normalized String                |
+---------------------------------------------------------+
                           |
                           v
+---------------------------------------------------------+
|                 VECTOR SPACE ENGINE (P6)                |
|  Query Vector q in R^V via TF-IDF Weighting             |
|  Cosine Similarity: Sim(q, d) = (q . d) / (|q| |d|)     |
|  Initial Ranking: Rank 1 ... Rank N                     |
+---------------------------------------------------------+
                           |
            +--------------+--------------+
            | (Optional Relevance Feedback)|
            v                             v
+-----------------------+     +--------------------------+
|  Relevant Docs (D_R)  |     | Non-Relevant Docs (D_NR) |
+-----------------------+     +--------------------------+
            \                             /
             \                           /
              v                         v
+---------------------------------------------------------+
|               ROCCHIO REFINEMENT ENGINE (P7)            |
|  Centroids: mu_R = mean(D_R), mu_NR = mean(D_NR)        |
|  q_new = alpha*q + beta*mu_R - gamma*mu_NR              |
|  q_new = max(0, q_new); q_new = q_new / ||q_new||       |
+---------------------------------------------------------+
                           |
                           v
+---------------------------------------------------------+
|               RE-RANKING & VISUALIZATION                |
|  Refined Ranks, Promotion/Demotion Deltas, Plotly Charts|
+---------------------------------------------------------+
```

---

## 4. MATHEMATICAL FORMULATION

### 4.1 Term Frequency (TF)
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

### 4.2 Inverse Document Frequency (IDF)
$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

### 4.3 TF-IDF Weight
$$w_{t, d} = \text{TF}(t, d) \times \text{IDF}(t)$$

### 4.4 Cosine Similarity
$$\text{Sim}(\vec{q}, \vec{d}) = \frac{\sum_{i=1}^V q_i d_i}{\sqrt{\sum_{i=1}^V q_i^2} \sqrt{\sum_{i=1}^V d_i^2}}$$

### 4.5 Rocchio Formulation
$$\vec{q}_{new} = \alpha \vec{q}_{original} + \frac{\beta}{|D_R|} \sum_{d \in D_R} \vec{d} - \frac{\gamma}{|D_{NR}|} \sum_{d \in D_{NR}} \vec{d}$$

---

## 5. EXPERIMENTAL RESULTS & VALIDATION

### Test Scenario: Ambiguous Query Disambiguation
- **Initial Query:** *"water places vacation"*
- **Initial Top Results:** 
  1. Manali (matches 'places', 'vacation') – 28.4%
  2. Goa (matches 'water', 'vacation') – 27.9%
  3. Alleppey (matches 'water') – 25.1%
- **User Feedback:**
  - **Liked (Relevant):** Goa (Beach), Gokarna (Beach)
  - **Disliked (Non-Relevant):** Manali (Mountain Snow)
- **Rocchio Hyperparameters:** $\alpha = 1.0, \beta = 0.75, \gamma = 0.25$
- **Observed Result:**
  - $\vec{q}_{new}$ was expanded with beach-related vocabulary: `beach`, `sea`, `ocean`, `sand`, `shack`.
  - Mountain terms (`snow`, `skiing`) were suppressed.
  - **Refined Top Results:**
    1. Goa (Promoted to #1, Match increased to 64.2% ↑)
    2. Gokarna (Promoted to #2, Match increased to 58.7% ↑)
    3. Varkala Beach (Promoted from #14 to #3 ↑)
    4. Manali (Demoted to #42 ↓)

---

## 6. CONCLUSION
WanderWise demonstrates that the classical Vector Space Model and Rocchio algorithm remain powerful, interpretable, and computationally efficient foundations for Information Retrieval. The application succeeds in translating academic algorithms into a deployable software product.
