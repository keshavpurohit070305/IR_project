# INFORMATION RETRIEVAL VIVA EXAMINATION PREPARATION
## Comprehensive Q&A Guide for Practicals 6 & 7

This document contains 25 high-yield viva examination questions and model academic answers covering the **Vector Space Model (Practical 6)** and **Rocchio Relevance Feedback (Practical 7)**.

---

### PART 1: VECTOR SPACE MODEL & TF-IDF (PRACTICAL 6)

#### Q1: What is the Vector Space Model (VSM)?
**Answer:** The Vector Space Model, formulated by Gerard Salton, is an algebraic retrieval model that represents both text documents and user queries as multi-dimensional vectors in a Euclidean space. The vocabulary terms of the collection serve as the orthogonal basis dimensions of the space.

#### Q2: What are the primary limitations of the Boolean Retrieval Model that VSM solves?
**Answer:** 
1. **No Ranked Retrieval:** Boolean retrieval is binary (a document either matches or does not); it cannot rank documents by degree of relevance.
2. **Rigid Matching:** It suffers from either "too few results" (when too many AND operators are used) or "too many results" (when OR operators are used).
3. **No Term Weighting:** Boolean models treat all matching terms with equal importance. VSM assigns continuous weights based on term significance.

#### Q3: What is Term Frequency (TF) and why can raw frequency be misleading?
**Answer:** Term Frequency measures how often a term appears in a document. Raw frequency ($f_{t,d}$) is misleading because a document that mentions a word 10 times is rarely 10 times more relevant than a document mentioning it once. To counteract this, sublinear scaling ($1 + \ln(f_{t,d})$) or length normalization ($\frac{f_{t,d}}{|d|}$) is used.

#### Q4: What is Inverse Document Frequency (IDF) and what is its mathematical intuition?
**Answer:** IDF measures the informative value of a term across the corpus:
$$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
Words that appear in almost every document (e.g. *place, trip, good*) provide little discriminative power ($\text{DF} \approx N \implies \text{IDF} \approx 0$). Rare words (e.g. *scuba, monastery, backwaters*) have low DF, yielding high IDF values that amplify their importance in scoring.

#### Q5: Why is Cosine Similarity preferred over Euclidean Distance in text retrieval?
**Answer:** Euclidean distance is strongly biased by document length. Two documents about the exact same subject—one 100 words long and the other 10,000 words long—will have vectors pointing in the exact same direction but with vastly different magnitudes, resulting in a large Euclidean distance. Cosine similarity calculates the cosine of the angle between vectors, intrinsically normalizing for document length:
$$\text{Cosine}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

#### Q6: What does a Cosine Similarity score of 0, 0.5, and 1.0 indicate?
**Answer:**
- **1.0:** The vectors are collinear (point in the identical direction in term space).
- **0.5:** Partial angle overlap; moderate semantic/lexical similarity.
- **0.0:** Orthogonal vectors; zero common vocabulary terms between query and document.

#### Q7: Why do we perform Stemming in the preprocessing pipeline?
**Answer:** Stemming (such as the Porter Stemmer) reduces morphological variants of words to their common root or stem (e.g., *trekking, trekked, treks* $\rightarrow$ *trek*). This consolidates vocabulary dimensions, reduces the curse of dimensionality, and prevents vocabulary mismatch between user queries and documents.

#### Q8: What is the difference between Stemming and Lemmatization?
**Answer:**
- **Stemming:** Heuristic rule-based suffix chopping. It is fast and self-contained but may produce non-dictionary tokens (e.g., *destinations* $\rightarrow$ *destin*).
- **Lemmatization:** Uses morphological analysis and vocabulary lookup to return the canonical dictionary lemma (e.g., *better* $\rightarrow$ *good*).

---

### PART 2: ROCCHIO ALGORITHM & RELEVANCE FEEDBACK (PRACTICAL 7)

#### Q9: What is the core objective of the Rocchio Algorithm?
**Answer:** The Rocchio algorithm reformulates the user's query vector by moving it closer to the centroid of documents marked relevant, and further away from the centroid of documents marked non-relevant, thereby maximizing the separation between relevant and non-relevant document clusters.

#### Q10: State the Rocchio formula and explain each variable.
**Answer:**
$$\vec{q}_{new} = \alpha \vec{q}_{original} + \frac{\beta}{|D_R|} \sum_{d \in D_R} \vec{d} - \frac{\gamma}{|D_{NR}|} \sum_{d \in D_{NR}} \vec{d}$$
- $\vec{q}_{original}$: Initial user query vector.
- $D_R$: Set of relevant document vectors.
- $D_{NR}$: Set of non-relevant document vectors.
- $\alpha$: Weight on original query (preserves initial user intent).
- $\beta$: Weight on positive feedback centroid.
- $\gamma$: Weight on negative feedback centroid.

#### Q11: Why is $\gamma$ usually assigned a smaller value than $\beta$ (e.g., $\beta=0.75, \gamma=0.25$)?
**Answer:** Positive evidence is focused and consistent (relevant documents generally share a coherent topic). In contrast, negative evidence is diffuse and noisy (documents can be non-relevant for hundreds of unrelated reasons). Giving high weight to negative feedback risks inadvertently penalizing valid terms that appear incidentally in non-relevant documents.

#### Q12: Why must negative components in the updated Rocchio vector be clamped to zero?
**Answer:** In the classical Vector Space Model, term occurrences and weights cannot be negative. Allowing negative weights would distort vector projections, violate the metric properties of cosine similarity, and produce invalid negative similarity scores.

#### Q13: How does Rocchio accomplish "Query Expansion"?
**Answer:** In the calculation $\vec{q}_{new} = \alpha \vec{q}_{orig} + \beta \vec{\mu}_R - \gamma \vec{\mu}_{NR}$, terms that had weight $0.0$ in $\vec{q}_{orig}$ but have non-zero weight in the relevant centroid $\vec{\mu}_R$ receive positive weights in $\vec{q}_{new}$. This automatically adds new, highly discriminative keywords to the query without requiring the user to retype anything.

#### Q14: What is Pseudo-Relevance Feedback (Blind Relevance Feedback)?
**Answer:** In Pseudo-Relevance Feedback, the system assumes that the top-$k$ retrieved documents from the initial search are relevant ($D_R = \{d_1, \dots, d_k\}$), without soliciting human interaction. It computes Rocchio feedback automatically to expand the query.

#### Q15: What is the risk of Pseudo-Relevance Feedback compared to True Relevance Feedback?
**Answer:** **Query Drift.** If the initial top-$k$ documents happen to be non-relevant, the expanded query will drift further away from the user's actual information need, worsening search quality.

#### Q16: How is the centroid of a document set calculated?
**Answer:** The centroid $\vec{\mu}$ is the arithmetic mean vector:
$$\vec{\mu} = \frac{1}{|D|} \sum_{d \in D} \vec{d}$$
Each dimension $i$ in $\vec{\mu}$ is the average weight of term $i$ across all documents in set $D$.

---

### PART 3: ARCHITECTURE, EVALUATION & SYSTEM DESIGN

#### Q17: What evaluation metrics are standard for Information Retrieval systems?
**Answer:**
1. **Precision:** $\frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Retrieved}|}$
2. **Recall:** $\frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Total Relevant in Corpus}|}$
3. **F1-Score:** Harmonic mean of Precision and Recall.
4. **Mean Average Precision (MAP):** Average precision across multiple recall levels.
5. **Normalized Discounted Cumulative Gain (NDCG):** Measures ranking quality when documents have graded relevance levels.

#### Q18: What is the time complexity of Cosine Similarity search over $N$ documents?
**Answer:** For a query vector with $|q|$ non-zero terms and $N$ documents in an inverted index, computing cosine similarity takes $\mathcal{O}\left(\sum_{t \in q} \text{DF}(t)\right)$. In matrix form without indexing, it is $\mathcal{O}(N \cdot V)$, where $V$ is vocabulary size.

#### Q19: Why did you implement this project without external paid APIs?
**Answer:** To guarantee academic rigor, full transparency of the underlying linear algebra, 100% offline reproducibility, zero cost for educational institutions, and deterministic behavior during live laboratory assessments.

#### Q20: How does your application handle an empty or stopword-only query?
**Answer:** If a user submits an empty string or words that are all stripped by the stopword filter, the preprocessing pipeline returns an empty token list. The engine generates a zero-vector $\vec{0}$ of shape $(1, V)$. The similarity computation detects $\|\vec{q}\| = 0$ and safely returns zero scores across all documents, displaying a helpful prompt instead of crashing or showing random results.

#### Q21: What is the curse of dimensionality in the context of the Vector Space Model?
**Answer:** As the corpus grows, the vocabulary size $V$ grows to tens of thousands of terms. Documents become extremely sparse vectors (mostly containing zeros). In extremely high dimensions, distance metrics can degenerate. Dimensionality reduction techniques like Latent Semantic Analysis (LSA/SVD) or stopword filtering mitigate this.

#### Q22: Can Rocchio handle multiple iterations of feedback?
**Answer:** Yes. The formula can be applied recursively:
$$\vec{q}^{(k+1)} = \alpha \vec{q}^{(k)} + \beta \vec{\mu}_R^{(k)} - \gamma \vec{\mu}_{NR}^{(k)}$$
In WanderWise, each time the user adjusts likes or dislikes, the updated vector is computed dynamically against the initial vector or current session state.

#### Q23: How does WanderWise prevent conflicting feedback?
**Answer:** In Streamlit session state, the application enforces mutual exclusivity: if a destination is marked as Relevant (👍), it is automatically removed from the Non-Relevant set (👎), and vice versa. Duplicate additions are prevented via Python `set` data structures.

#### Q24: What is the difference between sublinear TF and standard TF?
**Answer:** Standard TF uses raw count $f_{t, d}$. Sublinear TF replaces raw frequency with $1 + \log(f_{t, d})$ for $f_{t, d} > 0$. It dampens the influence of extremely frequent occurrences of a term within a single document.

#### Q25: Why is Streamlit Community Cloud chosen for deployment?
**Answer:** Streamlit Community Cloud natively integrates with GitHub repositories, continuously deploys from the `main` branch, provides free hosting for open-source Python data applications, and requires only a root-level `app.py` and `requirements.txt`.
