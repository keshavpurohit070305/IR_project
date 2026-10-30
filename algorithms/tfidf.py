"""
WanderWise - Vector Space Model & TF-IDF Engine
Information Retrieval Laboratory Project - Practical 6

Demonstrates:
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- TF-IDF Weighting Matrix
- Vocabulary mapping
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import preprocess_text, preprocess_tokens

class TFIDFEngine:
    """
    Vector Space Model retrieval engine using Scikit-Learn's TfidfVectorizer.
    Includes educational breakdown functions for academic viva demonstration.
    """
    def __init__(self, corpus: list[str] = None):
        # Configure TfidfVectorizer with sublinear_tf=False, norm='l2', smooth_idf=True
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            norm='l2',
            smooth_idf=True,
            sublinear_tf=False
        )
        self.corpus = corpus or []
        self.tfidf_matrix = None
        self.feature_names = []
        self.vocabulary = {}
        
        if self.corpus:
            self.fit(self.corpus)

    def fit(self, corpus: list[str]):
        """
        Fit the vectorizer on the corpus and compute document vectors.
        """
        self.corpus = corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.vocabulary = self.vectorizer.vocabulary_
        return self

    def transform(self, texts: list[str]):
        """
        Transform a list of texts into TF-IDF vectors using the fitted vocabulary.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Engine is not fitted yet. Call fit() first.")
        return self.vectorizer.transform(texts)

    def transform_query(self, query: str) -> np.ndarray:
        """
        Preprocess and transform a single user query into a 2D dense numpy vector.
        """
        processed_query = preprocess_text(query)
        if not processed_query.strip():
            # Return zero vector with correct shape (1, num_features)
            num_features = len(self.feature_names)
            return np.zeros((1, num_features))
            
        sparse_vec = self.vectorizer.transform([processed_query])
        return sparse_vec.toarray()

    def get_query_terms_weights(self, query: str) -> dict[str, float]:
        """
        Get the TF-IDF weights of words present in the query.
        Useful for viva explanation and Plotly bar charts.
        """
        vec = self.transform_query(query).flatten()
        weights = {}
        for term, idx in self.vocabulary.items():
            val = vec[idx]
            if val > 0:
                weights[term] = float(val)
        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    def get_document_terms_weights(self, doc_idx: int, top_n: int = 15) -> dict[str, float]:
        """
        Get top weighted terms for a specific document index.
        """
        if self.tfidf_matrix is None or doc_idx >= self.tfidf_matrix.shape[0]:
            return {}
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = doc_vec.argsort()[::-1][:top_n]
        return {self.feature_names[i]: float(doc_vec[i]) for i in top_indices if doc_vec[i] > 0}

    def compute_step_by_step_breakdown(self, sample_docs: list[str], sample_query: str):
        """
        Educational helper for Practical 6 viva demonstration:
        Computes manual/exact TF, DF, IDF, and TF-IDF tables for small sample text.
        """
        # Tokenize and collect vocabulary from sample docs
        doc_tokens = [preprocess_tokens(doc) for doc in sample_docs]
        query_tokens = preprocess_tokens(sample_query)
        
        vocab = sorted(list(set(term for tokens in doc_tokens for term in tokens)))
        N = len(sample_docs)
        
        # Document frequencies
        df_dict = {term: sum(1 for tokens in doc_tokens if term in tokens) for term in vocab}
        
        # Smooth IDF: ln((1 + N) / (1 + DF)) + 1
        idf_dict = {term: np.log((1 + N) / (1 + df_dict[term])) + 1.0 for term in vocab}
        
        # Calculate TF matrix (Term Frequency normalized by document length)
        records = []
        for term in vocab:
            row = {"Term": term, "DF (Doc Freq)": df_dict[term], "IDF": round(idf_dict[term], 4)}
            for i, tokens in enumerate(doc_tokens):
                tf_raw = tokens.count(term)
                tf_norm = tf_raw / len(tokens) if tokens else 0
                tfidf = tf_norm * idf_dict[term]
                row[f"Doc {i+1} Raw TF"] = tf_raw
                row[f"Doc {i+1} Norm TF"] = round(tf_norm, 4)
                row[f"Doc {i+1} TF-IDF"] = round(tfidf, 4)
                
            q_tf_raw = query_tokens.count(term)
            q_tf_norm = q_tf_raw / len(query_tokens) if query_tokens else 0
            q_tfidf = q_tf_norm * idf_dict[term]
            row["Query Raw TF"] = q_tf_raw
            row["Query TF-IDF"] = round(q_tfidf, 4)
            records.append(row)
            
        return pd.DataFrame(records)
