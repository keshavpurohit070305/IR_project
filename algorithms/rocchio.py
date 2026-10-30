"""
WanderWise - Rocchio Relevance Feedback & Query Expansion Engine
Information Retrieval Laboratory Project - Practical 7

Mathematical Foundation:
q_new = alpha * q_original + beta * centroid(D_relevant) - gamma * centroid(D_irrelevant)

Where:
- centroid(D_relevant) = (1 / |D_R|) * sum(d for d in D_R)
- centroid(D_irrelevant) = (1 / |D_NR|) * sum(d for d in D_NR)
- Negative components clamped to zero: q_new = max(0, q_new)
- Vector normalized to unit length: q_new = q_new / ||q_new||
"""

import numpy as np
import pandas as pd
from algorithms.cosine_similarity import compute_cosine_similarity

class RocchioRelevanceFeedback:
    """
    Genuine Rocchio algorithm for query expansion and vector space refinement.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def calculate_centroid(self, doc_indices: list[int], tfidf_matrix) -> np.ndarray:
        """
        Calculate mathematical centroid vector (mean) of given document indices.
        Returns a 1xD numpy array.
        """
        if not doc_indices or tfidf_matrix is None:
            num_features = tfidf_matrix.shape[1] if tfidf_matrix is not None else 0
            return np.zeros((1, num_features))
            
        # Extract document vectors
        selected_vectors = []
        for idx in doc_indices:
            if idx < tfidf_matrix.shape[0]:
                vec = tfidf_matrix[idx].toarray() if hasattr(tfidf_matrix[idx], "toarray") else tfidf_matrix[idx]
                selected_vectors.append(vec.flatten())
                
        if not selected_vectors:
            return np.zeros((1, tfidf_matrix.shape[1]))
            
        centroid = np.mean(selected_vectors, axis=0, keepdims=True)
        return centroid

    def update_query_vector(
        self,
        q_original: np.ndarray,
        relevant_indices: list[int],
        irrelevant_indices: list[int],
        tfidf_matrix
    ) -> np.ndarray:
        """
        Execute Rocchio formula to calculate the updated, expanded query vector.
        Clamps negative values to zero and normalizes.
        """
        if q_original is None:
            raise ValueError("Original query vector cannot be None.")
            
        # Ensure 2D shape (1, D)
        q_orig = q_original.reshape(1, -1) if len(q_original.shape) == 1 else q_original
        num_features = q_orig.shape[1]
        
        # Calculate centroids
        rel_centroid = self.calculate_centroid(relevant_indices, tfidf_matrix)
        irrel_centroid = self.calculate_centroid(irrelevant_indices, tfidf_matrix)
        
        # Apply Rocchio Formula
        q_new = (self.alpha * q_orig) + (self.beta * rel_centroid) - (self.gamma * irrel_centroid)
        
        # Clamp negative weights to zero (Standard Vector Space Model constraint)
        q_new = np.maximum(0.0, q_new)
        
        # L2 unit normalization
        norm = np.linalg.norm(q_new)
        if norm > 0:
            q_new = q_new / norm
            
        return q_new

    def get_query_shift_terms(
        self,
        q_original: np.ndarray,
        q_new: np.ndarray,
        feature_names: np.ndarray,
        top_k: int = 10
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Identify top expanded/boosted terms and top demoted terms for visual analysis.
        """
        diff = (q_new.flatten() - q_original.flatten())
        
        # Terms boosted (positive shift)
        boosted_indices = np.argsort(diff)[::-1][:top_k]
        boosted_terms = [
            (feature_names[i], float(diff[i])) 
            for i in boosted_indices if diff[i] > 0.001
        ]
        
        # Terms demoted (negative shift)
        demoted_indices = np.argsort(diff)[:top_k]
        demoted_terms = [
            (feature_names[i], float(abs(diff[i]))) 
            for i in demoted_indices if diff[i] < -0.001
        ]
        
        return boosted_terms, demoted_terms

    def compare_rankings(
        self,
        df_original_ranked: pd.DataFrame,
        q_new: np.ndarray,
        tfidf_matrix
    ) -> pd.DataFrame:
        """
        Recalculate similarity with updated query vector and produce
        a comparative ranking dataframe with promotion/demotion indicators.
        """
        # Recalculate similarity using updated query vector
        new_scores = compute_cosine_similarity(q_new, tfidf_matrix)
        
        # Construct refined dataframe
        refined_df = df_original_ranked.copy()
        refined_df["original_rank"] = refined_df["rank"]
        refined_df["original_score"] = refined_df["similarity_score"]
        refined_df["original_score_pct"] = (refined_df["similarity_score"] * 100).round(2)
        
        # Assign new scores mapped by original document index
        refined_scores = []
        for _, row in refined_df.iterrows():
            orig_idx = int(row["original_idx"])
            refined_scores.append(new_scores[orig_idx])
            
        refined_df["similarity_score"] = refined_scores
        refined_df["similarity_percentage"] = (np.array(refined_scores) * 100).round(2)
        
        # Re-sort according to refined similarity scores
        refined_df = refined_df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)
        refined_df["rank"] = refined_df.index + 1
        refined_df["refined_rank"] = refined_df["rank"]
        
        # Calculate rank movement: positive = promoted, negative = demoted
        refined_df["rank_shift"] = refined_df["original_rank"] - refined_df["refined_rank"]
        refined_df["score_change"] = (refined_df["similarity_score"] - refined_df["original_score"]).round(4)
        refined_df["score_change_pct"] = (refined_df["score_change"] * 100).round(2)
        
        # Status labels
        status_list = []
        for shift in refined_df["rank_shift"]:
            if shift > 0:
                status_list.append(f"Promoted (+{shift} ↑)")
            elif shift < 0:
                status_list.append(f"Demoted ({shift} ↓)")
            else:
                status_list.append("Unchanged (=)")
        refined_df["movement_status"] = status_list
        
        return refined_df
