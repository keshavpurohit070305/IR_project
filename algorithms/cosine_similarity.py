"""
WanderWise - Cosine Similarity & Ranking Engine
Information Retrieval Laboratory Project - Practical 6

Demonstrates:
- Dot product and norm calculation
- Cosine similarity between query vector and document matrix
- Ranking in descending order
- Keyword overlap extraction and match explanations
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessing import preprocess_tokens

def compute_cosine_similarity(query_vector: np.ndarray, document_matrix) -> np.ndarray:
    """
    Calculate genuine cosine similarity between a 1xD query vector and NxD document matrix.
    Uses sklearn's cosine_similarity.
    Returns 1D numpy array of similarity scores between 0.0 and 1.0.
    """
    if query_vector is None or document_matrix is None:
        return np.array([])
        
    # Ensure query vector is 2D (1, D)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
        
    # Check if query vector is all zeros (empty or unknown query)
    norm = np.linalg.norm(query_vector)
    if norm == 0:
        return np.zeros(document_matrix.shape[0])
        
    sim_scores = cosine_similarity(query_vector, document_matrix)
    return sim_scores.flatten()

def find_matching_keywords(
    query: str,
    doc_index: int,
    feature_names: np.ndarray,
    doc_vector: np.ndarray,
    top_n: int = 6
) -> list[tuple[str, float]]:
    """
    Find matching terms between the preprocessed query and the destination document,
    ranked by document TF-IDF importance.
    """
    query_tokens = set(preprocess_tokens(query))
    if not query_tokens:
        return []
        
    if hasattr(doc_vector, "toarray"):
        vec = doc_vector.toarray().flatten()
    else:
        vec = np.array(doc_vector).flatten()
        
    matches = []
    for term in query_tokens:
        indices = np.where(feature_names == term)[0]
        if len(indices) > 0:
            idx = indices[0]
            weight = float(vec[idx])
            if weight > 0:
                matches.append((term, weight))
                
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_n]

def generate_match_explanation(destination_name: str, matches: list[tuple[str, float]], score: float) -> str:
    """
    Generate an educational, human-readable explanation of why this destination matches.
    """
    if score <= 0.001 or not matches:
        return f"No direct vocabulary overlap detected for {destination_name}."
        
    term_list = [f"'{t}' (weight: {w:.2f})" for t, w in matches]
    terms_str = ", ".join(term_list[:4])
    
    if score >= 0.40:
        return f"Strong conceptual match for {destination_name}! High resonance with query terms: {terms_str}."
    elif score >= 0.20:
        return f"Moderate match for {destination_name}. Relevant overlaps identified on: {terms_str}."
    else:
        return f"Mild semantic match for {destination_name} with shared attributes: {terms_str}."

def rank_destinations(
    df: pd.DataFrame,
    query_vector: np.ndarray,
    tfidf_matrix,
    feature_names: np.ndarray,
    query_text: str = "",
    min_score_threshold: float = 0.0001
) -> pd.DataFrame:
    """
    Calculate similarity, rank destinations in descending order, and append
    IR metrics (score, percentage, matching terms, match explanation).
    """
    scores = compute_cosine_similarity(query_vector, tfidf_matrix)
    
    results = df.copy()
    results["similarity_score"] = scores
    results["similarity_percentage"] = (scores * 100).round(2)
    
    # Sort descending by genuine similarity score
    results = results.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1
    
    # Populate matching keywords and explanations
    match_keywords = []
    explanations = []
    
    for i, row in results.iterrows():
        score = row["similarity_score"]
        orig_idx = row.get("original_idx", i)
        
        # Get document vector
        if hasattr(tfidf_matrix, "__getitem__"):
            doc_vec = tfidf_matrix[orig_idx]
        else:
            doc_vec = np.zeros(len(feature_names))
            
        kw = find_matching_keywords(query_text, orig_idx, feature_names, doc_vec)
        expl = generate_match_explanation(row["name"], kw, score)
        
        match_keywords.append([k[0] for k in kw])
        explanations.append(expl)
        
    results["matching_keywords"] = match_keywords
    results["match_explanation"] = explanations
    
    return results
