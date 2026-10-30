import pytest
import numpy as np
import pandas as pd
from algorithms.tfidf import TFIDFEngine
from algorithms.cosine_similarity import compute_cosine_similarity
from algorithms.rocchio import RocchioRelevanceFeedback

@pytest.fixture
def corpus_and_engine():
    corpus = [
        "manali snow mountain skiing winter himachal cold",
        "shimla colonial hill station cool family mall road",
        "goa beach watersports party sun sand sea ocean",
        "gokarna peaceful beach ocean trek cliffs quiet"
    ]
    engine = TFIDFEngine(corpus)
    return corpus, engine

def test_rocchio_centroid(corpus_and_engine):
    _, engine = corpus_and_engine
    rocchio = RocchioRelevanceFeedback()
    
    centroid = rocchio.calculate_centroid([0, 1], engine.tfidf_matrix)
    assert centroid.shape == (1, len(engine.feature_names))
    assert np.linalg.norm(centroid) > 0
    
    empty_centroid = rocchio.calculate_centroid([], engine.tfidf_matrix)
    assert np.all(empty_centroid == 0)

def test_rocchio_query_vector_update(corpus_and_engine):
    _, engine = corpus_and_engine
    rocchio = RocchioRelevanceFeedback(alpha=1.0, beta=0.75, gamma=0.25)
    
    # Query with terms touching both mountain ("winter") and beach ("beach")
    q_orig = engine.transform_query("winter beach")
    
    # User likes Snow/Mountain (Doc 0), dislikes Beach (Doc 2)
    q_new = rocchio.update_query_vector(
        q_original=q_orig,
        relevant_indices=[0],
        irrelevant_indices=[2],
        tfidf_matrix=engine.tfidf_matrix
    )
    
    assert q_new.shape == q_orig.shape
    # No negative values allowed
    assert np.all(q_new >= 0.0)
    # Unit normalized
    assert np.isclose(np.linalg.norm(q_new), 1.0)
    
    # Check similarity movement
    sim_orig = compute_cosine_similarity(q_orig, engine.tfidf_matrix)
    sim_new = compute_cosine_similarity(q_new, engine.tfidf_matrix)
    
    # Relevant Doc 0 (Manali) score increases
    assert sim_new[0] > sim_orig[0]
    # Irrelevant Doc 2 (Goa) score decreases
    assert sim_new[2] < sim_orig[2]

def test_rocchio_rank_comparison(corpus_and_engine):
    _, engine = corpus_and_engine
    rocchio = RocchioRelevanceFeedback()
    
    df = pd.DataFrame([
        {"destination_id": "D1", "name": "Manali", "original_idx": 0, "rank": 1, "similarity_score": 0.4},
        {"destination_id": "D2", "name": "Goa", "original_idx": 2, "rank": 2, "similarity_score": 0.35},
        {"destination_id": "D3", "name": "Shimla", "original_idx": 1, "rank": 3, "similarity_score": 0.3},
        {"destination_id": "D4", "name": "Gokarna", "original_idx": 3, "rank": 4, "similarity_score": 0.25},
    ])
    
    q_orig = engine.transform_query("vacation")
    # Mark Gokarna and Goa as relevant (Beaches), Manali as irrelevant
    q_new = rocchio.update_query_vector(
        q_original=q_orig,
        relevant_indices=[2, 3],
        irrelevant_indices=[0],
        tfidf_matrix=engine.tfidf_matrix
    )
    
    refined = rocchio.compare_rankings(df, q_new, engine.tfidf_matrix)
    
    assert "rank_shift" in refined.columns
    assert "movement_status" in refined.columns
    # Beach destinations should move up
    beach_rows = refined[refined["name"].isin(["Goa", "Gokarna"])]
    for _, row in beach_rows.iterrows():
        assert row["similarity_score"] > 0
