import pytest
import numpy as np
import pandas as pd
from algorithms.tfidf import TFIDFEngine
from algorithms.cosine_similarity import (
    compute_cosine_similarity,
    rank_destinations,
    find_matching_keywords
)

@pytest.fixture
def sample_corpus():
    return [
        "manali snow mountain skiing himachal cold adventure",
        "goa beach watersports party sun sand sea ocean",
        "rishikesh yoga spiritual ganges rafting river ashram",
        "jaipur royal fort palace heritage rajasthan pink city"
    ]

def test_tfidf_engine_fit_transform(sample_corpus):
    engine = TFIDFEngine(sample_corpus)
    assert engine.tfidf_matrix is not None
    assert engine.tfidf_matrix.shape[0] == 4
    assert len(engine.feature_names) > 0
    assert "snow" in engine.vocabulary
    assert "beach" in engine.vocabulary

def test_query_vector_transformation(sample_corpus):
    engine = TFIDFEngine(sample_corpus)
    q_vec = engine.transform_query("snow mountain adventure")
    assert q_vec.shape == (1, len(engine.feature_names))
    assert np.linalg.norm(q_vec) > 0

def test_empty_query_vector(sample_corpus):
    engine = TFIDFEngine(sample_corpus)
    q_vec = engine.transform_query("")
    assert q_vec.shape == (1, len(engine.feature_names))
    assert np.all(q_vec == 0)

def test_cosine_similarity_ranking(sample_corpus):
    engine = TFIDFEngine(sample_corpus)
    q_vec = engine.transform_query("snow adventure")
    scores = compute_cosine_similarity(q_vec, engine.tfidf_matrix)
    
    assert len(scores) == 4
    # Document 0 (Manali) should have the highest score
    assert np.argmax(scores) == 0
    assert scores[0] > 0.3
    # Document 1 (Goa beach) should have near zero score
    assert scores[1] < 0.05

def test_rank_destinations_dataframe(sample_corpus):
    engine = TFIDFEngine(sample_corpus)
    df = pd.DataFrame([
        {"destination_id": "D1", "name": "Manali", "original_idx": 0},
        {"destination_id": "D2", "name": "Goa", "original_idx": 1},
        {"destination_id": "D3", "name": "Rishikesh", "original_idx": 2},
        {"destination_id": "D4", "name": "Jaipur", "original_idx": 3},
    ])
    
    q_vec = engine.transform_query("beach party")
    ranked = rank_destinations(
        df=df,
        query_vector=q_vec,
        tfidf_matrix=engine.tfidf_matrix,
        feature_names=engine.feature_names,
        query_text="beach party"
    )
    
    assert ranked.iloc[0]["name"] == "Goa"
    assert ranked.iloc[0]["similarity_score"] > 0
    assert ranked.iloc[0]["rank"] == 1
