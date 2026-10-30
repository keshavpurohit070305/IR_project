import pytest
from utils.preprocessing import (
    clean_text,
    tokenize,
    remove_stopwords,
    stem_tokens,
    preprocess_text,
    preprocess_tokens,
    PorterStemmer
)

def test_clean_text():
    raw = "Hello, World! 123 Welcome to India... @#$%"
    cleaned = clean_text(raw)
    assert cleaned == "hello world welcome to india"
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_tokenize():
    text = "peaceful mountain trekking destinations"
    tokens = tokenize(text)
    assert tokens == ["peaceful", "mountain", "trekking", "destinations"]
    assert tokenize("") == []

def test_stopwords_removal():
    tokens = ["i", "want", "peaceful", "beaches", "in", "goa"]
    filtered = remove_stopwords(tokens)
    assert "i" not in filtered
    assert "in" not in filtered
    assert "peaceful" in filtered
    assert "beaches" in filtered
    assert "goa" in filtered

def test_porter_stemmer():
    stemmer = PorterStemmer()
    assert stemmer.stem("destinations") == "destin"
    assert stemmer.stem("trekking") == "trek"
    assert stemmer.stem("beaches") == "beach"
    assert stemmer.stem("relaxing") == "relax"

def test_preprocess_pipeline_symmetry():
    doc = "I am looking for peaceful mountain trekking in Himachal!"
    query = "peaceful mountains trek"
    
    doc_proc = preprocess_text(doc)
    query_proc = preprocess_text(query)
    
    assert "peac" in doc_proc
    assert "mountain" in doc_proc
    assert "trek" in doc_proc
    
    assert "peac" in query_proc
    assert "mountain" in query_proc
    assert "trek" in query_proc
