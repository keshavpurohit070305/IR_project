"""
WanderWise - Helper Utilities
Data handling, document text aggregation, filtering, and presentation helpers.
"""

import os
import pandas as pd
from utils.preprocessing import preprocess_text

def get_data_path() -> str:
    """
    Get the relative or absolute path to the destinations dataset.
    Works robustly in local dev and Streamlit Community Cloud.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "destinations.csv")

def load_destinations_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load destinations CSV into a pandas DataFrame.
    Fills missing values safely.
    """
    if csv_path is None:
        csv_path = get_data_path()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Destination dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    # Fill any null values safely
    text_cols = [
        "name", "state", "region", "short_description", "detailed_description",
        "attractions", "activities", "travel_type", "budget_category",
        "best_season", "keywords", "suitable_duration", "image_url"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            
    if "estimated_daily_budget" in df.columns:
        df["estimated_daily_budget"] = pd.to_numeric(df["estimated_daily_budget"], errors="coerce").fillna(2500).astype(int)
        
    return df

def create_searchable_document(row: pd.Series) -> str:
    """
    Combine destination attributes into a comprehensive, weighted textual document
    optimized for TF-IDF representation.
    Emphasizes destination name, travel type, activities, attractions, and keywords.
    """
    parts = [
        # Boost important terms by controlled repetition for TF weighting
        f"{row.get('name', '')} {row.get('name', '')}",
        f"{row.get('travel_type', '')} {row.get('travel_type', '')}",
        f"{row.get('state', '')} {row.get('region', '')}",
        f"{row.get('short_description', '')}",
        f"{row.get('detailed_description', '')}",
        f"{row.get('attractions', '')}",
        f"{row.get('activities', '')}",
        f"{row.get('keywords', '')}",
        f"{row.get('best_season', '')}",
        f"{row.get('budget_category', '')}"
    ]
    raw_doc = " ".join(parts)
    return preprocess_text(raw_doc)

def build_corpus(df: pd.DataFrame) -> list[str]:
    """
    Build preprocessed textual corpus from DataFrame.
    """
    return [create_searchable_document(row) for _, row in df.iterrows()]

def apply_filters(
    df: pd.DataFrame,
    states: list[str] = None,
    regions: list[str] = None,
    travel_types: list[str] = None,
    budget_categories: list[str] = None,
    seasons: list[str] = None,
    max_budget: int = None
) -> pd.DataFrame:
    """
    Filter the destinations dataframe based on user criteria.
    Returns filtered subset.
    """
    filtered = df.copy()
    
    if states:
        filtered = filtered[filtered["state"].isin(states)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if travel_types:
        filtered = filtered[filtered["travel_type"].isin(travel_types)]
    if budget_categories:
        filtered = filtered[filtered["budget_category"].isin(budget_categories)]
    if seasons:
        filtered = filtered[filtered["best_season"].isin(seasons)]
    if max_budget is not None and max_budget > 0:
        filtered = filtered[filtered["estimated_daily_budget"] <= max_budget]
        
    return filtered

def format_currency(amount: int) -> str:
    """
    Format integer to Indian Rupee representation.
    """
    return f"₹{amount:,.0f}"

def get_image_fallback(destination_name: str) -> str:
    """
    Return a reliable fallback image or placeholder.
    """
    return "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80"
