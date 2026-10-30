"""
WanderWise - Search & Filtering Interface Component
Provides natural language search, example query triggers, and multi-faceted filters.
"""

import streamlit as st
import pandas as pd

EXAMPLE_QUERIES = [
    "I want peaceful mountain destinations with trekking.",
    "Find affordable beach destinations in India.",
    "Show me historical places with forts and architecture.",
    "I love nature, waterfalls and adventure.",
    "Suggest relaxing destinations suitable for a family trip."
]

def render_search_header(current_query: str = "") -> str:
    """
    Renders the main search bar and example query suggestions.
    Returns the user's active search query.
    """
    # Check if a pill was clicked
    query_key = "search_query_input"
    if query_key not in st.session_state:
        st.session_state[query_key] = current_query

    # Hero section with search
    st.markdown("""
    <div class="hero-container">
        <span style="background-color: #2D6A4F; color: white; padding: 4px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: inline-block;">
            AI-POWERED INFORMATION RETRIEVAL ENGINE
        </span>
        <h1 class="hero-title">Discover places that match your kind of adventure.</h1>
        <p class="hero-subtitle">
            Powered by Vector Space TF-IDF Cosine Similarity &amp; Rocchio Relevance Feedback algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Input Bar
    query = st.text_input(
        "🔍 Search by vibe, terrain, activities, culture or preferences:",
        value=st.session_state.get(query_key, ""),
        placeholder="e.g. peaceful snowy mountains with scenic trekking and cafes...",
        key="search_input_field"
    )
    st.session_state[query_key] = query

    # Example query suggestions
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #576561; margin-bottom: 6px;'>💡 Try an example search:</p>", unsafe_allow_html=True)
    
    cols = st.columns(len(EXAMPLE_QUERIES))
    for i, ex in enumerate(EXAMPLE_QUERIES):
        with cols[i]:
            short_label = ex.replace("I want ", "").replace("Find ", "").replace("Show me ", "").replace("Suggest ", "")
            if len(short_label) > 24:
                short_label = short_label[:22] + "..."
            if st.button(f"✨ {short_label}", key=f"ex_btn_{i}", use_container_width=True):
                st.session_state[query_key] = ex
                st.rerun()

    return query

def render_filter_sidebar(df: pd.DataFrame) -> dict:
    """
    Renders comprehensive filters in the sidebar and returns active filter criteria.
    """
    st.sidebar.markdown("### 🎯 Filter Destinations")
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state["filter_regions"] = []
        st.session_state["filter_types"] = []
        st.session_state["filter_budgets"] = []
        st.session_state["filter_seasons"] = []
        st.session_state["filter_max_budget"] = 8000
        st.rerun()

    st.sidebar.markdown("---")
    
    # Region filter
    available_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Geographic Region",
        options=available_regions,
        default=st.session_state.get("filter_regions", []),
        key="sidebar_regions"
    )
    st.session_state["filter_regions"] = selected_regions

    # Travel Type filter
    available_types = sorted(df["travel_type"].unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Travel Type / Vibe",
        options=available_types,
        default=st.session_state.get("filter_types", []),
        key="sidebar_types"
    )
    st.session_state["filter_types"] = selected_types

    # Budget Category filter
    available_budgets = ["Budget", "Mid-range", "Luxury"]
    selected_budgets = st.sidebar.multiselect(
        "Budget Category",
        options=available_budgets,
        default=st.session_state.get("filter_budgets", []),
        key="sidebar_budgets"
    )
    st.session_state["filter_budgets"] = selected_budgets

    # Max Daily Budget Slider
    min_budget = int(df["estimated_daily_budget"].min())
    max_budget = int(df["estimated_daily_budget"].max())
    chosen_max_budget = st.sidebar.slider(
        "Max Daily Budget (INR)",
        min_value=min_budget,
        max_value=max_budget,
        value=st.session_state.get("filter_max_budget", max_budget),
        step=200,
        key="sidebar_max_budget"
    )
    st.session_state["filter_max_budget"] = chosen_max_budget

    # Best Season filter
    available_seasons = sorted(df["best_season"].unique().tolist())
    selected_seasons = st.sidebar.multiselect(
        "Best Season to Visit",
        options=available_seasons,
        default=st.session_state.get("filter_seasons", []),
        key="sidebar_seasons"
    )
    st.session_state["filter_seasons"] = selected_seasons

    return {
        "regions": selected_regions,
        "travel_types": selected_types,
        "budget_categories": selected_budgets,
        "max_budget": chosen_max_budget,
        "seasons": selected_seasons
    }
