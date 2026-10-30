"""
WanderWise – AI Travel Destination Discovery Engine
===================================================
An Information Retrieval Laboratory Demonstration Project
Featuring:
- Practical 6: Vector Space Model using TF, IDF, TF-IDF Weighting & Cosine Similarity
- Practical 7: Rocchio Algorithm for Query Expansion & Relevance Feedback

Deployment Target: Streamlit Community Cloud
Repository: keshavpurohit070305/IR_project
Main Entrypoint: app.py (Root Directory)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first Streamlit call
st.set_page_config(
    page_title="WanderWise – AI Travel Discovery",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
from utils.helpers import (
    load_destinations_data,
    build_corpus,
    apply_filters,
    format_currency
)
from utils.preprocessing import preprocess_text
from algorithms.tfidf import TFIDFEngine
from algorithms.cosine_similarity import (
    compute_cosine_similarity,
    rank_destinations
)
from algorithms.rocchio import RocchioRelevanceFeedback
from components.destination_cards import render_destination_card, render_destination_grid
from components.search_interface import render_search_header, render_filter_sidebar, EXAMPLE_QUERIES
from components.visualizations import (
    plot_query_term_weights,
    plot_rocchio_feature_shifts,
    plot_rank_comparison,
    plot_similarity_score_comparison
)

# ---------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "liked_destinations" not in st.session_state:
    st.session_state["liked_destinations"] = set()

if "disliked_destinations" not in st.session_state:
    st.session_state["disliked_destinations"] = set()

if "search_query_input" not in st.session_state:
    st.session_state["search_query_input"] = ""

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Discover (Home)"

# ---------------------------------------------------------
# Data & IR Engine Caching
# ---------------------------------------------------------
@st.cache_data
def get_destinations_df():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "destinations.csv")
    df = load_destinations_data(csv_path)
    df["original_idx"] = df.index
    return df

@st.cache_resource
def get_tfidf_engine(corpus):
    return TFIDFEngine(corpus)

df_all = get_destinations_df()
corpus = build_corpus(df_all)
tfidf_engine = get_tfidf_engine(corpus)

# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 12px; border-bottom: 1px solid #E8E5DF; margin-bottom: 16px;">
        <h2 style="margin: 0; color: #1B4332; font-size: 1.8rem;">🧭 WanderWise</h2>
        <span style="font-size: 0.85rem; color: #576561; font-weight: 500;">
            AI Destination Discovery Engine
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        options=[
            "Discover (Home)",
            "Explore Destinations",
            "Rocchio Feedback Lab",
            "My Discoveries",
            "IR Theory & Viva Prep",
            "About & Lab Mapping"
        ],
        index=0,
        key="nav_selection"
    )
    
    st.markdown("---")
    
    # User feedback status widget in sidebar
    liked_count = len(st.session_state["liked_destinations"])
    disliked_count = len(st.session_state["disliked_destinations"])
    
    st.markdown("##### 📊 Feedback Status")
    col_fb1, col_fb2 = st.columns(2)
    with col_fb1:
        st.metric("Relevant (👍)", liked_count)
    with col_fb2:
        st.metric("Non-Rel (👎)", disliked_count)
        
    if liked_count > 0 or disliked_count > 0:
        if st.button("🗑️ Reset All Feedback", use_container_width=True):
            st.session_state["liked_destinations"] = set()
            st.session_state["disliked_destinations"] = set()
            st.rerun()

# =========================================================
# PAGE 1: DISCOVER (HOME)
# =========================================================
if page == "Discover (Home)":
    # Search header with example queries
    query = render_search_header(st.session_state.get("search_query_input", ""))
    
    # Sidebar Filters
    active_filters = render_filter_sidebar(df_all)
    filtered_df = apply_filters(
        df_all,
        regions=active_filters["regions"],
        travel_types=active_filters["travel_types"],
        budget_categories=active_filters["budget_categories"],
        seasons=active_filters["seasons"],
        max_budget=active_filters["max_budget"]
    )
    
    # Process Search Query
    if query.strip():
        # Preprocess query & compute TF-IDF vector
        query_vector = tfidf_engine.transform_query(query)
        term_weights = tfidf_engine.get_query_terms_weights(query)
        
        # Rank all destinations
        ranked_all = rank_destinations(
            df=df_all,
            query_vector=query_vector,
            tfidf_matrix=tfidf_engine.tfidf_matrix,
            feature_names=tfidf_engine.feature_names,
            query_text=query
        )
        
        # Filter ranked results
        display_results = ranked_all[ranked_all["destination_id"].isin(filtered_df["destination_id"])].copy()
        matching_count = len(display_results[display_results["similarity_score"] > 0.001])
        
        st.markdown("---")
        
        # Search Summary Header
        st.markdown(f"### Search Results for: *\"{query}\"*")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total In Catalog", len(df_all))
        with col_m2:
            st.metric("Filtered Candidates", len(filtered_df))
        with col_m3:
            st.metric("Relevant Matches", matching_count)
        with col_m4:
            top_match_pct = display_results.iloc[0]["similarity_percentage"] if not display_results.empty else 0
            st.metric("Top Match Score", f"{top_match_pct}%")
            
        # Optional Educational Query Vector Weights Expander
        if term_weights:
            with st.expander("🔬 View Query TF-IDF Representation (Practical 6)"):
                st.markdown("Below is the mathematical term frequency-inverse document frequency weighting of terms in your query:")
                fig_q = plot_query_term_weights(term_weights)
                if fig_q:
                    st.plotly_chart(fig_q, use_container_width=True)
                
                kw_badges = " ".join([f"<span class='badge-pill badge-green'>{k}: {v:.3f}</span>" for k, v in term_weights.items()])
                st.markdown(f"**Recognized Vocabulary Tokens:** {kw_badges}", unsafe_allow_html=True)
                
        # Zero results handling
        if matching_count == 0:
            st.warning("⚠️ No destinations had direct vocabulary overlap with your query. Try broadening your terms (e.g. 'beach', 'mountains', 'monsoon', 'temple', 'safari').")
        else:
            st.markdown("#### 🌟 Top Matched Destinations (Ranked by Cosine Similarity)")
            render_destination_grid(display_results, columns=3, max_cards=12, key_prefix="home_search")
            
    else:
        # Initial Landing State: Featured Destinations
        st.markdown("### 🏔️ Featured Trending Destinations across India")
        st.markdown("<p style='color: #576561; margin-bottom: 1.5rem;'>Curated highlights spanning heritage, serene beaches, misty hill stations, and royal sanctuaries.</p>", unsafe_allow_html=True)
        
        featured_ids = ["DEST001", "DEST017", "DEST027", "DEST029", "DEST034", "DEST049"]
        featured_df = df_all[df_all["destination_id"].isin(featured_ids)].copy()
        
        render_destination_grid(featured_df, columns=3, max_cards=6, key_prefix="home_featured")
        
        st.markdown("---")
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 24px; border: 1px solid #E8E5DF; text-align: center;">
            <h3 style="color: #1B4332; margin-bottom: 8px;">Ready to find your personalized getaway?</h3>
            <p style="color: #576561; max-width: 650px; margin: 0 auto 16px auto;">
                Type your desired travel experience into the search box above (e.g., <em>"peaceful hill station with trekking and tea gardens"</em>) to see the Vector Space TF-IDF model in action!
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PAGE 2: EXPLORE DESTINATIONS
# =========================================================
elif page == "Explore Destinations":
    st.markdown("## 🗺️ Explore All Indian Destinations")
    st.markdown("Browse our curated catalog of 66 verified destinations with comprehensive multi-criteria filtering.")
    
    # Sidebar Filters
    active_filters = render_filter_sidebar(df_all)
    filtered_df = apply_filters(
        df_all,
        regions=active_filters["regions"],
        travel_types=active_filters["travel_types"],
        budget_categories=active_filters["budget_categories"],
        seasons=active_filters["seasons"],
        max_budget=active_filters["max_budget"]
    )
    
    # Internal filter search
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        keyword = st.text_input("Filter within catalog by name, state, or attraction:", placeholder="e.g. Kerala, Fort, Waterfall...")
    with col_s2:
        sort_by = st.selectbox("Sort By", ["Name (A-Z)", "Daily Budget (Low to High)", "Daily Budget (High to Low)"])
        
    if keyword.strip():
        kw_clean = keyword.lower()
        filtered_df = filtered_df[
            filtered_df["name"].str.lower().str.contains(kw_clean) |
            filtered_df["state"].str.lower().str.contains(kw_clean) |
            filtered_df["attractions"].str.lower().str.contains(kw_clean) |
            filtered_df["keywords"].str.lower().str.contains(kw_clean)
        ]
        
    if sort_by == "Name (A-Z)":
        filtered_df = filtered_df.sort_values(by="name")
    elif sort_by == "Daily Budget (Low to High)":
        filtered_df = filtered_df.sort_values(by="estimated_daily_budget", ascending=True)
    elif sort_by == "Daily Budget (High to Low)":
        filtered_df = filtered_df.sort_values(by="estimated_daily_budget", ascending=False)
        
    st.markdown(f"**Showing {len(filtered_df)} of {len(df_all)} destinations**")
    st.markdown("---")
    
    render_destination_grid(filtered_df, columns=3, max_cards=66, key_prefix="explore")

# =========================================================
# PAGE 3: ROCCHIO FEEDBACK LAB (PRACTICAL 7)
# =========================================================
elif page == "Rocchio Feedback Lab":
    st.markdown("## 🔬 Practical 7: Rocchio Relevance Feedback Lab")
    st.markdown("""
    Experience how user interaction refines information retrieval through the genuine **Rocchio Algorithm**.
    Mark destinations as **Relevant (👍)** or **Non-Relevant (👎)**, tune the parameters, and watch the query vector shift in the vector space!
    """)
    
    # Query input for the lab
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        lab_query = st.text_input(
            "Initial Query for Rocchio Refinement:",
            value=st.session_state.get("search_query_input", "peaceful water places for vacation"),
            key="rocchio_lab_query"
        )
    with col_q2:
        if st.button("Apply Query", use_container_width=True):
            st.session_state["search_query_input"] = lab_query
            st.rerun()

    # Parameter Tuning Controls
    st.markdown("#### ⚙️ Rocchio Weight Hyperparameters")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        alpha = st.slider("Alpha (α) - Original Query Weight", 0.0, 2.0, 1.0, 0.05, 
                          help="Controls fidelity to original user query.")
    with p_col2:
        beta = st.slider("Beta (β) - Relevant Docs Weight", 0.0, 2.0, 0.75, 0.05,
                         help="Pulls the query vector toward the centroid of liked documents.")
    with p_col3:
        gamma = st.slider("Gamma (γ) - Non-Relevant Docs Weight", 0.0, 2.0, 0.25, 0.05,
                          help="Pushes the query vector away from the centroid of disliked documents.")

    # Mathematical Formula Box
    st.markdown(f"""
    <div class="viva-box">
        <div class="viva-title">📐 Rocchio Mathematical Formulation</div>
        <div class="viva-math">
            <b>q_new</b> = ({alpha} &times; <b>q_orig</b>) + ({beta} &times; <b>&mu;_relevant</b>) - ({gamma} &times; <b>&mu;_irrelevant</b>)
        </div>
        <span style="font-size: 0.85rem; color: #576561;">
            Where &mu;_relevant is the centroid of Relevant items, and &mu;_irrelevant is the centroid of Non-Relevant items. Negative term weights are clipped to 0.0, followed by L2 normalization.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Execute initial TF-IDF search
    q_orig_vec = tfidf_engine.transform_query(lab_query)
    orig_ranked = rank_destinations(
        df=df_all,
        query_vector=q_orig_vec,
        tfidf_matrix=tfidf_engine.tfidf_matrix,
        feature_names=tfidf_engine.feature_names,
        query_text=lab_query
    )
    
    # Retrieve active user feedback
    liked_ids = list(st.session_state["liked_destinations"])
    disliked_ids = list(st.session_state["disliked_destinations"])
    
    # Map IDs to corpus indices
    liked_indices = [
        int(df_all[df_all["destination_id"] == did].index[0]) 
        for did in liked_ids if not df_all[df_all["destination_id"] == did].empty
    ]
    disliked_indices = [
        int(df_all[df_all["destination_id"] == did].index[0]) 
        for did in disliked_ids if not df_all[df_all["destination_id"] == did].empty
    ]
    
    # Feedback Summary
    fb_col1, fb_col2, fb_col3 = st.columns(3)
    with fb_col1:
        st.info(f"👍 **Relevant Selected ({len(liked_ids)}):** " + (", ".join(df_all[df_all["destination_id"].isin(liked_ids)]["name"].tolist()) if liked_ids else "None yet"))
    with fb_col2:
        st.warning(f"👎 **Non-Relevant Selected ({len(disliked_ids)}):** " + (", ".join(df_all[df_all["destination_id"].isin(disliked_ids)]["name"].tolist()) if disliked_ids else "None yet"))
    with fb_col3:
        if st.button("🔄 Reset Feedback Selections", use_container_width=True):
            st.session_state["liked_destinations"] = set()
            st.session_state["disliked_destinations"] = set()
            st.rerun()

    if not liked_indices and not disliked_indices:
        st.warning("👉 **Action Required:** Please like (👍 Relevant) or dislike (👎 Not Rel) at least one destination card below to enable Rocchio calculation.")
        st.markdown("#### Initial Search Results (Select feedback below):")
        render_destination_grid(orig_ranked, columns=3, max_cards=6, key_prefix="rocchio_init")
    else:
        # Perform genuine Rocchio calculation
        rocchio = RocchioRelevanceFeedback(alpha=alpha, beta=beta, gamma=gamma)
        q_new_vec = rocchio.update_query_vector(
            q_original=q_orig_vec,
            relevant_indices=liked_indices,
            irrelevant_indices=disliked_indices,
            tfidf_matrix=tfidf_engine.tfidf_matrix
        )
        
        # Compare Rankings
        refined_ranked = rocchio.compare_rankings(
            df_original_ranked=orig_ranked,
            q_new=q_new_vec,
            tfidf_matrix=tfidf_engine.tfidf_matrix
        )
        
        # Extract vocabulary shifts
        boosted, demoted = rocchio.get_query_shift_terms(
            q_original=q_orig_vec,
            q_new=q_new_vec,
            feature_names=tfidf_engine.feature_names,
            top_k=8
        )
        
        st.markdown("---")
        st.markdown("### 📊 Rocchio Refinement Visualizations")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_shift = plot_rocchio_feature_shifts(boosted, demoted)
            if fig_shift:
                st.plotly_chart(fig_shift, use_container_width=True)
            else:
                st.info("Feature weights unchanged.")
                
        with v_col2:
            fig_rank = plot_rank_comparison(refined_ranked, top_n=8)
            st.plotly_chart(fig_rank, use_container_width=True)
            
        fig_scores = plot_similarity_score_comparison(refined_ranked, top_n=8)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Promotion / Demotion Summary Table
        st.markdown("#### 📋 Rank Movement Comparative Table")
        comparison_display = refined_ranked[[
            "rank", "original_rank", "name", "state", "travel_type", 
            "original_score_pct", "similarity_percentage", "rank_shift", "movement_status"
        ]].head(10).rename(columns={
            "rank": "Refined Rank",
            "original_rank": "Initial Rank",
            "name": "Destination",
            "state": "State",
            "travel_type": "Travel Type",
            "original_score_pct": "Initial Match %",
            "similarity_percentage": "Refined Match %",
            "rank_shift": "Movement",
            "movement_status": "Status"
        })
        st.dataframe(comparison_display, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🚀 Refined Search Results (Post-Rocchio)")
        render_destination_grid(refined_ranked, columns=3, max_cards=9, key_prefix="rocchio_refined")

# =========================================================
# PAGE 4: MY DISCOVERIES
# =========================================================
elif page == "My Discoveries":
    st.markdown("## 💖 My Saved Destinations")
    st.markdown("All destinations you marked as **Relevant (👍)** during discovery are organized here.")
    
    liked_ids = list(st.session_state["liked_destinations"])
    
    if not liked_ids:
        st.info("You haven't liked any destinations yet. Explore or search destinations and click '👍 Relevant' to save them here.")
    else:
        saved_df = df_all[df_all["destination_id"].isin(liked_ids)].copy()
        
        total_daily_est = saved_df["estimated_daily_budget"].sum()
        avg_budget = saved_df["estimated_daily_budget"].mean()
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Saved Places", len(saved_df))
        with col_s2:
            st.metric("Average Daily Budget", format_currency(avg_budget))
        with col_s3:
            st.metric("Estimated Daily Combined", format_currency(total_daily_est))
            
        st.markdown("---")
        
        for _, row in saved_df.iterrows():
            did = row["destination_id"]
            c1, c2, c3, c4 = st.columns([1, 2.5, 1, 0.8])
            with c1:
                st.image(row["image_url"], use_container_width=True)
            with c2:
                st.markdown(f"### {row['name']}")
                st.markdown(f"📍 **{row['state']}** &bull; {row['travel_type']} &bull; {row['suitable_duration']}")
                st.write(row["short_description"])
            with c3:
                st.markdown(f"**Daily Budget:** {format_currency(row['estimated_daily_budget'])}")
                st.markdown(f"**Best Season:** {row['best_season']}")
            with c4:
                if st.button("❌ Remove", key=f"remove_saved_{did}", use_container_width=True):
                    st.session_state["liked_destinations"].discard(did)
                    st.rerun()
            st.markdown("---")

# =========================================================
# PAGE 5: IR THEORY & VIVA PREPARATION
# =========================================================
elif page == "IR Theory & Viva Prep":
    st.markdown("## 📚 Information Retrieval Viva Examination Guide")
    st.markdown("Comprehensive theoretical walkthrough and mathematical proofs for **Practical 6** and **Practical 7**.")
    
    tab_p6, tab_p7, tab_matrix, tab_viva = st.tabs([
        "Practical 6: Vector Space Model",
        "Practical 7: Rocchio Algorithm",
        "Live Step-by-Step Calculation Matrix",
        "Viva Q&A Master Checklist"
    ])
    
    with tab_p6:
        st.markdown("""
        ### Practical 6: Vector Space Model (VSM) using TF-IDF & Cosine Similarity
        
        #### 1. What is the Vector Space Model?
        In the Vector Space Model (Salton, 1975), documents and queries are represented as high-dimensional vectors in a shared euclidean space, where each dimension corresponds to a distinct vocabulary term.
        
        #### 2. Term Frequency (TF):
        Measures the occurrence frequency of term $t$ in document $d$:
        $$\\text{TF}(t, d) = \\frac{f_{t, d}}{\\sum_{t' \\in d} f_{t', d}}$$
        
        #### 3. Inverse Document Frequency (IDF):
        Measures how rare or informative a term is across the entire corpus of $N$ documents:
        $$\\text{IDF}(t) = \\ln\\left(\\frac{N + 1}{\\text{DF}(t) + 1}\\right) + 1$$
        Where $\\text{DF}(t)$ is the document frequency (number of documents containing term $t$).
        
        #### 4. TF-IDF Weighting:
        $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$
        Common words across all documents get low IDF; unique words specific to a destination get high weights.
        
        #### 5. Cosine Similarity:
        Measures the angle between the query vector $\\vec{q}$ and document vector $\\vec{d}$:
        $$\\text{Cosine Similarity}(\\vec{q}, \\vec{d}) = \\frac{\\vec{q} \\cdot \\vec{d}}{\\|\\vec{q}\\| \\|\\vec{d}\\|} = \\frac{\\sum_{i=1}^{V} q_i \\cdot d_i}{\\sqrt{\\sum_{i=1}^V q_i^2} \\cdot \\sqrt{\\sum_{i=1}^V d_i^2}}$$
        The score ranges from 0.0 (orthogonal / completely unrelated) to 1.0 (identical directional orientation).
        """)
        
    with tab_p7:
        st.markdown("""
        ### Practical 7: Rocchio Relevance Feedback & Query Expansion
        
        #### 1. Motivation:
        User queries are frequently short, ambiguous, or miss key domain vocabulary. Rocchio relevance feedback allows the retrieval engine to incorporate user feedback (Likes and Dislikes) to reformulate the query in vector space.
        
        #### 2. Rocchio Mathematical Formula:
        $$\\vec{q}_{new} = \\alpha \\vec{q}_{original} + \\frac{\\beta}{|D_R|} \\sum_{d \\in D_R} \\vec{d} - \\frac{\\gamma}{|D_{NR}|} \\sum_{d \\in D_{NR}} \\vec{d}$$
        
        - $\\vec{q}_{original}$: Initial query TF-IDF vector.
        - $D_R$: Set of documents judged **Relevant** (Liked).
        - $D_{NR}$: Set of documents judged **Non-Relevant** (Disliked).
        - $\\alpha$: Weight on original query (default: 1.0).
        - $\\beta$: Weight on relevant document centroid (default: 0.75).
        - $\\gamma$: Weight on irrelevant document centroid (default: 0.25).
        
        #### 3. Non-Negative Constraint:
        In vector space information retrieval, negative frequencies are undefined. Thus:
        $$\\vec{q}_{new} = \\max(0, \\vec{q}_{new})$$
        
        #### 4. Query Expansion Effect:
        Terms appearing in relevant documents that were **not** in the original query receive positive weights in $\\vec{q}_{new}$, naturally expanding the query without requiring manual keyword retyping!
        """)
        
    with tab_matrix:
        st.markdown("### 🔢 Live Numerical Calculation Walkthrough")
        st.markdown("Observe the exact token frequencies, DF, IDF, and TF-IDF weights computed live on a controlled 3-document sub-corpus:")
        
        sample_doc_1 = "Manali snow mountain skiing winter cold adventure"
        sample_doc_2 = "Goa beach party watersports sun sand sea ocean"
        sample_doc_3 = "Rishikesh yoga spiritual ganges rafting river ashram"
        sample_q = "snow mountain skiing"
        
        sample_docs = [sample_doc_1, sample_doc_2, sample_doc_3]
        breakdown_df = tfidf_engine.compute_step_by_step_breakdown(sample_docs, sample_q)
        st.dataframe(breakdown_df, use_container_width=True)

    with tab_viva:
        st.markdown("""
        ### 🎯 Viva Examination Questions & Model Answers
        
        **Q1: Why is Cosine Similarity preferred over Euclidean Distance in text retrieval?**
        > *Answer:* Euclidean distance is heavily biased by document length (a long document and short document about the exact same topic would have large Euclidean distance). Cosine similarity normalizes for length by measuring the angle between vectors rather than magnitude, ensuring fair ranking regardless of document size.
        
        **Q2: What happens if IDF is not used and only TF is used?**
        > *Answer:* Frequent but non-discriminative terms (like 'place', 'visit', 'good') would dominate the ranking simply because they appear often, degrading search precision. IDF penalizes common words and rewards rare, content-rich terms.
        
        **Q3: How does Rocchio achieve query expansion?**
        > *Answer:* By computing the centroid $\\frac{1}{|D_R|}\\sum \\vec{d}$ of relevant documents, terms that were not present in the user's initial query but are prevalent in the liked documents get positive weights added to $\\vec{q}_{new}$.
        
        **Q4: Why is $\\gamma$ typically smaller than $\\beta$?**
        > *Answer:* In Information Retrieval, positive evidence is far more specific and reliable than negative evidence. Irrelevant documents may be irrelevant for hundreds of disparate reasons; hence penalizing them too aggressively (high $\\gamma$) could inadvertently suppress valuable terms.
        
        **Q5: Why must negative components in the updated Rocchio query vector be set to zero?**
        > *Answer:* In the standard Vector Space Model, term frequencies cannot be negative. Allowing negative weights would disrupt cosine similarity calculations and produce mathematically invalid similarity scores.
        """)

# =========================================================
# PAGE 6: ABOUT & LAB MAPPING
# =========================================================
elif page == "About & Lab Mapping":
    st.markdown("## ℹ️ About WanderWise")
    
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.markdown("""
        ### Project Overview
        **WanderWise** is an academic Information Retrieval laboratory engine developed for live viva and practical examination demonstrations.
        
        ### Syllabus Practical Mapping
        - **Practical 6:** Implementation of the Vector Space Model (VSM) using Term Frequency (TF), Inverse Document Frequency (IDF), TF-IDF weighting, and Cosine Similarity ranking.
        - **Practical 7:** Implementation of Query Expansion and Relevance Feedback using the Rocchio Algorithm.
        
        ### Architecture & Stack
        - **Language & Framework:** Python 3, Streamlit
        - **Retrieval Engine:** Scikit-Learn `TfidfVectorizer`, Scikit-Learn `cosine_similarity`
        - **Linear Algebra:** NumPy vector operations & centroid mathematics
        - **Visualizations:** Plotly interactive charts
        - **Dataset:** 66 Curated Indian Destinations (`data/destinations.csv`)
        - **Zero External Dependencies:** No external paid APIs, no API keys, zero network downloads during runtime.
        """)
    with col_a2:
        st.markdown("""
        <div style="background: white; border: 1px solid #E8E5DF; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <h4 style="color: #1B4332; margin-top: 0;">Deployment Details</h4>
            <p style="font-size: 0.85rem; color: #576561;">
                Configured for 1-click deployment on <b>Streamlit Community Cloud</b>.
            </p>
            <hr style="border: 0; border-top: 1px solid #ECEEEF; margin: 12px 0;">
            <p style="font-size: 0.85rem; margin: 4px 0;"><b>Repository:</b> keshavpurohit070305/IR_project</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><b>Branch:</b> main</p>
            <p style="font-size: 0.85rem; margin: 4px 0;"><b>Main file path:</b> app.py</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8C98A4; font-size: 0.85rem; padding: 12px 0;">
    WanderWise – AI Travel Discovery Engine &bull; Academic Information Retrieval Lab Demonstration &bull; Practical 6 &amp; 7
</div>
""", unsafe_allow_html=True)
