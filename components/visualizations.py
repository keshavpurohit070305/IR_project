"""
WanderWise - Information Retrieval Educational Visualizations
Interactive Plotly charts demonstrating Vector Space Model & Rocchio Algorithm.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_query_term_weights(weights_dict: dict[str, float]):
    """
    Plotly bar chart of TF-IDF feature weights for terms present in the query.
    """
    if not weights_dict:
        return None
        
    df_plot = pd.DataFrame({
        "Term": list(weights_dict.keys()),
        "TF-IDF Weight": list(weights_dict.values())
    }).sort_values(by="TF-IDF Weight", ascending=True)
    
    fig = px.bar(
        df_plot,
        x="TF-IDF Weight",
        y="Term",
        orientation="h",
        title="Query Vector: Term Weights (TF-IDF)",
        color="TF-IDF Weight",
        color_continuous_scale=["#B7E4C7", "#2D6A4F", "#1B4332"],
        text_auto=".3f"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        coloraxis_showscale=False
    )
    return fig

def plot_rocchio_feature_shifts(boosted_terms: list, demoted_terms: list):
    """
    Plotly diverging/comparative bar chart showing terms boosted by relevant documents
    and terms suppressed by non-relevant documents.
    """
    data = []
    for term, val in boosted_terms[:8]:
        data.append({"Term": f"+ {term}", "Shift": val, "Type": "Boosted (Relevant)"})
    for term, val in demoted_terms[:8]:
        data.append({"Term": f"- {term}", "Shift": -val, "Type": "Suppressed (Non-Relevant)"})
        
    if not data:
        return None
        
    df = pd.DataFrame(data).sort_values(by="Shift", ascending=True)
    
    fig = px.bar(
        df,
        x="Shift",
        y="Term",
        orientation="h",
        color="Type",
        color_discrete_map={
            "Boosted (Relevant)": "#2D6A4F",
            "Suppressed (Non-Relevant)": "#E07A5F"
        },
        title="Rocchio Query Expansion: Vocabulary Weight Shifts (Δ Vector)",
        text_auto=".3f"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_rank_comparison(refined_df: pd.DataFrame, top_n: int = 10):
    """
    Plotly slope/scatter chart visualizing movement from Original Rank to Refined Rank.
    """
    subset = refined_df.head(top_n).copy()
    
    fig = go.Figure()
    
    for _, row in subset.iterrows():
        name = row["name"]
        orig_r = row["original_rank"]
        ref_r = row["refined_rank"]
        shift = row["rank_shift"]
        
        # Color coding: Green for promoted, Red for demoted, Gray for unchanged
        color = "#2D6A4F" if shift > 0 else "#E07A5F" if shift < 0 else "#9CA3AF"
        
        # Line connecting original and refined rank
        fig.add_trace(go.Scatter(
            x=["Initial Search", "Refined (Rocchio)"],
            y=[orig_r, ref_r],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            hovertemplate=f"<b>{name}</b><br>Initial: #{orig_r}<br>Refined: #{ref_r}<br>Movement: {shift:+d}<extra></extra>"
        ))
        
    fig.update_layout(
        title=f"Rank Movement: Initial vs. Refined Rankings (Top {top_n})",
        yaxis=dict(
            autorange="reversed",
            title="Rank Position (1 is Highest)",
            tickmode="linear",
            tick0=1,
            dtick=1
        ),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=50, b=30)
    )
    
    return fig

def plot_similarity_score_comparison(refined_df: pd.DataFrame, top_n: int = 8):
    """
    Grouped bar chart comparing original vs refined cosine similarity scores.
    """
    subset = refined_df.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            name="Original Similarity",
            x=subset["name"],
            y=subset["original_score_pct"],
            marker_color="#A7D7C5"
        ),
        go.Bar(
            name="Refined Similarity",
            x=subset["name"],
            y=subset["similarity_percentage"],
            marker_color="#1B4332"
        )
    ])
    
    fig.update_layout(
        barmode="group",
        title=f"Cosine Similarity Score Shift (% Match for Top {top_n})",
        yaxis=dict(title="Similarity Match (%)"),
        xaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
