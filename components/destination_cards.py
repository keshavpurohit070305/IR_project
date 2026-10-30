"""
WanderWise - Destination Cards Component
Renders responsive destination cards, action triggers (Like/Dislike),
and comprehensive details modal/expanders with IR metrics.
"""

import streamlit as st
import pandas as pd
from utils.helpers import format_currency, get_image_fallback

def render_destination_card(
    row: pd.Series,
    card_index: int,
    show_ir_metrics: bool = True,
    allow_feedback: bool = True,
    key_prefix: str = "card"
):
    """
    Render a single modern destination card with image, badges, details,
    and interactive Like/Dislike feedback buttons.
    """
    dest_id = row.get("destination_id", f"DEST_{card_index}")
    name = row.get("name", "Unknown Destination")
    state = row.get("state", "")
    region = row.get("region", "")
    travel_type = row.get("travel_type", "Leisure")
    budget_cat = row.get("budget_category", "Mid-range")
    daily_budget = row.get("estimated_daily_budget", 2500)
    duration = row.get("suitable_duration", "3-4 Days")
    best_season = row.get("best_season", "All Year")
    short_desc = row.get("short_description", "")
    detailed_desc = row.get("detailed_description", "")
    attractions = row.get("attractions", "")
    activities = row.get("activities", "")
    image_url = row.get("image_url", get_image_fallback(name))
    
    # IR metrics if present
    similarity_score = row.get("similarity_score", None)
    similarity_pct = row.get("similarity_percentage", None)
    matching_keywords = row.get("matching_keywords", [])
    match_explanation = row.get("match_explanation", "")
    rank = row.get("rank", None)
    
    # Movement status for Rocchio
    movement_status = row.get("movement_status", None)
    rank_shift = row.get("rank_shift", None)
    score_change_pct = row.get("score_change_pct", None)
    
    # State tracking
    is_liked = dest_id in st.session_state.get("liked_destinations", set())
    is_disliked = dest_id in st.session_state.get("disliked_destinations", set())

    with st.container():
        # Top image with overlay badges
        badge_html = ""
        if similarity_pct is not None and similarity_pct > 0:
            badge_color = "#1B4332" if similarity_pct >= 30 else "#E07A5F" if similarity_pct >= 15 else "#6C757D"
            badge_html = f'<div style="position: absolute; top: 12px; right: 12px; background: {badge_color}; color: white; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">{similarity_pct:.1f}% Match</div>'
            
        type_badge_html = f'<div style="position: absolute; top: 12px; left: 12px; background: rgba(255,255,255,0.92); color: #1B4332; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.75rem;">{travel_type}</div>'

        card_html = f"""
        <div style="background: white; border: 1px solid #E8E5DF; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.05); margin-bottom: 12px;">
            <div style="position: relative; height: 180px; width: 100%; overflow: hidden; background-color: #ECEEEF;">
                <img src="{image_url}" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80';" style="width: 100%; height: 100%; object-fit: cover;" alt="{name}" />
                {type_badge_html}
                {badge_html}
            </div>
            <div style="padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                    <h3 style="margin: 0; font-size: 1.25rem; color: #1B4332; font-weight: 700;">{name}</h3>
                    <span style="font-size: 0.8rem; color: #667085; font-weight: 600;">{duration}</span>
                </div>
                <div style="color: #667085; font-size: 0.85rem; font-weight: 500; margin-bottom: 8px;">
                    📍 {state} &bull; {region}
                </div>
                <p style="color: #344054; font-size: 0.88rem; line-height: 1.45; margin-bottom: 12px; min-height: 48px;">
                    {short_desc}
                </p>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #F2F4F7; padding-top: 10px;">
                    <div>
                        <span style="font-size: 0.75rem; color: #667085; display: block;">Daily Est. Budget</span>
                        <span style="font-size: 1rem; font-weight: 700; color: #1B4332;">{format_currency(daily_budget)}</span>
                        <span style="font-size: 0.75rem; color: #98A2B3;">/ day</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; color: #667085; display: block;">Best Season</span>
                        <span style="font-size: 0.85rem; font-weight: 600; color: #E07A5F;">☀️ {best_season}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Promotion / Demotion indicator if Rocchio was run
        if movement_status:
            if rank_shift and rank_shift > 0:
                st.success(f"🚀 **{movement_status}** | Score changed by +{score_change_pct}%")
            elif rank_shift and rank_shift < 0:
                st.error(f"📉 **{movement_status}** | Score changed by {score_change_pct}%")
            else:
                st.info(f"⚖️ **{movement_status}**")
        
        # Action Buttons (Like / Dislike / Details)
        col1, col2, col3 = st.columns([1, 1, 1.4])
        
        with col1:
            like_label = "❤️ Liked" if is_liked else "👍 Relevant"
            like_type = "primary" if is_liked else "secondary"
            if st.button(like_label, key=f"{key_prefix}_like_{dest_id}_{card_index}", type=like_type, use_container_width=True):
                # Toggle like
                if is_liked:
                    st.session_state["liked_destinations"].discard(dest_id)
                else:
                    st.session_state["liked_destinations"].add(dest_id)
                    # Disallow simultaneous dislike
                    st.session_state["disliked_destinations"].discard(dest_id)
                st.rerun()

        with col2:
            dislike_label = "🚫 Disliked" if is_disliked else "👎 Not Rel"
            dislike_type = "primary" if is_disliked else "secondary"
            if st.button(dislike_label, key=f"{key_prefix}_dislike_{dest_id}_{card_index}", type=dislike_type, use_container_width=True):
                # Toggle dislike
                if is_disliked:
                    st.session_state["disliked_destinations"].discard(dest_id)
                else:
                    st.session_state["disliked_destinations"].add(dest_id)
                    # Disallow simultaneous like
                    st.session_state["liked_destinations"].discard(dest_id)
                st.rerun()
                
        with col3:
            details_exp = st.expander("🔍 View Details")
            with details_exp:
                st.markdown(f"#### About {name}")
                st.write(detailed_desc)
                st.markdown(f"**🏛️ Top Attractions:** {attractions}")
                st.markdown(f"**🏄 Exciting Activities:** {activities}")
                st.markdown(f"**🏷️ Budget Category:** {budget_cat}")
                
                if show_ir_metrics and similarity_score is not None:
                    st.markdown("---")
                    st.markdown("##### 🔬 Information Retrieval Analysis")
                    st.write(f"- **Rank:** #{rank}")
                    st.write(f"- **Cosine Similarity Score:** `{similarity_score:.4f}` ({similarity_pct}%)")
                    if matching_keywords:
                        kw_str = ", ".join([f"`{k}`" for k in matching_keywords])
                        st.markdown(f"- **Matching Terms:** {kw_str}")
                    st.markdown(f"- **Relevance Explanation:** *{match_explanation}*")

def render_destination_grid(
    df: pd.DataFrame,
    columns: int = 3,
    max_cards: int = 15,
    key_prefix: str = "grid"
):
    """
    Render a responsive multi-column grid of destination cards.
    """
    if df.empty:
        st.info("No destinations matched your criteria. Try adjusting your search query or filters.")
        return

    subset = df.head(max_cards)
    total = len(subset)
    
    # Split into rows of `columns`
    for i in range(0, total, columns):
        row_items = subset.iloc[i : i + columns]
        cols = st.columns(columns)
        for col_idx, (_, row) in enumerate(row_items.iterrows()):
            with cols[col_idx]:
                render_destination_card(
                    row=row,
                    card_index=i + col_idx,
                    key_prefix=key_prefix
                )
