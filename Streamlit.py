#!/usr/bin/env python3
"""
AI Reverse Logistics Agent - Streamlit app
Ready to upload as a single-file app (app.py).
"""
import os
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

# ---- Configuration & Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reverse-logistics-app")

st.set_page_config(page_title="AI Reverse Logistics Agent", page_icon="📦", layout="wide", initial_sidebar_state="expanded")

# ---- Styles ----
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Constants ----
INDIAN_LOCATIONS_COORDS = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
}

HUB_COORDS = {
    "Hub A": {"lat": 20.0, "lon": 75.0},  # Central India
    "Hub B": {"lat": 28.0, "lon": 77.0},  # North India
    "Hub C": {"lat": 15.0, "lon": 76.0},  # South India
    "Hub D": {"lat": 23.0, "lon": 85.0},  # East India
    "Hub E": {"lat": 22.0, "lon": 70.0},  # West India
}

# ---- Classes ----
class DispositionClassifier:
    """AI-powered grading and triage system for returned goods"""

    def __init__(self, confidence_threshold: float = 0.75):
        self.categories = ["Resell as New", "Refurbish", "Liquidate", "Recycle", "Dispose"]
        self.confidence_threshold = float(confidence_threshold)

    def classify_return(self, condition: Any, category: Any, value: Any, return_reason: Any) -> Tuple[str, float, Dict]:
        # Defensive conversions
        condition_str = str(condition) if pd.notna(condition) else "Unknown"
        return_reason_str = str(return_reason) if pd.notna(return_reason) else "Unknown"
        try:
            value_float = float(value) if pd.notna(value) else 0.0
        except Exception:
            value_float = 0.0

        scores = {cat: 0.0 for cat in self.categories}

        condition_weights = {
            "Unopened": {"Resell as New": 0.9, "Refurbish": 0.05, "Liquidate": 0.05},
            "Like New": {"Resell as New": 0.7, "Refurbish": 0.2, "Liquidate": 0.1},
            "Minor Defect": {"Refurbish": 0.7, "Liquidate": 0.2, "Resell as New": 0.1},
            "Major Defect": {"Refurbish": 0.4, "Liquidate": 0.4, "Recycle": 0.2},
            "Damaged": {"Recycle": 0.6, "Liquidate": 0.3, "Dispose": 0.1},
            "Unknown": {"Dispose": 1.0},
        }

        for cat, weight in condition_weights.get(condition_str, {}).items():
            scores[cat] += weight * 0.5

        # Value-based adjustments
        if value_float > 100:
            scores["Resell as New"] += 0.2
            scores["Refurbish"] += 0.15
        elif value_float > 50:
            scores["Refurbish"] += 0.15
            scores["Liquidate"] += 0.1
        else:
            scores["Liquidate"] += 0.15
            scores["Recycle"] += 0.1

        # Reason-based adjustments
        if "defect" in return_reason_str.lower():
            scores["Refurbish"] += 0.15
        elif "wrong" in return_reason_str.lower() or "unwanted" in return_reason_str.lower():
            scores["Resell as New"] += 0.2

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            scores = {cat: 0.0 for cat in self.categories}
            scores["Dispose"] = 1.0

        best = max(scores, key=scores.get)
        confidence = float(scores[best])
        recovery_rates = {
            "Resell as New": 0.85,
            "Refurbish": 0.60,
            "Liquidate": 0.30,
            "Recycle": 0.05,
            "Dispose": 0.0,
        }
        expected_recovery = value_float * recovery_rates.get(best, 0)
        processing_time = self._estimate_processing_time(best)

        metadata = {
            "confidence": confidence,
            "all_scores": scores,
            "expected_recovery": expected_recovery,
            "processing_days": processing_time,
            "needs_manual_review": confidence < self.confidence_threshold,
        }
        return best, confidence, metadata

    def _estimate_processing_time(self, disposition: str) -> int:
        time_map = {"Resell as New": 1, "Refurbish": 5, "Liquidate": 3, "Recycle": 7, "Dispose": 2}
        return int(time_map.get(disposition, 3))


class ReverseRoutingOptimizer:
    """Dynamic routing optimizer for return flows (deterministic if seed provided)"""

    def __init__(self, seed: Optional[int] = None):
        self.hubs = list(HUB_COORDS.keys())
        self.location_coords = INDIAN_LOCATIONS_COORDS
        self.hub_coords = HUB_COORDS
        # deterministic rng for testing
        if seed is None:
            self.rng = np.random
            self._using_default_rng = True
        else:
            try:
                # Use Generator when seed available for deterministic behavior
                self.rng = np.random.default_rng(seed)
                self._using_default_rng = False
            except Exception:
                self.rng = np.random
                self._using_default_rng = True

    def optimize_routes(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        optimized = returns_df.copy()

        required_input_cols = ["origin_location", "product_category", "urgency_score"]
        for col in required_input_cols:
            if col not in optimized.columns:
                optimized[col] = np.nan

        output_cols_to_initialize = [
            "optimal_hub",
            "transport_cost",
            "transit_days",
            "carbon_kg",
            "origin_lat",
            "origin_lon",
            "hub_lat",
            "hub_lon",
            "transport_mode",
        ]
        for col in output_cols_to_initialize:
            if col not in optimized.columns:
                optimized[col] = np.nan

        if not optimized.empty:
            optimized["optimal_hub"] = optimized.apply(
                lambda row: self._select_optimal_hub(row["origin_location"], row["urgency_score"])
                if pd.notna(row["origin_location"]) and pd.notna(row["urgency_score"])
                else np.nan,
                axis=1,
            )

            optimized["transport_cost"] = optimized.apply(
                lambda row: self._calculate_cost(row["origin_location"], row["optimal_hub"])
                if pd.notna(row["origin_location"]) and pd.notna(row["optimal_hub"])
                else np.nan,
                axis=1,
            )

            optimized["transit_days"] = optimized.apply(
                lambda row: self._calculate_transit_time(row["origin_location"], row["optimal_hub"])
                if pd.notna(row["origin_location"]) and pd.notna(row["optimal_hub"])
                else np.nan,
                axis=1,
            )

            optimized["carbon_kg"] = optimized.apply(
                lambda row: self._calculate_carbon(row["origin_location"], row["optimal_hub"])
                if pd.notna(row["origin_location"]) and pd.notna(row["optimal_hub"])
                else np.nan,
                axis=1,
            )

            optimized["transport_mode"] = optimized.apply(
                lambda row: self._calculate_transport_mode(row["urgency_score"], row["transport_cost"])
                if pd.notna(row["urgency_score"]) and pd.notna(row["transport_cost"])
                else "Unknown",
                axis=1,
            )

            optimized["origin_lat"] = optimized["origin_location"].apply(
                lambda x: self.location_coords.get(x, {}).get("lat") if pd.notna(x) else np.nan
            )
            optimized["origin_lon"] = optimized["origin_location"].apply(
                lambda x: self.location_coords.get(x, {}).get("lon") if pd.notna(x) else np.nan
            )
            optimized["hub_lat"] = optimized["optimal_hub"].apply(
                lambda x: self.hub_coords.get(x, {}).get("lat") if pd.notna(x) else np.nan
            )
            optimized["hub_lon"] = optimized["optimal_hub"].apply(
                lambda x: self.hub_coords.get(x, {}).get("lon") if pd.notna(x) else np.nan
            )

        return optimized

    def _select_optimal_hub(self, origin: str, urgency: float) -> str:
        # use safe default coords
        ox = self.location_coords.get(origin, {}).get("lat", 0.0)
        oy = self.location_coords.get(origin, {}).get("lon", 0.0)
        scores = {}
        for hub in self.hubs:
            hx = self.hub_coords[hub]["lat"]
            hy = self.hub_coords[hub]["lon"]
            dist = ((ox - hx) ** 2 + (oy - hy) ** 2) ** 0.5
            proximity_score = 1 / (1 + dist)
            # use rng consistently (default numpy or Generator)
            rand = self._rand_uniform(0.5, 1.0)
            weight_cost = 0.4 if (isinstance(urgency, (int, float)) and urgency < 0.7) else 0.2
            weight_speed = 0.3 if (isinstance(urgency, (int, float)) and urgency >= 0.7) else 0.2
            weight_carbon = 0.3
            scores[hub] = weight_cost * proximity_score + weight_speed * rand + weight_carbon * self._rand_uniform(0.4, 0.9)
        return max(scores, key=scores.get)

    def _calculate_cost(self, origin: str, hub: str) -> float:
        # fair random cost for demo
        return round(self._rand_uniform(15, 50) * self._rand_uniform(0.8, 1.5), 2)

    def _calculate_transit_time(self, origin: str, hub: str) -> int:
        # integer days 1..5
        return int(self._rand_integers(1, 6))

    def _calculate_carbon(self, origin: str, hub: str) -> float:
        return round(self._rand_uniform(5, 25), 2)

    def _calculate_transport_mode(self, urgency_score: float, transport_cost: float) -> str:
        try:
            if urgency_score >= 0.8:
                return "Air" if transport_cost < 30 else "Road"
            if urgency_score >= 0.5:
                return "Road" if transport_cost < 20 else "Rail"
            return "Rail" if transport_cost < 15 else "Sea"
        except Exception:
            return "Unknown"

    # helpers for RNG abstraction
    def _rand_uniform(self, a: float, b: float) -> float:
        if hasattr(self.rng, "uniform") and not self._using_default_rng:
            # numpy Generator (default_rng) uses .uniform(low, high)
            return float(self.rng.uniform(a, b))
        # fallback to numpy module RNG
        return float(np.random.uniform(a, b))

    def _rand_integers(self, low: int, high: int) -> int:
        if not self._using_default_rng and hasattr(self.rng, "integers"):
            return int(self.rng.integers(low, high))
        return int(np.random.randint(low, high))


class GeminiAIAssistant:
    """Wrapper for Gemini-like API with safe parsing and mock fallback"""

    def __init__(self, api_key: Optional[str] = None, endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"):
        self.api_key = api_key
        self.endpoint = endpoint

    def get_suggestions(self, context: Dict) -> Dict:
        # If no API key provided, use mock suggestions
        if not self.api_key or self.api_key == "your-gemini-api-key-here":
            return self._mock_suggestions(context)
        try:
            prompt = self._build_prompt(context)
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(f"{self.endpoint}?key={self.api_key}", headers=headers, json=data, timeout=10)
            if response.status_code != 200:
                logger.warning("Gemini API returned non-200 status (%s). Falling back to mock.", response.status_code)
                return self._mock_suggestions(context)
            result = response.json()
            text = self._safely_extract_text_from_gemini_result(result)
            if not text:
                return self._mock_suggestions(context)
            parsed = self._parse_response(text)
            # If parsing produced no items, fallback
            if not any(parsed.values()):
                return self._mock_suggestions(context)
            return parsed
        except Exception as exc:
            logger.exception("Gemini request failed: %s", exc)
            return self._mock_suggestions(context)

    def _build_prompt(self, context: Dict) -> str:
        return (
            "Analyze the following reverse logistics data and provide actionable insights:\n\n"
            f"Total Returns: {context.get('total_returns', 0)}\n"
            f"Average Recovery Rate: {context.get('avg_recovery', 0):.1%}\n"
            f"Total Carbon Footprint: {context.get('total_carbon', 0):.1f} kg CO2\n"
            f"High Confidence Dispositions: {context.get('high_confidence', 0)}\n"
            f"Items Needing Manual Review: {context.get('manual_review', 0)}\n\n"
            "Provide 3 specific, actionable recommendations to:\n"
            "1. Improve disposition accuracy and reduce manual reviews\n"
            "2. Optimize routing efficiency and reduce costs\n"
            "3. Minimize environmental impact\n\n"
            "Format as: RECOMMENDATION: [title] | DETAIL: [explanation] | IMPACT: [expected improvement]\n"
        )

    def _safely_extract_text_from_gemini_result(self, result: Dict) -> str:
        # defensive extraction from nested content
        try:
            if not isinstance(result, dict):
                return ""
            candidates = result.get("candidates") or result.get("candidate") or []
            if not candidates:
                return ""
            first = candidates[0]
            if isinstance(first, dict):
                # maybe result['candidates'][0]['content']['parts'][0]['text']
                content = first.get("content") or {}
                parts = content.get("parts") or []
                if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                    return str(parts[0].get("text", "")).strip()
                # fallback: maybe candidate itself has text
                if "text" in first:
                    return str(first.get("text", "")).strip()
            return ""
        except Exception:
            logger.exception("Error extracting text from Gemini result.")
            return ""

    def _parse_response(self, text: str) -> Dict:
        suggestions = {"disposition_optimization": [], "routing_efficiency": [], "environmental_impact": []}
        try:
            lines = text.split("\n")
            for line in lines:
                if "RECOMMENDATION:" in line.upper():
                    parts = [p.strip() for p in line.split("|")]
                    title, detail, impact = "", "", ""
                    for part in parts:
                        up = part.upper()
                        if up.startswith("RECOMMENDATION:"):
                            title = part.split(":", 1)[1].strip()
                        elif up.startswith("DETAIL:"):
                            detail = part.split(":", 1)[1].strip()
                        elif up.startswith("IMPACT:"):
                            impact = part.split(":", 1)[1].strip()
                    rec = {"title": title or "Recommendation", "detail": detail, "impact": impact}
                    t = rec["title"].lower()
                    if "disposition" in t or "review" in t or "grade" in t:
                        suggestions["disposition_optimization"].append(rec)
                    elif "routing" in t or "cost" in t or "route" in t:
                        suggestions["routing_efficiency"].append(rec)
                    else:
                        suggestions["environmental_impact"].append(rec)
        except Exception:
            logger.exception("Failed to parse Gemini response text.")
        return suggestions

    def _mock_suggestions(self, context: Dict) -> Dict:
        total_returns = max(int(context.get("total_returns", 1)), 1)
        manual_review_rate = (context.get("manual_review", 0) / total_returns) if total_returns else 0.0
        return {
            "disposition_optimization": [
                {
                    "title": "Implement ML-Enhanced Image Recognition",
                    "detail": f"With {manual_review_rate:.1%} items requiring manual review, deploy computer vision to auto-grade product conditions",
                    "impact": "Reduce manual reviews by 40-60%, accelerate processing by 2-3 days",
                },
                {
                    "title": "Dynamic Threshold Adjustment",
                    "detail": "Adjust confidence thresholds based on product category and historical accuracy",
                    "impact": "Improve disposition accuracy by 15-20% while maintaining efficiency",
                },
            ],
            "routing_efficiency": [
                {
                    "title": "Consolidation Hub Strategy",
                    "detail": f"Current average cost is ${context.get('avg_cost', 0):.2f}. Implement micro-consolidation centers for low-density flows",
                    "impact": "Reduce per-unit transport costs by 25-35%",
                },
                {
                    "title": "Predictive Demand Routing",
                    "detail": "Route high-value returns to hubs with higher refurbishment demand forecasts",
                    "impact": "Increase recovery value by 10-15% through faster remarketing",
                },
            ],
            "environmental_impact": [
                {
                    "title": "Carbon-Aware Route Selection",
                    "detail": f"Current footprint: {context.get('total_carbon', 0):.1f} kg CO2. Prioritize rail/consolidated shipping for non-urgent returns",
                    "impact": "Reduce carbon emissions by 30-40% with <5% cost increase",
                },
                {
                    "title": "Local Refurbishment Network",
                    "detail": "Establish regional refurbishment partnerships to minimize long-haul transportation",
                    "impact": "Cut average transit distance by 50%, reducing both cost and emissions",
                },
            ],
        }


# ---- Helpers ----
def generate_sample_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    conditions = ["Unopened", "Like New", "Minor Defect", "Major Defect", "Damaged"]
    categories = ["Electronics", "Apparel", "Home Goods", "Toys", "Books"]
    reasons = ["Wrong Item", "Defective", "Changed Mind", "Not as Described", "No Longer Needed"]
    locations = list(INDIAN_LOCATIONS_COORDS.keys())
    data = {
        "return_id": [f"RET-{1000+i}" for i in range(n)],
        "product_category": np.random.choice(categories, n),
        "condition": np.random.choice(conditions, n, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        "return_reason": np.random.choice(reasons, n),
        "original_value": np.random.uniform(20, 500, n).round(2),
        "origin_location": np.random.choice(locations, n),
        "return_date": [datetime.now() - timedelta(days=int(np.random.randint(0, 30))) for _ in range(n)],
        "urgency_score": np.random.uniform(0.3, 1.0, n).round(2),
    }
    return pd.DataFrame(data)


def process_returns(df: pd.DataFrame, classifier: DispositionClassifier, optimizer: ReverseRoutingOptimizer) -> Dict[str, Any]:
    """
    Processes returns using classifier (dispositions) and optimizer (routing).
    Returns dict: {'processed_df': pd.DataFrame, 'warnings': List[str]}
    """
    warnings = []
    results = []
    df = df.copy().reset_index(drop=True)

    for _, row in df.iterrows():
        # Basic required fields
        if all(pd.notna(row.get(col)) for col in ["condition", "product_category", "original_value", "return_reason"]):
            disposition, confidence, metadata = classifier.classify_return(
                row["condition"], row["product_category"], row["original_value"], row["return_reason"]
            )
        else:
            disposition = "Dispose"
            confidence = 0.0
            metadata = {
                "confidence": 0.0,
                "all_scores": {"Dispose": 1.0},
                "expected_recovery": 0.0,
                "processing_days": 2,
                "needs_manual_review": True,
            }
            warnings.append(f"Skipping classification for return_id {row.get('return_id', 'N/A')} due to missing data.")

        result = row.to_dict()
        result["ai_disposition"] = disposition
        result["confidence"] = float(confidence)
        result["expected_recovery"] = metadata["expected_recovery"]
        result["processing_days"] = metadata["processing_days"]
        result["needs_review"] = bool(metadata["needs_manual_review"])
        results.append(result)

    processed_df = pd.DataFrame(results)
    processed_df = optimizer.optimize_routes(processed_df)

    # Post-process: ensure types
    numeric_cols = ["original_value", "expected_recovery", "transport_cost", "carbon_kg", "confidence"]
    for c in numeric_cols:
        if c in processed_df.columns:
            processed_df[c] = pd.to_numeric(processed_df[c], errors="coerce")

    return {"processed_df": processed_df, "warnings": warnings}


# ---- UI / Main ----
def main():
    st.markdown('<h1 class="main-header">🤖 AI Reverse Logistics Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Disposition & Dynamic Routing Optimization**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration & Data")
        api_key = st.text_input("Gemini API Key (optional)", value=os.getenv("GEMINI_API_KEY", ""))
        seed_checkbox = st.checkbox("Use deterministic seed for routing (helpful for demos/tests)", value=False)
        seed_val = None
        if seed_checkbox:
            seed_val = st.number_input("Seed (integer)", min_value=0, value=42, step=1)
        n_returns = st.slider("Number of Returns (for generated data)", 50, 500, 100, 50)
        uploaded_file = st.file_uploader("Upload Returns CSV", type=["csv"])
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)

    # Session state initialization
    if "returns_data" not in st.session_state:
        st.session_state.returns_data = None
    if "ai_suggestions" not in st.session_state:
        st.session_state.ai_suggestions = {}

    # Load or generate data
    if uploaded_file is not None:
        try:
            st.session_state.returns_data = pd.read_csv(uploaded_file)
            st.success("CSV uploaded successfully.")
        except Exception as exc:
            st.error(f"Error reading CSV: {exc}")
            st.session_state.returns_data = None
    elif st.session_state.returns_data is None:
        st.session_state.returns_data = generate_sample_data(n_returns)

    # Instantiate agents
    classifier = DispositionClassifier(confidence_threshold=confidence_threshold)
    optimizer = ReverseRoutingOptimizer(seed=int(seed_val) if seed_val is not None else None)
    gemini = GeminiAIAssistant(api_key=api_key if api_key else None)

    # Process
    with st.spinner("Processing returns through AI systems..."):
        out = process_returns(st.session_state.returns_data, classifier, optimizer)
    processed_data: pd.DataFrame = out["processed_df"]
    warnings = out["warnings"]

    if warnings:
        # show up to first 6 warnings to avoid UI spam
        display_text = "\n".join(warnings[:6])
        if len(warnings) > 6:
            display_text += f"\n...and {len(warnings)-6} more warnings."
        st.warning(display_text)

    if processed_data.empty:
        st.info("No return data available to display. Please upload a valid CSV or generate new data.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Returns", len(processed_data), delta=f"{len(processed_data[processed_data['needs_review'] == False])} auto-processed")
    with col2:
        total_original = processed_data["original_value"].sum() if "original_value" in processed_data.columns else 0
        avg_recovery = (processed_data["expected_recovery"].sum() / total_original) if total_original != 0 else 0.0
        st.metric("Avg Recovery Rate", f"{avg_recovery:.1%}", delta=f"${processed_data['expected_recovery'].sum():,.0f} total")
    with col3:
        avg_confidence = processed_data["confidence"].mean() if "confidence" in processed_data.columns else 0.0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}", delta=f"{(processed_data['confidence'] >= confidence_threshold).sum()} high confidence")
    with col4:
        total_carbon = processed_data["carbon_kg"].sum() if "carbon_kg" in processed_data.columns else 0.0
        st.metric("Carbon Footprint", f"{total_carbon:.0f} kg CO₂", delta=f"{(processed_data['carbon_kg'].mean() if 'carbon_kg' in processed_data.columns else 0.0):.1f} kg avg")

    # Tabs
    tab1, tab2, tab_map, tab3, tab4 = st.tabs(
        ["📊 Disposition Analysis", "🚚 Routing Optimization", "🌍 India Map", "🤖 AI Insights", "📋 Data Table"]
    )

    # Tab: Disposition Analysis
    with tab1:
        st.subheader("Disposition Bottleneck Resolution — Expanded Insights")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            disp_counts = processed_data["ai_disposition"].value_counts()
            fig_disp = go.Figure(data=[go.Pie(labels=disp_counts.index, values=disp_counts.values, hole=0.4)])
            fig_disp.update_layout(title="Disposition Distribution", height=380)
            st.plotly_chart(fig_disp, use_container_width=True)

            st.markdown("**Top Return Reasons by Expected Recovery (value at risk)**")
            reasons_rank = (
                processed_data.groupby("return_reason")
                .agg({"expected_recovery": "sum"})
                .reset_index()
                .sort_values("expected_recovery", ascending=False)
            )
            st.table(reasons_rank.head(5).assign(expected_recovery=lambda df: df["expected_recovery"].map("${:,.2f}".format)))

            st.markdown("**Product Categories with most manual reviews**")
            cat_reviews = processed_data.groupby("product_category").agg({"needs_review": ["sum", "count"]})
            cat_reviews.columns = ["manual_reviews", "total"]
            cat_reviews = cat_reviews.reset_index()
            cat_reviews["review_rate"] = cat_reviews["manual_reviews"] / cat_reviews["total"]
            st.table(cat_reviews.sort_values("manual_reviews", ascending=False).head(6).assign(review_rate=lambda df: (df["review_rate"] * 100).round(1).astype(str) + "%"))

        with col_b:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            manual_review_count = int(processed_data["needs_review"].sum())
            auto_process_count = len(processed_data) - manual_review_count
            avg_proc_days = processed_data["processing_days"].mean()
            st.markdown(
                f"**Processing Efficiency**\n- ✅ Auto-processed: {auto_process_count} ({auto_process_count/len(processed_data)*100:.1f}%)\n"
                f"- 👤 Manual review needed: {manual_review_count} ({manual_review_count/len(processed_data)*100:.1f}%)\n"
                f"- ⚡ Avg processing time: {avg_proc_days:.1f} days"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            value_at_risk = processed_data[processed_data["needs_review"] == True]["original_value"].sum()
            st.markdown(
                f"**Value Metrics**\n- 💰 Total value at risk: ${value_at_risk:,.2f}\n"
                f"- 🎯 High-confidence items: {(processed_data['confidence'] >= confidence_threshold).sum()}\n"
                f"- 📉 Avg confidence: {processed_data['confidence'].mean():.1%}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Recovery Performance by Condition")
        recovery_by_cond = processed_data.groupby("condition").agg({"expected_recovery": "sum", "original_value": "sum"}).reset_index()
        recovery_by_cond["recovery_rate"] = recovery_by_cond["expected_recovery"] / recovery_by_cond["original_value"]
        fig_rc = px.bar(recovery_by_cond, x="condition", y="recovery_rate", title="Recovery Rate by Condition", labels={"recovery_rate": "Recovery Rate"})
        fig_rc.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_rc, use_container_width=True)

    # Tab: Routing Optimization
    with tab2:
        st.subheader("Routing Optimization — Expanded Insights")
        hub_stats = processed_data.groupby("optimal_hub").agg({"return_id": "count", "transport_cost": "sum", "carbon_kg": "sum", "transit_days": "mean"}).reset_index()
        hub_stats.columns = ["Hub", "Returns", "Total Cost", "Total Carbon", "Avg Transit Days"]
        st.markdown("**Hub Summary**")
        st.dataframe(hub_stats)

        st.markdown("**Transport Mode Distribution (overall)**")
        mode_counts = processed_data["transport_mode"].value_counts().reset_index()
        mode_counts.columns = ["mode", "count"]
        fig_mode = px.pie(mode_counts, names="mode", values="count", title="Transport Mode Split")
        st.plotly_chart(fig_mode, use_container_width=True)

        st.markdown("**Top 5 Most Expensive Routes (by transport cost)**")
        expensive = processed_data.sort_values("transport_cost", ascending=False).head(5)[["return_id", "origin_location", "optimal_hub", "transport_cost", "transport_mode"]]
        st.table(expensive)

        st.markdown("**Top 5 Highest Carbon Routes**")
        high_carbon = processed_data.sort_values("carbon_kg", ascending=False).head(5)[["return_id", "origin_location", "optimal_hub", "carbon_kg", "transport_mode"]]
        st.table(high_carbon)

    # Tab: Map
    with tab_map:
        st.subheader("🌍 Optimized Routes (map) — Routes + Transport Mode Only")
        df_map = processed_data.dropna(subset=["origin_lat", "origin_lon", "hub_lat", "hub_lon"]).copy()
        if df_map.empty:
            st.info("No routes to display on the map.")
        else:
            mode_colors = {"Air": "red", "Road": "orange", "Rail": "green", "Sea": "blue", "Unknown": "gray"}
            fig_map = go.Figure()
            for _, row in df_map.iterrows():
                fig_map.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lon=[row["origin_lon"], row["hub_lon"]],
                        lat=[row["origin_lat"], row["hub_lat"]],
                        line=dict(width=2, color=mode_colors.get(row["transport_mode"], "gray")),
                        hoverinfo="text",
                        text=f"{row['return_id']}: {row['origin_location']} → {row['optimal_hub']}<br>Mode: {row['transport_mode']}<br>Cost: ${row['transport_cost']:.2f}<br>Transit days: {row['transit_days']}<br>Carbon: {row['carbon_kg']:.2f} kg",
                        showlegend=False,
                    )
                )
            origin_modes = (
                df_map.groupby(["origin_location", "origin_lat", "origin_lon"])
                .agg({"transport_mode": lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown", "transport_cost": "mean", "carbon_kg": "mean", "return_id": "count"})
                .reset_index()
            )
            fig_map.add_trace(
                go.Scattermapbox(
                    mode="markers+text",
                    lon=origin_modes["origin_lon"],
                    lat=origin_modes["origin_lat"],
                    marker=dict(size=12, color=[mode_colors.get(m, "gray") for m in origin_modes["transport_mode"]]),
                    text=origin_modes["transport_mode"],
                    textposition="top center",
                    hoverinfo="text",
                    textfont=dict(size=10),
                    name="Origin (mode)",
                )
            )
            hub_points_df = pd.DataFrame(HUB_COORDS).T.reset_index()
            hub_points_df.columns = ["Location", "lat", "lon"]
            active_hubs = df_map["optimal_hub"].dropna().unique().tolist()
            filtered_hub_points_df = hub_points_df[hub_points_df["Location"].isin(active_hubs)]
            fig_map.add_trace(
                go.Scattermapbox(
                    mode="markers+text",
                    lon=filtered_hub_points_df["lon"],
                    lat=filtered_hub_points_df["lat"],
                    marker=dict(size=14, color="black", symbol="star"),
                    text=filtered_hub_points_df["Location"],
                    textposition="bottom center",
                    hoverinfo="text",
                    name="Hubs",
                )
            )
            fig_map.update_layout(mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center={"lat": 22.35, "lon": 78.66}, height=700, margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

    # Tab: AI Insights
    with tab3:
        st.subheader("🤖 AI-Powered Insights & Recommendations")
        if st.button("🔮 Generate AI Suggestions"):
            with st.spinner("Consulting Gemini AI..."):
                context = {
                    "total_returns": len(processed_data),
                    "avg_recovery": (processed_data["expected_recovery"].sum() / processed_data["original_value"].sum()) if processed_data["original_value"].sum() != 0 else 0.0,
                    "total_carbon": processed_data["carbon_kg"].sum() if "carbon_kg" in processed_data.columns else 0.0,
                    "high_confidence": (processed_data["confidence"] >= confidence_threshold).sum() if "confidence" in processed_data.columns else 0,
                    "manual_review": int(processed_data["needs_review"].sum()) if "needs_review" in processed_data.columns else 0,
                    "avg_cost": processed_data["transport_cost"].mean() if "transport_cost" in processed_data.columns else 0.0,
                }
                suggestions = gemini.get_suggestions(context)
                st.session_state.ai_suggestions = suggestions

        if st.session_state.ai_suggestions:
            suggestions = st.session_state.ai_suggestions
            st.markdown("### 🎯 Disposition Optimization")
            for i, suggestion in enumerate(suggestions.get("disposition_optimization", []), 1):
                with st.expander(f"💡 {suggestion.get('title', 'Suggestion')}", expanded=(i == 1)):
                    st.markdown(f"**Details:** {suggestion.get('detail', '')}")
                    st.markdown(f"**Expected Impact:** {suggestion.get('impact', '')}")
            st.markdown("### 🚚 Routing Efficiency")
            for i, suggestion in enumerate(suggestions.get("routing_efficiency", []), 1):
                with st.expander(f"💡 {suggestion.get('title', 'Suggestion')}", expanded=(i == 1)):
                    st.markdown(f"**Details:** {suggestion.get('detail', '')}")
                    st.markdown(f"**Expected Impact:** {suggestion.get('impact', '')}")
            st.markdown("### 🌱 Environmental Impact")
            for i, suggestion in enumerate(suggestions.get("environmental_impact", []), 1):
                with st.expander(f"💡 {suggestion.get('title', 'Suggestion')}", expanded=(i == 1)):
                    st.markdown(f"**Details:** {suggestion.get('detail', '')}")
                    st.markdown(f"**Expected Impact:** {suggestion.get('impact', '')}")
        else:
            st.info("Click 'Generate AI Suggestions' to get personalized recommendations from Gemini AI")

    # Tab: Data Table + Filters + Download
    with tab4:
        st.subheader("📋 Detailed Returns Data")
        # safe options (dropna -> avoid nan in defaults)
        disposition_options = processed_data["ai_disposition"].dropna().unique().tolist()
        category_options = processed_data["product_category"].dropna().unique().tolist()
        reason_options = processed_data["return_reason"].dropna().unique().tolist()
        hub_options = processed_data["optimal_hub"].dropna().unique().tolist()

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            disposition_filter = st.multiselect("Filter by Disposition", options=disposition_options, default=disposition_options)
        with col2:
            category_filter = st.multiselect("Filter by Product Category", options=category_options, default=category_options)
        with col3:
            reason_filter = st.multiselect("Filter by Return Reason", options=reason_options, default=reason_options)
        with col4:
            hub_filter = st.multiselect("Filter by Optimal Hub", options=hub_options, default=hub_options)
        with col5:
            min_val, max_val = float(processed_data["original_value"].min()), float(processed_data["original_value"].max())
            value_range = st.slider("Original Value Range ($)", min_value=min_val, max_value=max_val, value=(min_val, max_val))
        with col6:
            min_urgency, max_urgency = float(processed_data["urgency_score"].min()), float(processed_data["urgency_score"].max())
            urgency_range = st.slider("Urgency Score Range", min_value=min_urgency, max_value=max_urgency, value=(min_urgency, max_urgency), step=0.05)
        with col7:
            review_filter = st.checkbox("Show only items needing manual review")

        all_columns = processed_data.columns.tolist()
        default_columns = [
            "return_id",
            "product_category",
            "condition",
            "return_reason",
            "original_value",
            "origin_location",
            "ai_disposition",
            "confidence",
            "expected_recovery",
            "needs_review",
            "optimal_hub",
            "transport_cost",
            "transit_days",
            "carbon_kg",
            "transport_mode",
        ]
        selected_columns = st.multiselect("Select Columns to Display", options=all_columns, default=default_columns)

        # apply filters
        filtered_data = (
            processed_data[
                processed_data["ai_disposition"].isin(disposition_filter)
                & processed_data["product_category"].isin(category_filter)
                & processed_data["return_reason"].isin(reason_filter)
                & processed_data["optimal_hub"].isin(hub_filter)
                & (processed_data["original_value"] >= value_range[0])
                & (processed_data["original_value"] <= value_range[1])
                & (processed_data["urgency_score"] >= urgency_range[0])
                & (processed_data["urgency_score"] <= urgency_range[1])
            ]
            .copy()
        )
        if review_filter:
            filtered_data = filtered_data[filtered_data["needs_review"] == True]

        if not filtered_data.empty and selected_columns:
            st.dataframe(filtered_data[selected_columns])
            csv = filtered_data[selected_columns].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download filtered CSV", data=csv, file_name="processed_returns_filtered.csv", mime="text/csv")
        elif not filtered_data.empty and not selected_columns:
            st.warning("Please select at least one column to display.")
        else:
            st.info("No data matches the current filter selection.")

    # End main


if __name__ == "__main__":
    main()
