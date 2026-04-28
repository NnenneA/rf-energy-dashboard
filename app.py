import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import warnings
import os
import gdown
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RF Energy Harvestability — London",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    h1 { color: #1A2E5A; }
    h2 { color: #2E4A7A; }
    h3 { color: #2E4A7A; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_main_data():
    file_id = "1LVLDCRkv-YMFakdl3l2DQ7WKfwTwmDAE"
    output = "london_4g_combined.csv"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    df = pd.read_csv(output, low_memory=False)
    signal_cols = ['rsrp_top1_3uk', 'rsrp_top1_ee', 'rsrp_top1_o2', 'rsrp_top1_vf']
    for col in signal_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    df['hour_of_day'] = pd.to_datetime(df['hour_ref'], unit='s', utc=True).dt.hour
    df['month'] = pd.to_datetime(df['month_year'], dayfirst=True).dt.month
    df['year'] = pd.to_datetime(df['month_year'], dayfirst=True).dt.year
    df['harvestable'] = (
        (df['rsrp_top1_3uk'].notna() & (df['rsrp_top1_3uk'] >= -40)) |
        (df['rsrp_top1_ee'].notna()  & (df['rsrp_top1_ee']  >= -40)) |
        (df['rsrp_top1_o2'].notna()  & (df['rsrp_top1_o2']  >= -40)) |
        (df['rsrp_top1_vf'].notna()  & (df['rsrp_top1_vf']  >= -40))
    ).astype(int)
    return df

@st.cache_data
def load_results():
    file_id = "1PsX2Ic5Hw1g1jctN46ma-vdd3YF0bKwc"
    output = "r_full_results.csv"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return pd.read_csv(output)

@st.cache_data
def precompute_thresholds(_df):
    """Pre-calculate harvestability counts for all threshold levels once at startup."""
    thresholds = [-60, -55, -50, -45, -40, -35, -30, -25, -20]
    results = {}
    for t in thresholds:
        results[t] = int((
            (_df['rsrp_top1_3uk'].notna() & (_df['rsrp_top1_3uk'] >= t)) |
            (_df['rsrp_top1_ee'].notna()  & (_df['rsrp_top1_ee']  >= t)) |
            (_df['rsrp_top1_o2'].notna()  & (_df['rsrp_top1_o2']  >= t)) |
            (_df['rsrp_top1_vf'].notna()  & (_df['rsrp_top1_vf']  >= t))
        ).sum())
    return results

@st.cache_data
def precompute_map_sample(_df):
    """Pre-sample map points once at startup with harvestable flag at all thresholds."""
    thresholds = [-60, -55, -50, -45, -40, -35, -30, -25, -20]
    df_map = _df.dropna(subset=['latitude', 'longitude']).copy()
    df_map = df_map.sample(min(5000, len(df_map)), random_state=42)
    for t in thresholds:
        col_name = f'harvest_{abs(t)}'
        df_map[col_name] = (
            (df_map['rsrp_top1_3uk'].notna() & (df_map['rsrp_top1_3uk'] >= t)) |
            (df_map['rsrp_top1_ee'].notna()  & (df_map['rsrp_top1_ee']  >= t)) |
            (df_map['rsrp_top1_o2'].notna()  & (df_map['rsrp_top1_o2']  >= t)) |
            (df_map['rsrp_top1_vf'].notna()  & (df_map['rsrp_top1_vf']  >= t))
        ).astype(int)
    return df_map

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://cdn.ymaws.com/elia.site-ym.com/resource/resmgr/news_items/university_east_london_logo.png",
    width=180
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 RF Energy Harvestability")
st.sidebar.markdown("**From Passive to Predictive Harvesting**")
st.sidebar.markdown("*MSc Data Science DS7010*")
st.sidebar.markdown("*Nnenne Opemipo Agwunedu*")
st.sidebar.markdown("*University of East London*")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select Page",
    [" Overview", " Geographic Map", " Temporal Analysis",
     " Model Performance", " Deployment Recommendations"]
)

# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading dataset..."):
    try:
        df = load_main_data()
        results = load_results()
        threshold_counts = precompute_thresholds(df)
        df_map_precomputed = precompute_map_sample(df)
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == " Overview":
    st.title("📡 RF Energy Harvestability Dashboard")
    st.markdown("**Predicting ambient RF energy availability for wireless sensor networks across London**")
    st.markdown("---")

    if data_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Measurements", f"{len(df):,}", "Ofcom 4G LTE 2024-2025")
        with col2:
            hc = df['harvestable'].sum()
            st.metric("Harvestable Locations", f"{hc:,}", f"{hc/len(df)*100:.2f}% of total")
        with col3:
            st.metric("Operators", "4", "Three UK, EE, O2, Vodafone")
        with col4:
            st.metric("Best Model", "XGBoost", "F1 = 0.2769 | PR-AUC = 0.1739")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### About This Research")
            st.markdown("""
            This dashboard presents findings from a machine learning study 
            predicting ambient RF energy harvestability across Greater London 
            using Ofcom's open 4G LTE drive-test dataset.

            A location is classified as harvestable if any operator records 
            RSRP ≥ −40 dBm, the activation threshold derived from prior 
            rectenna circuit research (Agwunedu, 2022).

            **Four models were evaluated:**
            - Random Forest
            - XGBoost *(best performing)*
            - Support Vector Machine
            - Long Short-Term Memory (LSTM)
            """)

        with col2:
            st.markdown("### Key Findings")
            class_data = pd.DataFrame({
                'Class': ['Not Harvestable', 'Harvestable'],
                'Count': [len(df) - df['harvestable'].sum(), df['harvestable'].sum()]
            })
            fig = px.bar(class_data, x='Class', y='Count', color='Class',
                         color_discrete_map={'Not Harvestable': '#F44336', 'Harvestable': '#4CAF50'},
                         title='Harvestability Class Balance', text='Count')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=350, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### RSRP Descriptive Statistics by Operator")

        rsrp_cols = {'Three UK': 'rsrp_top1_3uk', 'EE': 'rsrp_top1_ee',
                     'O2': 'rsrp_top1_o2', 'Vodafone': 'rsrp_top1_vf'}
        stats_data = []
        for op, col in rsrp_cols.items():
            if col in df.columns:
                stats_data.append({
                    'Operator': op,
                    'Mean (dBm)': round(df[col].mean(), 2),
                    'Median (dBm)': round(df[col].median(), 2),
                    'SD (dBm)': round(df[col].std(), 2),
                    'Min (dBm)': round(df[col].min(), 2),
                    'Max (dBm)': round(df[col].max(), 2),
                    '% Below -40 dBm': round((df[col] < -40).mean() * 100, 2)
                })
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2: GEOGRAPHIC MAP
# ============================================================

elif page == " Geographic Map":
    st.title(" Geographic Distribution of RF Harvestability")
    st.markdown("Spatial distribution of harvestable locations across London's road network.")
    st.markdown("---")

    if data_loaded:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("### Harvestability Threshold")
            threshold_dbm = st.slider(
                "RSRP Threshold (dBm)",
                min_value=-60, max_value=-20, value=-40, step=5,
                help="Move to explore how harvestability changes at different signal thresholds. The study uses -40 dBm."
            )

            harvestable_at_threshold = threshold_counts.get(threshold_dbm, 0)
            harvestable_pct = harvestable_at_threshold / len(df) * 100

            if threshold_dbm == -40:
                st.info(f"**Study threshold:** {threshold_dbm} dBm\n\n{harvestable_at_threshold:,} harvestable ({harvestable_pct:.2f}%)")
            elif threshold_dbm < -40:
                st.success(f"**Relaxed threshold:** {threshold_dbm} dBm\n\n{harvestable_at_threshold:,} harvestable ({harvestable_pct:.2f}%)")
            else:
                st.warning(f"**Strict threshold:** {threshold_dbm} dBm\n\n{harvestable_at_threshold:,} harvestable ({harvestable_pct:.2f}%)")

            st.markdown("---")
            st.caption("−40 dBm is the rectenna activation point (Agwunedu, 2022). Map shows a 5,000-point sample.")

        with col2:
            col_name = f'harvest_{abs(threshold_dbm)}'
            harvestable_pts = df_map_precomputed[df_map_precomputed[col_name] == 1]
            not_harvestable_pts = df_map_precomputed[df_map_precomputed[col_name] == 0].sample(
                min(2000, len(df_map_precomputed[df_map_precomputed[col_name] == 0])),
                random_state=42
            )

            m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='CartoDB positron')

            for _, row in not_harvestable_pts.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=2, color='#F44336', fill=True,
                    fill_opacity=0.4, weight=0
                ).add_to(m)

            for _, row in harvestable_pts.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5, color='#4CAF50', fill=True,
                    fill_opacity=0.8, weight=1,
                    popup=f"Harvestable at {threshold_dbm} dBm"
                ).add_to(m)

            legend_html = f'''
            <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                 background-color:white;padding:10px;border-radius:8px;
                 box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:13px;">
                <b>Harvestability at {threshold_dbm} dBm</b><br>
                <span style="color:#4CAF50;">●</span> Harvestable ({len(harvestable_pts):,} pts in sample)<br>
                <span style="color:#F44336;">●</span> Not Harvestable
            </div>'''
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=800, height=550)

        st.markdown("---")
        st.markdown("### Threshold Sensitivity Analysis")
        st.markdown("How harvestability changes across different RSRP thresholds across the full dataset:")

        threshold_table = []
        for t in [-60, -55, -50, -45, -40, -35, -30]:
            count = threshold_counts.get(t, 0)
            threshold_table.append({
                'Threshold (dBm)': t,
                'Harvestable Count': f"{count:,}",
                'Harvestable (%)': f"{count/len(df)*100:.2f}%",
                'Note': '← Study threshold' if t == -40 else ''
            })

        st.dataframe(pd.DataFrame(threshold_table), use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Harvestable at Study Threshold", f"{df['harvestable'].sum():,}",
                      f"{df['harvestable'].mean()*100:.2f}% of dataset")
        with col2:
            st.metric("Strongest Mean Signal", "Vodafone", "-71.36 dBm average RSRP")
        with col3:
            st.metric("Most Restricted", "O2", "99.95% below -40 dBm threshold")

# ============================================================
# PAGE 3: TEMPORAL ANALYSIS
# ============================================================

elif page == " Temporal Analysis":
    st.title(" Temporal Patterns in RF Signal Strength")
    st.markdown("Signal variation across hours of the day, months, and years.")
    st.markdown("---")

    if data_loaded:
        st.markdown("### Mean RSRP by Hour of Day")

        rsrp_cols_ops = {'Three UK': 'rsrp_top1_3uk', 'EE': 'rsrp_top1_ee',
                         'O2': 'rsrp_top1_o2', 'Vodafone': 'rsrp_top1_vf'}

        hourly_data = []
        for op, col in rsrp_cols_ops.items():
            if col in df.columns:
                hourly = df.groupby('hour_of_day')[col].mean().reset_index()
                hourly.columns = ['Hour', 'Mean RSRP']
                hourly['Operator'] = op
                hourly_data.append(hourly)

        hourly_df = pd.concat(hourly_data)
        fig_hourly = px.line(hourly_df, x='Hour', y='Mean RSRP', color='Operator',
            title='Mean RSRP by Hour of Day — All Operators',
            color_discrete_map={'Three UK': '#2196F3', 'EE': '#FF9800',
                                'O2': '#4CAF50', 'Vodafone': '#E91E63'}, markers=True)
        fig_hourly.add_hline(y=-40, line_dash="dash", line_color="red",
            annotation_text="-40 dBm threshold", annotation_position="right")
        fig_hourly.update_layout(height=450, plot_bgcolor='white',
            xaxis=dict(tickmode='array', tickvals=sorted(df['hour_of_day'].unique()),
                       title='Hour of Day'), yaxis_title='Mean RSRP (dBm)')
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.info("⚠️ Hours 7 and 14 are absent from the dataset due to Ofcom's drive-test scheduling.")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Measurement Count by Hour")
            hour_counts = df['hour_of_day'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Hour', 'Count']
            fig_counts = px.bar(hour_counts, x='Hour', y='Count',
                title='Measurement Count by Hour of Day',
                color_discrete_sequence=['#2196F3'])
            fig_counts.update_layout(height=350, plot_bgcolor='white')
            st.plotly_chart(fig_counts, use_container_width=True)

        with col2:
            st.markdown("### Harvestability Rate by Hour")
            harvest_hour = df.groupby('hour_of_day')['harvestable'].mean().reset_index()
            harvest_hour.columns = ['Hour', 'Harvestability Rate']
            harvest_hour['Harvestability Rate'] = harvest_hour['Harvestability Rate'] * 100
            fig_harvest_hour = px.bar(harvest_hour, x='Hour', y='Harvestability Rate',
                title='Harvestability Rate (%) by Hour',
                color_discrete_sequence=['#4CAF50'])
            fig_harvest_hour.update_layout(height=350, plot_bgcolor='white',
                yaxis_title='Harvestability Rate (%)')
            st.plotly_chart(fig_harvest_hour, use_container_width=True)

        st.markdown("---")
        st.markdown("### RSRP Distribution by Operator")
        operator_filter = st.multiselect("Select Operators",
            ["Three UK", "EE", "O2", "Vodafone"],
            default=["Three UK", "EE", "O2", "Vodafone"])

        colors = {'Three UK': '#2196F3', 'EE': '#FF9800',
                  'O2': '#4CAF50', 'Vodafone': '#E91E63'}
        fig_dist = go.Figure()
        for op in operator_filter:
            col = rsrp_cols_ops[op]
            if col in df.columns:
                fig_dist.add_trace(go.Histogram(x=df[col].dropna(), name=op,
                    opacity=0.7, nbinsx=80, marker_color=colors[op]))
        fig_dist.add_vline(x=-40, line_dash="dash", line_color="red",
            annotation_text="-40 dBm threshold", annotation_position="top right")
        fig_dist.update_layout(barmode='overlay', title='RSRP Distribution by Operator',
            xaxis_title='RSRP (dBm)', yaxis_title='Count',
            height=400, plot_bgcolor='white')
        st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

elif page == " Model Performance":
    st.title(" Model Performance Comparison")
    st.markdown("Comparative evaluation of all four machine learning models under time-based split.")
    st.markdown("---")

    if data_loaded:
        st.markdown("### Model Results: Time-Based Split (R Implementation)")

        model_results = pd.DataFrame({
            'Model': ['RF Baseline', 'RF Tuned (ntree=200, mtry=7)',
                      'XGBoost Baseline', 'XGBoost Tuned',
                      'SVM Baseline', 'SVM Tuned (gamma=1, cost=1)',
                      'SVM Linear Kernel', 'LSTM (all configs)'],
            'Precision': [1.0000, 1.0000, 1.0000, 1.0000, '—', '—', 0.2928, 0.0083],
            'Recall':    [0.0514, 0.1574, 0.1607, 0.1607, 0.0000, 0.0000, 0.0495, 1.0000],
            'F1':        [0.0977, 0.2720, 0.2769, 0.2769, '—', '—', 0.0847, 0.0165],
            'ROC-AUC':   [0.5807, 0.5663, 0.5803, 0.5803, 0.5023, 0.8344, 0.5681, '0.46–0.52'],
            'PR-AUC':    [0.1717, 0.1700, 0.1739, 0.1739, 0.0133, 0.0334, 0.0440, '—'],
            'Kappa':     [0.0970, 0.2703, 0.2752, 0.2752, 0.0000, 0.0000, 0.0825, 0.0000]
        })

        def highlight_best(row):
            if 'XGBoost Tuned' in str(row['Model']):
                return ['background-color: #E8F5E9'] * len(row)
            return [''] * len(row)

        st.dataframe(model_results.style.apply(highlight_best, axis=1),
                     use_container_width=True, hide_index=True)
        st.caption("— indicates undefined metric (model predicted no positive cases). SVM Linear Kernel reflects extended tuning following supervisor feedback.")
        st.success("✅ XGBoost Tuned is the best performing model: F1 = 0.2769, Kappa = 0.2752, PR-AUC = 0.1739 (21× above random baseline)")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            f1_data = pd.DataFrame({
                'Model': ['RF Baseline', 'RF Tuned', 'XGBoost', 'XGBoost Tuned', 'SVM Linear', 'LSTM'],
                'F1': [0.0977, 0.2720, 0.2769, 0.2769, 0.0847, 0.0165]
            })
            fig_f1 = px.bar(f1_data, x='Model', y='F1', title='F1 Score Comparison',
                color='F1', color_continuous_scale='Blues', text='F1')
            fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_f1.update_layout(height=400, plot_bgcolor='white',
                showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_f1, use_container_width=True)

        with col2:
            prauc_data = pd.DataFrame({
                'Model': ['RF Baseline', 'RF Tuned', 'XGBoost', 'XGBoost Tuned',
                          'SVM Baseline', 'SVM Tuned', 'SVM Linear'],
                'PR-AUC': [0.1717, 0.1700, 0.1739, 0.1739, 0.0133, 0.0334, 0.0440]
            })
            fig_prauc = px.bar(prauc_data, x='Model', y='PR-AUC',
                title='PR-AUC Comparison (random baseline = 0.0083)',
                color='PR-AUC', color_continuous_scale='Greens', text='PR-AUC')
            fig_prauc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_prauc.add_hline(y=0.0083, line_dash="dash", line_color="red",
                annotation_text="Random baseline")
            fig_prauc.update_layout(height=400, plot_bgcolor='white',
                showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_prauc, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Confusion Matrix Counts")
            cm_data = pd.DataFrame({
                'Model': ['RF Baseline', 'RF Tuned', 'XGBoost Baseline', 'XGBoost Tuned',
                          'SVM (all configs)', 'SVM Linear Kernel', 'LSTM (all configs)'],
                'TN': [255188, 255188, 255188, 255188, 255188, 254932, 0],
                'FP': [0, 0, 0, 0, 0, 256, 255188],
                'FN': [2031, 1804, 1797, 1797, 2141, 2035, 0],
                'TP': [110, 337, 344, 344, 0, 106, 2141]
            })
            st.dataframe(cm_data, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### True Positives Found")
            st.markdown("*Out of 2,141 harvestable locations in test set*")
            tp_data = pd.DataFrame({
                'Model': ['RF Baseline', 'RF Tuned', 'XGBoost', 'SVM Linear', 'SVM', 'LSTM'],
                'True Positives': [110, 337, 344, 106, 0, 2141]
            })
            fig_tp = px.bar(tp_data, x='Model', y='True Positives',
                title='Harvestable Locations Correctly Identified',
                color='True Positives', color_continuous_scale='RdYlGn', text='True Positives')
            fig_tp.add_hline(y=2141, line_dash="dash", line_color="gray",
                annotation_text="Total harvestable = 2,141")
            fig_tp.update_traces(textposition='outside')
            fig_tp.update_layout(height=350, plot_bgcolor='white', showlegend=False)
            st.plotly_chart(fig_tp, use_container_width=True)

        st.markdown("---")
        st.markdown("### Random Split vs Time-Based Split")
        st.warning("""
        **Random split** produced perfect scores (F1 = 1.0000) because September 2025 
        accounts for 94% of the dataset — training and test sets were near-identical. 
        The **time-based split** is the honest evaluation: train on 2024 and August 2025, 
        test on September 2025. The contrast reveals the true difficulty of generalising 
        to future unseen measurements.
        """)

# ============================================================
# PAGE 5: DEPLOYMENT RECOMMENDATIONS
# ============================================================

elif page == " Deployment Recommendations":
    st.title(" WSN Deployment Recommendations")
    st.markdown("Evidence-based guidance for wireless sensor network deployment across London.")
    st.markdown("---")

    if data_loaded:
        st.markdown("### Five Key Recommendations")

        with st.expander("📍 1. Prioritise areas with stronger measured signal strength (Vodafone and Three UK)", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                Vodafone recorded the highest mean RSRP at **-71.36 dBm** and the lowest 
                percentage of measurements below the -40 dBm threshold at **99.60%**. 
                Three UK followed at **-73.48 dBm**.

                Locations where **multiple operators simultaneously approach -40 dBm** 
                represent the most reliable harvesting opportunities. Deploying at locations 
                where only one operator reaches threshold may be less robust to 
                operator-specific variation over time.
                """)
            with col2:
                op_means = pd.DataFrame({
                    'Operator': ['Vodafone', 'Three UK', 'O2', 'EE'],
                    'Mean RSRP': [-71.36, -73.48, -75.32, -76.44]
                })
                fig_op = px.bar(op_means, x='Operator', y='Mean RSRP', color='Operator',
                    color_discrete_map={'Three UK': '#2196F3', 'EE': '#FF9800',
                                        'O2': '#4CAF50', 'Vodafone': '#E91E63'},
                    title='Mean RSRP by Operator')
                fig_op.add_hline(y=-40, line_dash="dash", line_color="red")
                fig_op.update_layout(height=250, showlegend=False, plot_bgcolor='white')
                st.plotly_chart(fig_op, use_container_width=True)

        with st.expander("🤖 2. Use XGBoost predictions to shortlist deployment locations"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                XGBoost identified **344 confirmed harvestable locations** with 
                **zero false positives**. Every location it classified as harvestable 
                genuinely met the -40 dBm threshold.

                Use XGBoost as a **first-pass filter**:
                - Run the trained model against candidate location lists
                - All flagged locations are confirmed harvestable
                - Physical survey only needed for flagged locations
                - Expect to find ~16% of all harvestable locations this way
                """)
            with col2:
                st.metric("Confirmed Harvestable", "344 locations")
                st.metric("False Positives", "0")
                st.metric("Precision", "1.0000")
                st.metric("Recall", "16.07%")

        with st.expander("⏱️ 3. Avoid deployment during low-signal hours"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                Signal variation across measured hours reached **up to 15 dBm** 
                between the lowest and highest hourly mean values.

                For locations where harvesting viability is borderline — RSRP values 
                close to but not reliably above -40 dBm — operational scheduling that 
                prioritises energy-intensive tasks during historically stronger signal 
                hours will improve overall energy budget reliability.

                **Note:** Hours 7 and 14 are absent from the Ofcom dataset due to 
                drive-test scheduling constraints.
                """)
            with col2:
                hourly_vf = df.groupby('hour_of_day')['rsrp_top1_vf'].mean().reset_index()
                hourly_vf.columns = ['Hour', 'Mean RSRP']
                fig_h = px.line(hourly_vf, x='Hour', y='Mean RSRP',
                    title='Vodafone Mean RSRP by Hour',
                    color_discrete_sequence=['#E91E63'], markers=True)
                fig_h.add_hline(y=-40, line_dash="dash", line_color="red")
                fig_h.update_layout(height=250, plot_bgcolor='white')
                st.plotly_chart(fig_h, use_container_width=True)

        with st.expander("🗺️ 4. Prioritise West London boroughs for initial deployment"):
            st.markdown("""
            Borough-level spatial analysis identified **Hillingdon** as the highest-harvestability 
            borough at **2.18%**, followed by Harrow (1.38%), Brent (1.25%), and Ealing (1.22%). 
            These four boroughs form a contiguous cluster in West London.

            **Hounslow**, adjacent to Hillingdon, recorded **zero harvestable locations** 
            despite 7,054 measurements — demonstrating that proximity alone does not guarantee 
            harvestability and that borough-level planning must be grounded in measurement data 
            rather than geographic assumption.

            WSN planners targeting initial deployments should prioritise the West London cluster 
            while treating the borough analysis as a starting point pending full spatial 
            coverage across all 33 London boroughs.
            """)

            col1, col2 = st.columns([1, 1])
            with col1:
                borough_data = pd.DataFrame({
                    'Borough': ['Hillingdon', 'Harrow', 'Brent', 'Ealing', 'Enfield',
                                'Redbridge', 'Barnet', 'Waltham Forest', 'Hounslow'],
                    'Harvestability Rate (%)': [2.18, 1.38, 1.25, 1.22, 0.86,
                                                0.38, 0.37, 0.22, 0.00]
                })
                fig_borough = px.bar(borough_data, x='Borough', y='Harvestability Rate (%)',
                    title='Harvestability Rate by London Borough',
                    color='Harvestability Rate (%)', color_continuous_scale='Greens',
                    text='Harvestability Rate (%)')
                fig_borough.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_borough.update_layout(height=400, plot_bgcolor='white',
                    showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig_borough, use_container_width=True)

            with col2:
                if os.path.exists("spatial plot.jpeg"):
                    st.image(
                        "spatial plot.jpeg",
                        caption="RF Harvestability Rate by London Borough (%)",
                        use_container_width=True
                    )
                else:
                    st.info("Borough spatial map: add 'spatial plot.jpeg' to the repository.")

        with st.expander("📶 5. Plan for the 5G transition"):
            st.markdown("""
            The Ofcom 5G dataset covering Greater London was identified during data collection 
            but not analysed within this study's timeframe.

            As London's cellular infrastructure transitions from 4G LTE to 5G NR, 
            signal characteristics that drive harvestability will change. 5G operates 
            at higher frequencies with denser base station deployment, which may:

            - **Increase** ambient RF energy density in urban cores
            - **Reduce** coverage in areas currently served by fewer 4G towers

            The 4G-derived recommendations here should be treated as a **current baseline** 
            pending future 5G validation. Extending this framework to 5G data is the most 
            immediate next step for future research.
            """)

        st.markdown("---")
        st.markdown("### Signal Strength Explorer")
        st.markdown("Explore signal statistics for specific areas of London.")

        col1, col2 = st.columns(2)
        with col1:
            lat_input = st.number_input("Latitude", min_value=51.28, max_value=51.69,
                value=51.5074, step=0.01)
            lon_input = st.number_input("Longitude", min_value=-0.51, max_value=0.33,
                value=-0.1278, step=0.01)
            radius = st.slider("Search radius (km)", 0.5, 5.0, 1.0, 0.5)

        with col2:
            lat_deg = radius / 111
            lon_deg = radius / (111 * np.cos(np.radians(lat_input)))
            nearby = df[
                (df['latitude'].between(lat_input - lat_deg, lat_input + lat_deg)) &
                (df['longitude'].between(lon_input - lon_deg, lon_input + lon_deg))
            ]
            if len(nearby) > 0:
                st.metric("Measurements in area", f"{len(nearby):,}")
                st.metric("Harvestable locations", f"{nearby['harvestable'].sum():,}",
                          f"{nearby['harvestable'].mean()*100:.2f}%")
                rsrp_means = {
                    'Three UK': nearby['rsrp_top1_3uk'].mean(),
                    'EE': nearby['rsrp_top1_ee'].mean(),
                    'O2': nearby['rsrp_top1_o2'].mean(),
                    'Vodafone': nearby['rsrp_top1_vf'].mean()
                }
                best_op = max(rsrp_means, key=rsrp_means.get)
                st.metric("Strongest operator nearby", best_op,
                          f"{rsrp_means[best_op]:.2f} dBm")
                if nearby['harvestable'].mean() > 0.005:
                    st.success("✅ This area has above-average harvestability potential")
                else:
                    st.warning("⚠️ This area has below-average harvestability potential")
            else:
                st.warning("No measurements found in this area. Try increasing the search radius.")

        st.markdown("---")
        st.markdown("### Research Summary")
        st.info("""
        **From Passive to Predictive Harvesting**

        This research demonstrates that machine learning can identify confirmed RF 
        harvestable locations across London with perfect precision. XGBoost identified 
        344 locations with zero false positives — every flagged location genuinely meets 
        the -40 dBm rectenna activation threshold.

        While recall is limited to 16.1%, partial identification of harvestable locations 
        is substantially more useful than no identification at all. The framework is 
        reproducible, built on open government data, and transferable to other UK cities 
        as Ofcom releases future measurement datasets.

        **Citation:** Agwunedu, N.O. (2026). From Passive Harvesting to Predictive Harvesting: 
        A Machine Learning Approach to RF Energy Availability Forecasting for Wireless Sensor 
        Networks in Urban London. MSc Dissertation, University of East London.
        """)