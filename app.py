import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import gc
warnings.filterwarnings('ignore')


def apply_plotly_readable(fig):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#0f172a", family="Poppins"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(15,23,42,0.08)")
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    return fig

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Mental Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MAGICAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root{
  --bg1:#667eea;
  --bg2:#764ba2;
  --text:#0f172a;
  --muted:#475569;
  --card:#ffffff;
  --card2:#f8fafc;
  --border:rgba(15,23,42,0.12);
}

*{font-family:'Poppins',sans-serif;}

/* Page background */
.stApp{
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
}

/* Main readable panel */
.block-container{
  padding: 2rem 2.5rem;
  background: rgba(255,255,255,0.97);
  border-radius: 18px;
  margin: 1rem;
  box-shadow: 0 18px 55px rgba(0,0,0,0.28);
}

/* Make normal text always readable */
h1,h2,h3,h4,h5,h6,p,li,span,div{color:var(--text);}
.stCaption, .stMarkdown p{color:var(--muted);}

/* ---------- KPI / Metric cards (reliable selector) ---------- */
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, var(--card), var(--card2));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 8px 18px rgba(2,6,23,0.08);
}
div[data-testid="stMetric"] *{
  color: var(--text) !important;
}
div[data-testid="stMetricValue"]{
  font-size: 28px !important;
  font-weight: 800 !important;
}
div[data-testid="stMetricLabel"]{
  font-weight: 700 !important;
}
div[data-testid="stMetricDelta"]{
  font-weight: 700 !important;
  font-size: 13px !important;
}

/* Headers */
h1{
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  font-weight:800;
  font-size:3rem;
  text-align:center;
  margin-bottom:0.4rem;
}
h2{color:var(--bg1); font-weight:700;}
h3{color:var(--bg2); font-weight:700;}

/* Insight box — change to soft background + dark text */
.insight-box{
  background: #eef2ff;
  border: 1px solid rgba(102,126,234,0.25);
  border-left: 6px solid var(--bg1);
  padding: 18px;
  border-radius: 14px;
  box-shadow: 0 10px 22px rgba(2,6,23,0.08);
}
.insight-box, .insight-box *{
  color: var(--text) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap:8px;
  background:#f1f5f9;
  padding:10px;
  border-radius:14px;
  border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  height:46px;
  background:#ffffff;
  border-radius:12px;
  color:var(--text);
  font-weight:700;
  border:1px solid var(--border);
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%) !important;
  color:#ffffff !important;
  border:none !important;
}
.stTabs [aria-selected="true"] p{color:#ffffff !important;}

/* Buttons */
.stButton>button, .stDownloadButton>button{
  background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
  color:#ffffff !important;
  border:none;
  border-radius:12px;
  font-weight:700;
  padding:10px 16px;
  box-shadow: 0 8px 18px rgba(102,126,234,0.25);
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform: translateY(-1px);
}

/* Sidebar background */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

/* Sidebar titles stay white */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p{
  color: #ffffff !important;
}

/* IMPORTANT: Inputs in sidebar must be dark text on light background */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea{
  color: #0f172a !important;
}

/* Selectbox / multiselect container backgrounds + text */
[data-testid="stSidebar"] [data-baseweb="select"]{
  background: rgba(255,255,255,0.98) !important;
  border-radius: 10px;
}
[data-testid="stSidebar"] [data-baseweb="select"] *{
  color: #0f172a !important;
}

/* Multiselect tags readable */
[data-testid="stSidebar"] .stMultiSelect span{
  color:#0f172a !important;
}

/* Remove unstable legacy class hooks */
.css-1d391kg, .css-1v0mbdj { color: inherit !important; }

</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="✨ Loading data...")
def load_and_preprocess_data():
    """Load and preprocess data"""
    try:
        df = pd.read_csv('data/Mental_Health_and_Lifestyle_Research.csv')
        df.columns = df.columns.str.strip()
        
        # Handle missing values
        categorical_cols = ['Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
        
        # Convert numeric columns
        numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                        'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle categorical as strings (not category dtype to avoid issues)
        string_cols = ['Gender', 'Mental_Health_Status', 'Social_Interaction_Freq', 
                      'Diet_Quality', 'Substance_Use', 'Physical_Health_Condition']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Handle boolean
        if 'Has_Close_Friends' in df.columns:
            df['Has_Close_Friends'] = df['Has_Close_Friends'].astype(str).str.lower().map({
                'true': True, 'false': False
            })
        
        return df
        
    except FileNotFoundError:
        st.error("❌ CSV file not found at 'data/Mental_Health_and_Lifestyle_Research.csv'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def apply_filters(df, filters):
    """Apply filters efficiently"""
    mask = pd.Series(True, index=df.index)
    
    if 'age_range' in filters:
        mask &= df['Age'].between(filters['age_range'][0], filters['age_range'][1])
    
    filter_mapping = {
        'gender': 'Gender',
        'mental_health': 'Mental_Health_Status',
        'diet_quality': 'Diet_Quality',
        'social_interaction': 'Social_Interaction_Freq',
        'substance_use': 'Substance_Use',
        'physical_health': 'Physical_Health_Condition'
    }
    
    for filter_key, col_name in filter_mapping.items():
        if filter_key in filters and filters[filter_key]:
            mask &= df[col_name].isin(filters[filter_key])
    
    if 'has_friends' in filters and filters['has_friends'] != 'All':
        mask &= df['Has_Close_Friends'] == (filters['has_friends'] == 'True')
    
    return df[mask].copy()

# ============================================================================
# SIDEBAR
# ============================================================================

def create_sidebar(df):
    """Create sidebar"""
    st.sidebar.markdown("# 🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    filters = {}
    
    with st.sidebar.expander("👥 Demographics", expanded=True):
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        filters['age_range'] = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        filters['gender'] = st.multiselect("Gender", 
                                          sorted(df['Gender'].unique().tolist()), 
                                          sorted(df['Gender'].unique().tolist()))
    
    with st.sidebar.expander("🧠 Mental Health"):
        filters['mental_health'] = st.multiselect("Status", 
                                                  sorted(df['Mental_Health_Status'].unique().tolist()),
                                                  sorted(df['Mental_Health_Status'].unique().tolist()))
    
    with st.sidebar.expander("🍎 Lifestyle"):
        filters['diet_quality'] = st.multiselect("Diet", 
                                                sorted(df['Diet_Quality'].unique().tolist()),
                                                sorted(df['Diet_Quality'].unique().tolist()))
        filters['social_interaction'] = st.multiselect("Social", 
                                                      sorted(df['Social_Interaction_Freq'].unique().tolist()),
                                                      sorted(df['Social_Interaction_Freq'].unique().tolist()))
        filters['has_friends'] = st.selectbox("Friends", ['All', 'True', 'False'])
        filters['substance_use'] = st.multiselect("Substance", 
                                                  sorted(df['Substance_Use'].unique().tolist()),
                                                  sorted(df['Substance_Use'].unique().tolist()))
    
    with st.sidebar.expander("🏥 Physical Health"):
        filters['physical_health'] = st.multiselect("Condition", 
                                                    sorted(df['Physical_Health_Condition'].unique().tolist()),
                                                    sorted(df['Physical_Health_Condition'].unique().tolist()))
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()
    
    return filters

# ============================================================================
# KPI CARDS
# ============================================================================

def display_kpis(df):
    """Display KPI cards"""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        avg_wb = df['Overall_Wellbeing_Score'].mean()
        st.metric("🎯 Wellbeing", f"{avg_wb:.1f}/10", 
                 f"{avg_wb - 5.5:+.1f}", delta_color="normal")
    
    with col2:
        avg_stress = df['Stress_Level'].mean()
        st.metric("😰 Stress", f"{avg_stress:.1f}/10",
                 f"{5.5 - avg_stress:+.1f}", delta_color="inverse")
    
    with col3:
        avg_sleep = df['Hours_of_Sleep'].mean()
        st.metric("😴 Sleep", f"{avg_sleep:.1f}h",
                 f"{avg_sleep - 7:+.1f}h")
    
    with col4:
        avg_screen = df['Screen_Time_per_Day'].mean()
        st.metric("📱 Screen", f"{avg_screen:.1f}h",
                 f"{4 - avg_screen:+.1f}h", delta_color="inverse")
    
    with col5:
        avg_work = df['Work_Hours_per_Day'].mean()
        st.metric("💼 Work", f"{avg_work:.1f}h",
                 f"{avg_work - 8:+.1f}h")
    
    with col6:
        pct_friends = (df['Has_Close_Friends'].sum() / len(df)) * 100
        st.metric("👥 Friends", f"{pct_friends:.0f}%")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

def page_overview(df):
    """Overview page"""
    st.markdown("# 📊 Executive Dashboard")
    st.markdown(f"<p style='text-align: center; color: #666; font-size: 1.1rem;'>Analyzing <strong>{len(df):,}</strong> individuals</p>", 
                unsafe_allow_html=True)
    
    display_kpis(df)
    st.markdown("---")
    
    # Gauge Charts
    col1, col2 = st.columns(2)
    
    with col1:
        avg_wellbeing = df['Overall_Wellbeing_Score'].mean()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_wellbeing,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Wellbeing Score", 'font': {'size': 24, 'color': '#667eea'}},
            delta={'reference': 5.5, 'increasing': {'color': "#4CAF50"}, 'decreasing': {'color': "#F44336"}},
            gauge={
                'axis': {'range': [None, 10], 'tickwidth': 2, 'tickcolor': "#667eea"},
                'bar': {'color': "#667eea", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#667eea",
                'steps': [
                    {'range': [0, 3], 'color': '#ffebee'},
                    {'range': [3, 5], 'color': '#fff9c4'},
                    {'range': [5, 7], 'color': '#c8e6c9'},
                    {'range': [7, 10], 'color': '#a5d6a7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        fig = apply_plotly_readable(fig) 
        st.plotly_chart(fig, use_container_width=True, key="gauge_wellbeing")
        st.caption("🎯 Target: 7+ for optimal wellbeing")
    
    with col2:
        avg_stress = df['Stress_Level'].mean()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_stress,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Stress Level", 'font': {'size': 24, 'color': '#f5576c'}},
            delta={'reference': 5.5, 'increasing': {'color': "#F44336"}, 'decreasing': {'color': "#4CAF50"}},
            gauge={
                'axis': {'range': [None, 10], 'tickwidth': 2, 'tickcolor': "#f5576c"},
                'bar': {'color': "#f5576c", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#f5576c",
                'steps': [
                    {'range': [0, 3], 'color': '#a5d6a7'},
                    {'range': [3, 5], 'color': '#c8e6c9'},
                    {'range': [5, 7], 'color': '#fff9c4'},
                    {'range': [7, 10], 'color': '#ffebee'}
                ],
                'threshold': {
                    'line': {'color': "green", 'width': 4},
                    'thickness': 0.75,
                    'value': 3
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig = apply_plotly_readable(fig) 
        st.plotly_chart(fig, use_container_width=True, key="gauge_stress")
        st.caption("✅ Target: Below 5 for healthy stress")
    
    st.markdown("---")
    
    # Sunburst Chart
    st.markdown("### 🌅 Mental Health Landscape")
    
    sunburst_data = df.groupby(['Gender', 'Social_Interaction_Freq', 'Mental_Health_Status']).size().reset_index(name='count')
    
    fig = px.sunburst(
        sunburst_data,
        path=['Gender', 'Social_Interaction_Freq', 'Mental_Health_Status'],
        values='count',
        color='count',
        color_continuous_scale='RdYlGn_r',
        title='Mental Health Distribution by Gender & Social Interaction'
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        font={'family': 'Poppins', 'size': 12}
    )
    fig = apply_plotly_readable(fig) 
    st.plotly_chart(fig, use_container_width=True, key="sunburst_mental")
    st.caption("💡 Click on segments to drill down into specific groups")
    
    st.markdown("---")
    
    # Violin and 3D Scatter
    col3, col4 = st.columns(2)
    
    with col3:
        fig = go.Figure()
        
        diet_categories = sorted(df['Diet_Quality'].unique())
        colors = px.colors.qualitative.Set2
        
        for idx, diet in enumerate(diet_categories):
            diet_data = df[df['Diet_Quality'] == diet]['Overall_Wellbeing_Score']
            fig.add_trace(go.Violin(
                y=diet_data,
                name=diet,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[idx % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            title='Wellbeing Distribution by Diet Quality',
            yaxis_title='Wellbeing Score',
            xaxis_title='Diet Quality',
            height=400,
            showlegend=True,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        fig = apply_plotly_readable(fig) 
        st.plotly_chart(fig, use_container_width=True, key="violin_diet")
        st.caption("🍎 Wider shapes indicate more people at that wellbeing level")
    
    with col4:
        sample_size = min(500, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        
        fig = px.scatter_3d(
            df_sample,
            x='Age',
            y='Hours_of_Sleep',
            z='Overall_Wellbeing_Score',
            color='Stress_Level',
            size='Physical_Activity',
            color_continuous_scale='Viridis',
            title=f'3D View: Age, Sleep & Wellbeing (n={sample_size})',
            labels={'Age': 'Age', 'Hours_of_Sleep': 'Sleep (hrs)', 
                   'Overall_Wellbeing_Score': 'Wellbeing'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="3d_scatter")
        st.caption("🔄 Drag to rotate | Scroll to zoom")
    
    # Insights
    st.markdown("---")
    st.markdown("### 🔍 Key Insights")
    
    avg_wb = df['Overall_Wellbeing_Score'].mean()
    avg_stress = df['Stress_Level'].mean()
    avg_sleep = df['Hours_of_Sleep'].mean()
    
    # Find best diet safely
    diet_wellbeing = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].mean()
    best_diet = diet_wellbeing.idxmax() if len(diet_wellbeing) > 0 else "N/A"
    
    insights_html = f"""
    <div class="insight-box">
        <h3 style="margin-top: 0;">📊 Current Population Insights</h3>
        <ul style="font-size: 1.05rem; line-height: 1.8;">
            <li><strong>Wellbeing Status:</strong> {avg_wb:.1f}/10 - {'🟢 Excellent' if avg_wb >= 7 else '🟡 Good' if avg_wb >= 5 else '🔴 Needs Attention'}</li>
            <li><strong>Stress Level:</strong> {avg_stress:.1f}/10 - {'🔴 High Alert' if avg_stress >= 7 else '🟡 Moderate' if avg_stress >= 5 else '🟢 Healthy'}</li>
            <li><strong>Sleep Pattern:</strong> {avg_sleep:.1f} hours - {'Below recommended' if avg_sleep < 7 else 'Adequate'}</li>
            <li><strong>Best Diet for Wellbeing:</strong> {best_diet}</li>
            <li><strong>Sample Size:</strong> {len(df)} people analyzed</li>
        </ul>
    </div>
    """
    st.markdown(insights_html, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: LIFESTYLE
# ============================================================================

def page_lifestyle(df):
    """Lifestyle analysis"""
    st.markdown("# 🏃 Lifestyle Intelligence")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Discover hidden patterns in lifestyle choices</p>", 
                unsafe_allow_html=True)
    
    # Bubble Chart
    st.markdown("### 📱 Digital Lifestyle Impact")
    
    fig = px.scatter(
        df,
        x='Screen_Time_per_Day',
        y='Overall_Wellbeing_Score',
        size='Physical_Activity',
        color='Stress_Level',
        hover_data=['Age', 'Hours_of_Sleep', 'Work_Hours_per_Day'],
        color_continuous_scale='RdYlGn_r',
        title='Screen Time vs Wellbeing (Bubble size = Physical Activity)',
        labels={'Screen_Time_per_Day': 'Screen Time (hours/day)', 
               'Overall_Wellbeing_Score': 'Wellbeing Score'},
        size_max=30
    )
    
    # Add trendline
    z = np.polyfit(df['Screen_Time_per_Day'], df['Overall_Wellbeing_Score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Screen_Time_per_Day'].min(), df['Screen_Time_per_Day'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig.update_layout(
        height=500,
        font={'family': 'Poppins'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="bubble_screen")
    
    corr = df[['Screen_Time_per_Day', 'Overall_Wellbeing_Score']].corr().iloc[0, 1]
    st.caption(f"📊 Correlation: {corr:.3f} - {'Negative relationship' if corr < -0.1 else 'Positive relationship' if corr > 0.1 else 'Weak relationship'}")
    
    st.markdown("---")
    
    # Parallel Coordinates
    st.markdown("### 🌈 Multi-Dimensional Lifestyle Patterns")
    
    # Prepare normalized data for parallel coordinates
    parallel_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                     'Work_Hours_per_Day', 'Screen_Time_per_Day', 'Overall_Wellbeing_Score']
    
    parallel_df = df[parallel_cols].copy()
    
    # Normalize all columns to 0-10 scale
    for col in parallel_df.columns:
        if col != 'Overall_Wellbeing_Score':
            col_min = parallel_df[col].min()
            col_max = parallel_df[col].max()
            if col_max > col_min:
                parallel_df[col] = (parallel_df[col] - col_min) / (col_max - col_min) * 10
            else:
                parallel_df[col] = 5  # Default middle value if no variance
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['Overall_Wellbeing_Score'],
                colorscale='RdYlGn',
                showscale=True,
                cmin=df['Overall_Wellbeing_Score'].min(),
                cmax=df['Overall_Wellbeing_Score'].max()
            ),
            dimensions=[
                dict(label='Age', values=parallel_df['Age']),
                dict(label='Sleep', values=parallel_df['Hours_of_Sleep']),
                dict(label='Stress', values=parallel_df['Stress_Level']),
                dict(label='Activity', values=parallel_df['Physical_Activity']),
                dict(label='Work', values=parallel_df['Work_Hours_per_Day']),
                dict(label='Screen', values=parallel_df['Screen_Time_per_Day']),
                dict(label='Wellbeing', values=df['Overall_Wellbeing_Score'])
            ]
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates: Lifestyle Factors (Color = Wellbeing)',
        height=500,
        font={'family': 'Poppins', 'size': 12},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="parallel_coords")
    st.caption("💡 Drag on axes to filter | Green lines = high wellbeing | Red lines = low wellbeing")
    
    st.markdown("---")
    
    # Radar Charts
    st.markdown("### 🎯 Lifestyle Profile Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_wb = df[df['Overall_Wellbeing_Score'] >= 7]
        
        if len(high_wb) > 0:
            categories = ['Sleep', 'Low Stress', 'Activity', 'Social', 'Diet']
            
            # Calculate values safely
            sleep_val = (high_wb['Hours_of_Sleep'].mean() / 9) * 10
            stress_val = 10 - high_wb['Stress_Level'].mean()
            activity_val = (high_wb['Physical_Activity'].mean() / df['Physical_Activity'].max()) * 10 if df['Physical_Activity'].max() > 0 else 5
            
            # Map social interaction to numeric
            social_map = {'Low': 3, 'Moderate': 6, 'High': 10}
            social_numeric = high_wb['Social_Interaction_Freq'].map(social_map)
            social_val = social_numeric.mean() if not social_numeric.isna().all() else 5
            
            # Map diet quality to numeric
            diet_map = {'Poor': 2, 'Fair': 4, 'Good': 6, 'Very Good': 8, 'Excellent': 10}
            diet_numeric = high_wb['Diet_Quality'].map(diet_map)
            diet_val = diet_numeric.mean() if not diet_numeric.isna().all() else 5
            
            high_values = [sleep_val, stress_val, activity_val, social_val, diet_val]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=high_values,
                theta=categories,
                fill='toself',
                name='High Wellbeing (7+)',
                line_color='#4CAF50',
                fillcolor='rgba(76, 175, 80, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                title='High Wellbeing Profile',
                height=400,
                font={'family': 'Poppins'},
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="radar_high")
        else:
            st.info("Not enough data for high wellbeing profile")
    
    with col2:
        low_wb = df[df['Overall_Wellbeing_Score'] < 5]
        
        if len(low_wb) > 0:
            # Calculate values safely
            sleep_val = (low_wb['Hours_of_Sleep'].mean() / 9) * 10
            stress_val = 10 - low_wb['Stress_Level'].mean()
            activity_val = (low_wb['Physical_Activity'].mean() / df['Physical_Activity'].max()) * 10 if df['Physical_Activity'].max() > 0 else 5
            
            # Map social interaction
            social_numeric = low_wb['Social_Interaction_Freq'].map(social_map)
            social_val = social_numeric.mean() if not social_numeric.isna().all() else 5
            
            # Map diet quality
            diet_numeric = low_wb['Diet_Quality'].map(diet_map)
            diet_val = diet_numeric.mean() if not diet_numeric.isna().all() else 5
            
            low_values = [sleep_val, stress_val, activity_val, social_val, diet_val]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=low_values,
                theta=categories,
                fill='toself',
                name='Low Wellbeing (<5)',
                line_color='#F44336',
                fillcolor='rgba(244, 67, 54, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                title='Low Wellbeing Profile',
                height=400,
                font={'family': 'Poppins'},
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="radar_low")
        else:
            st.info("Not enough data for low wellbeing profile")
    
    st.caption("📊 Larger area = better lifestyle factors")
    
    st.markdown("---")
    
    # Heatmap
    st.markdown("### ⏰ Sleep vs Work Hours Impact Matrix")
    
    df_temp = df.copy()
    df_temp['Sleep_Bin'] = pd.cut(df_temp['Hours_of_Sleep'], bins=5, labels=['Very Low', 'Low', 'Medium', 'Good', 'Excellent'])
    df_temp['Work_Bin'] = pd.cut(df_temp['Work_Hours_per_Day'], bins=5, labels=['<6h', '6-7h', '7-8h', '8-9h', '9+h'])
    
    heatmap_data = df_temp.groupby(['Sleep_Bin', 'Work_Bin'], observed=True)['Overall_Wellbeing_Score'].mean().unstack()
    
    if not heatmap_data.empty:
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Wellbeing")
        ))
        
        fig.update_layout(
            title='Average Wellbeing by Sleep & Work Hours',
            xaxis_title='Work Hours per Day',
            yaxis_title='Sleep Quality',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="heatmap_sleep_work")
        st.caption("🟢 Green = High wellbeing | 🔴 Red = Low wellbeing")
    else:
        st.info("Not enough data for heatmap visualization")

# ============================================================================
# PAGE 3: SEGMENTS
# ============================================================================

def page_segments(df):
    """Segment analysis"""
    st.markdown("# 🔍 Segment Intelligence")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Deep dive into population segments</p>", 
                unsafe_allow_html=True)
    
    # Waterfall Chart
    st.markdown("### 💧 Wellbeing Contributors Analysis")
    
    baseline = df['Overall_Wellbeing_Score'].mean()
    
    factors = {}
    factors['Baseline'] = baseline
    
    # Calculate impacts safely
    good_sleep = df[df['Hours_of_Sleep'] >= 7]
    factors['Good Sleep (7+h)'] = good_sleep['Overall_Wellbeing_Score'].mean() - baseline if len(good_sleep) > 0 else 0
    
    low_stress = df[df['Stress_Level'] < 5]
    factors['Low Stress (<5)'] = low_stress['Overall_Wellbeing_Score'].mean() - baseline if len(low_stress) > 0 else 0
    
    has_friends = df[df['Has_Close_Friends'] == True]
    factors['Has Friends'] = has_friends['Overall_Wellbeing_Score'].mean() - baseline if len(has_friends) > 0 else 0
    
    high_activity = df[df['Physical_Activity'] > df['Physical_Activity'].median()]
    factors['High Activity'] = high_activity['Overall_Wellbeing_Score'].mean() - baseline if len(high_activity) > 0 else 0
    
    good_diet = df[df['Diet_Quality'].isin(['Excellent', 'Very Good'])]
    factors['Good Diet'] = good_diet['Overall_Wellbeing_Score'].mean() - baseline if len(good_diet) > 0 else 0
    
    fig = go.Figure(go.Waterfall(
        name="Wellbeing",
        orientation="v",
        measure=["absolute"] + ["relative"] * 5,
        x=list(factors.keys()),
        textposition="outside",
        text=[f"{v:+.2f}" for v in factors.values()],
        y=list(factors.values()),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#4CAF50"}},
        decreasing={"marker": {"color": "#F44336"}},
        totals={"marker": {"color": "#667eea"}}
    ))
    
    fig.update_layout(
        title="Impact of Lifestyle Factors on Wellbeing",
        showlegend=False,
        height=450,
        font={'family': 'Poppins'},
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Wellbeing Score Impact"
    )
    
    st.plotly_chart(fig, use_container_width=True, key="waterfall_wellbeing")
    st.caption("📊 Shows how each factor contributes to overall wellbeing")
    
    st.markdown("---")
    
    # Box Plots
    st.markdown("### 📦 Statistical Comparison: Gender & Friendship Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        genders = sorted(df['Gender'].unique())
        colors = ['#667eea', '#f5576c', '#4CAF50', '#FFC107']
        
        for idx, gender in enumerate(genders):
            gender_data = df[df['Gender'] == gender]['Overall_Wellbeing_Score']
            fig.add_trace(go.Box(
                y=gender_data,
                name=gender,
                boxmean='sd',
                marker_color=colors[idx % len(colors)]
            ))
        
        fig.update_layout(
            title='Wellbeing Distribution by Gender',
            yaxis_title='Wellbeing Score',
            height=400,
            showlegend=True,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="box_gender")
    
    with col2:
        fig = go.Figure()
        
        df_temp = df.copy()
        df_temp['Friends_Label'] = df_temp['Has_Close_Friends'].map({True: 'Has Friends', False: 'No Friends'})
        
        for idx, friend_status in enumerate(['Has Friends', 'No Friends']):
            friend_data = df_temp[df_temp['Friends_Label'] == friend_status]['Overall_Wellbeing_Score']
            if len(friend_data) > 0:
                fig.add_trace(go.Box(
                    y=friend_data,
                    name=friend_status,
                    boxmean='sd',
                    marker_color='#4CAF50' if friend_status == 'Has Friends' else '#F44336'
                ))
        
        fig.update_layout(
            title='Wellbeing Distribution by Friendship Status',
            yaxis_title='Wellbeing Score',
            height=400,
            showlegend=True,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="box_friends")
    
    st.caption("📊 Box shows middle 50% | Line = median | Diamond = mean")
    
    st.markdown("---")
    
    # Segment Performance Tables
    st.markdown("### 🏆 Top & Bottom Performing Segments")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🌟 Highest Wellbeing Segments")
        
        segments = []
        
        # By Gender
        for gender in df['Gender'].unique():
            subset = df[df['Gender'] == gender]
            if len(subset) > 0:
                segments.append({
                    'Segment': f'Gender: {gender}',
                    'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                    'Count': len(subset),
                    'Avg Stress': subset['Stress_Level'].mean()
                })
        
        # By Social Interaction
        for social in df['Social_Interaction_Freq'].unique():
            subset = df[df['Social_Interaction_Freq'] == social]
            if len(subset) > 0:
                segments.append({
                    'Segment': f'Social: {social}',
                    'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                    'Count': len(subset),
                    'Avg Stress': subset['Stress_Level'].mean()
                })
        
        # By Friends
        for friends in [True, False]:
            subset = df[df['Has_Close_Friends'] == friends]
            if len(subset) > 0:
                label = 'Has Friends' if friends else 'No Friends'
                segments.append({
                    'Segment': label,
                    'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                    'Count': len(subset),
                    'Avg Stress': subset['Stress_Level'].mean()
                })
        
        if segments:
            segments_df = pd.DataFrame(segments).sort_values('Avg Wellbeing', ascending=False).head(5)
            segments_df['Avg Wellbeing'] = segments_df['Avg Wellbeing'].round(2)
            segments_df['Avg Stress'] = segments_df['Avg Stress'].round(2)
            
            st.dataframe(segments_df, hide_index=True, use_container_width=True)
        else:
            st.info("No segment data available")
    
    with col4:
        st.markdown("#### ⚠️ Highest Stress Segments")
        
        if segments:
            segments_stress_df = pd.DataFrame(segments).sort_values('Avg Stress', ascending=False).head(5)
            segments_stress_df['Avg Wellbeing'] = segments_stress_df['Avg Wellbeing'].round(2)
            segments_stress_df['Avg Stress'] = segments_stress_df['Avg Stress'].round(2)
            
            st.dataframe(segments_stress_df, hide_index=True, use_container_width=True)
        else:
            st.info("No segment data available")

# ============================================================================
# PAGE 4: CORRELATIONS
# ============================================================================

def page_correlations(df):
    """Correlation analysis"""
    st.markdown("# 📈 Correlation Intelligence")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Discover hidden relationships in the data</p>", 
                unsafe_allow_html=True)
    
    numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                    'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
    
    corr_df = df[numeric_cols].corr()
    
    # Interactive Correlation Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Interactive Correlation Matrix',
        height=600,
        font={'family': 'Poppins'},
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap_advanced")
    
    st.markdown("""
    <div class="highlight-card">
        <h4>📖 How to Read This Heatmap:</h4>
        <ul>
            <li><strong>+1 (Dark Blue):</strong> Perfect positive correlation - variables move together</li>
            <li><strong>-1 (Dark Red):</strong> Perfect negative correlation - variables move opposite</li>
            <li><strong>0 (White):</strong> No correlation - variables are independent</li>
            <li><strong>Hover</strong> over cells to see exact correlation values</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Correlations Analysis
    st.markdown("### 🔝 Strongest Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Positive Correlations")
        
        corr_pairs = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                corr_pairs.append({
                    'Variable 1': corr_df.columns[i],
                    'Variable 2': corr_df.columns[j],
                    'Correlation': corr_df.iloc[i, j]
                })
        
        corr_pairs_df = pd.DataFrame(corr_pairs)
        top_positive = corr_pairs_df.nlargest(5, 'Correlation')
        top_positive['Correlation'] = top_positive['Correlation'].round(3)
        
        # Add interpretation
        top_positive['Strength'] = top_positive['Correlation'].apply(
            lambda x: '🟢 Strong' if abs(x) > 0.5 else '🟡 Moderate' if abs(x) > 0.3 else '⚪ Weak'
        )
        
        st.dataframe(top_positive, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### ⛔ Negative Correlations")
        
        top_negative = corr_pairs_df.nsmallest(5, 'Correlation')
        top_negative['Correlation'] = top_negative['Correlation'].round(3)
        
        top_negative['Strength'] = top_negative['Correlation'].apply(
            lambda x: '🟢 Strong' if abs(x) > 0.5 else '🟡 Moderate' if abs(x) > 0.3 else '⚪ Weak'
        )
        
        st.dataframe(top_negative, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Data Quality Dashboard
    st.markdown("### 🔍 Data Quality Dashboard")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with col4:
        st.metric("📈 Numeric Features", len(numeric_cols))
    
    with col5:
        completeness = ((df.size - df.isna().sum().sum()) / df.size) * 100
        st.metric("✅ Completeness", f"{completeness:.1f}%")
    
    with col6:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
    
    # Missing values visualization
    st.markdown("---")
    st.markdown("### 🕳️ Missing Data Analysis")
    
    missing_data = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if col in ['Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']:
            unknown_count = (df[col] == 'Unknown').sum()
            total_missing = missing_count + unknown_count
        else:
            total_missing = missing_count
        
        if total_missing > 0:
            missing_data.append({
                'Column': col,
                'Missing': total_missing,
                'Percentage': (total_missing / len(df)) * 100
            })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data).sort_values('Missing', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_df['Column'],
            y=missing_df['Percentage'],
            text=missing_df['Percentage'].round(1).astype(str) + '%',
            textposition='outside',
            marker=dict(
                color=missing_df['Percentage'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="% Missing")
            )
        ))
        
        fig.update_layout(
            title='Missing Data by Column',
            xaxis_title='Column',
            yaxis_title='Percentage Missing (%)',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="missing_data")
    else:
        st.success("✅ No missing data detected!")

# ============================================================================
# PAGE 5: ML PREDICTOR
# ============================================================================

@st.cache_resource(show_spinner="🤖 Training AI model...")
def train_ml_model(_df):
    """Train ML model"""
    df_ml = _df.copy()
    
    label_encoders = {}
    categorical_features = ['Gender', 'Diet_Quality', 'Social_Interaction_Freq', 
                           'Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']
    
    for col in categorical_features:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
    
    df_ml['Has_Close_Friends_encoded'] = df_ml['Has_Close_Friends'].astype(int)
    
    feature_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                   'Work_Hours_per_Day', 'Screen_Time_per_Day',
                   'Gender_encoded', 'Diet_Quality_encoded', 'Social_Interaction_Freq_encoded',
                   'Has_Close_Friends_encoded', 'Substance_Use_encoded', 
                   'Physical_Health_Condition_encoded']
    
    X = df_ml[feature_cols]
    y = df_ml['Overall_Wellbeing_Score']
    
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=6,
        random_state=42,
        n_jobs=1,
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    del X_train, X_test, y_train, df_ml
    gc.collect()
    
    return model, label_encoders, feature_cols, rmse, r2, y_test, y_pred

def page_ml_predictor(df):
    """ML prediction page"""
    st.markdown("# 🤖 AI Wellbeing Predictor")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Powered by Random Forest Machine Learning</p>", 
                unsafe_allow_html=True)
    
    model, label_encoders, feature_cols, rmse, r2, y_test, y_pred = train_ml_model(df)
    
    # Model Performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 R² Score", f"{r2:.3f}", help="Closer to 1 = better model")
    
    with col2:
        st.metric("📊 RMSE", f"{rmse:.3f}", help="Lower = better predictions")
    
    with col3:
        accuracy = max(0, (1 - (rmse / df['Overall_Wellbeing_Score'].std())) * 100)
        st.metric("✅ Accuracy", f"{accuracy:.0f}%")
    
    with col4:
        st.metric("📈 Predictions", f"{len(y_test)}", help="Test set size")
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🎯 What Matters Most?")
    
    feature_importance = pd.DataFrame({
        'Feature': [col.replace('_encoded', '').replace('_', ' ').title() for col in feature_cols],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=feature_importance['Feature'],
        x=feature_importance['Importance'],
        orientation='h',
        marker=dict(
            color=feature_importance['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=feature_importance['Importance'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Importance: Which Factors Predict Wellbeing?',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        font={'family': 'Poppins'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250,250,250,1)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="feature_importance")
    
    st.markdown("---")
    
    # Actual vs Predicted
    st.markdown("### 📊 Model Accuracy Visualization")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=y_test,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Actual Value"),
                opacity=0.6
            ),
            text=[f'Actual: {a:.2f}<br>Predicted: {p:.2f}' for a, p in zip(y_test, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Wellbeing',
            xaxis_title='Actual Wellbeing Score',
            yaxis_title='Predicted Wellbeing Score',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key="actual_vs_pred")
    
    with col_chart2:
        residuals = y_test - y_pred
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color=np.abs(residuals),
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Error Size"),
                opacity=0.6
            ),
            text=[f'Predicted: {p:.2f}<br>Error: {r:.2f}' for p, r in zip(y_pred, residuals)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Perfect Prediction")
        
        fig.update_layout(
            title='Prediction Errors (Residuals)',
            xaxis_title='Predicted Wellbeing Score',
            yaxis_title='Prediction Error',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="residuals")
    
    st.markdown("---")
    
    # Interactive Predictor
    st.markdown("### 🎮 Try It Yourself: Predict Your Wellbeing")
    st.markdown("**Enter your information below to get a personalized wellbeing prediction:**")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal Info**")
            input_age = st.slider("Age", 20, 35, 27)
            input_gender = st.selectbox("Gender", sorted(df['Gender'].unique().tolist()))
            input_friends = st.selectbox("Have Close Friends?", ['Yes', 'No'])
        
        with col2:
            st.markdown("**😴 Health & Lifestyle**")
            input_sleep = st.slider("Hours of Sleep", 5.0, 9.0, 7.0, 0.5)
            input_stress = st.slider("Stress Level (1-10)", 1, 10, 5)
            input_activity = st.slider("Physical Activity Score", 
                                      int(df['Physical_Activity'].min()), 
                                      int(df['Physical_Activity'].max()), 
                                      int(df['Physical_Activity'].mean()))
        
        with col3:
            st.markdown("**💼 Work & Screen**")
            input_work = st.slider("Work Hours per Day", 5.0, 11.0, 8.0, 0.5)
            input_screen = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, 0.5)
            input_diet = st.selectbox("Diet Quality", sorted(df['Diet_Quality'].unique().tolist()))
        
        col4, col5 = st.columns(2)
        
        with col4:
            input_social = st.selectbox("Social Interaction Frequency", 
                                       sorted(df['Social_Interaction_Freq'].unique().tolist()))
        
        with col5:
            input_substance = st.selectbox("Substance Use", 
                                          sorted(df['Substance_Use'].unique().tolist()))
        
        input_physical = st.selectbox("Physical Health Condition", 
                                     sorted(df['Physical_Health_Condition'].unique().tolist()))
        
        submitted = st.form_submit_button("🔮 Predict My Wellbeing Score", 
                                         type="primary", 
                                         use_container_width=True)
    
    if submitted:
        input_data = {
            'Age': input_age,
            'Hours_of_Sleep': input_sleep,
            'Stress_Level': input_stress,
            'Physical_Activity': input_activity,
            'Work_Hours_per_Day': input_work,
            'Screen_Time_per_Day': input_screen,
            'Gender_encoded': label_encoders['Gender'].transform([input_gender])[0],
            'Diet_Quality_encoded': label_encoders['Diet_Quality'].transform([input_diet])[0],
            'Social_Interaction_Freq_encoded': label_encoders['Social_Interaction_Freq'].transform([input_social])[0],
            'Has_Close_Friends_encoded': 1 if input_friends == 'Yes' else 0,
            'Substance_Use_encoded': label_encoders['Substance_Use'].transform([input_substance])[0],
            'Physical_Health_Condition_encoded': label_encoders['Physical_Health_Condition'].transform([input_physical])[0]
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.markdown("### 🎯 Your Predicted Wellbeing Score")
        
        if prediction >= 7:
            color, status, emoji = "#4CAF50", "Excellent", "🌟"
        elif prediction >= 5:
            color, status, emoji = "#FFC107", "Good", "👍"
        else:
            color, status, emoji = "#F44336", "Needs Attention", "⚠️"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; 
                    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    border-radius: 20px; border: 4px solid {color}; margin: 20px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
            <h1 style='font-size: 72px; margin: 0; color: {color}; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>{prediction:.1f}</h1>
            <p style='font-size: 20px; color: #666; margin: 10px 0; font-weight: 600;'>out of 10</p>
            <h2 style='margin: 15px 0; color: {color}; font-size: 32px;'>{emoji} {status}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Personalized recommendations
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = []
        
        if input_sleep < 7:
            recommendations.append("😴 **Improve Sleep:** Try to get at least 7-8 hours of sleep per night for better wellbeing.")
        
        if input_stress >= 7:
            recommendations.append("🧘 **Manage Stress:** Consider stress-reduction techniques like meditation, exercise, or talking to someone.")
        
        if input_screen > 6:
            recommendations.append("📱 **Reduce Screen Time:** High screen time may impact wellbeing. Try taking regular breaks.")
        
        if input_activity < df['Physical_Activity'].median():
            recommendations.append("🏃 **Increase Physical Activity:** Regular exercise is strongly linked to better mental health.")
        
        if input_friends == 'No':
            recommendations.append("👥 **Build Social Connections:** Having close friends is associated with higher wellbeing scores.")
        
        if input_work > 9:
            recommendations.append("⚖️ **Work-Life Balance:** Long work hours may contribute to stress. Try to maintain balance.")
        
        if not recommendations:
            recommendations.append("✅ **Keep It Up!** Your lifestyle factors look good. Maintain these healthy habits!")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        st.markdown("---")
        
        # Comparison with dataset
        st.markdown("### 📊 How You Compare")
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            avg_wellbeing = df['Overall_Wellbeing_Score'].mean()
            diff = prediction - avg_wellbeing
            st.metric(
                "vs Average Wellbeing",
                f"{prediction:.1f}",
                f"{diff:+.1f}",
                delta_color="normal"
            )
        
        with col_comp2:
            percentile = (df['Overall_Wellbeing_Score'] < prediction).sum() / len(df) * 100
            st.metric(
                "Your Percentile",
                f"{percentile:.0f}%",
                "Higher is better"
            )
        
        with col_comp3:
            similar_count = len(df[(df['Overall_Wellbeing_Score'] >= prediction - 0.5) & 
                                  (df['Overall_Wellbeing_Score'] <= prediction + 0.5)])
            st.metric(
                "Similar People",
                f"{similar_count}",
                f"in dataset"
            )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    df = load_and_preprocess_data()
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 0;'>🧠 Mental Health Analytics</h1>
        <p style='font-size: 1.3rem; color: #667eea; font-weight: 600;'>
            Powered by AI & Advanced Data Science
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    filters = create_sidebar(df)
    df_filtered = apply_filters(df, filters)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No data matches your filters. Please adjust your filter selections.")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Filtered Records", f"{len(df_filtered):,} / {len(df):,}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export Data")
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_mental_health_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.sidebar.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Mental Health & Lifestyle Dashboard v3.0**
        
        This dashboard analyzes the relationship between lifestyle factors and mental health outcomes.
        
        **Features:**
        - Interactive filtering
        - Real-time analytics
        - ML predictions
        - Data export
        - 15+ Advanced visualizations
        
        **Built with:**
        - Streamlit
        - Plotly
        - Scikit-learn
        
        **Version:** 3.0.0
        """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏃 Lifestyle Drivers",
        "🔍 Segment Comparison",
        "📈 Correlations & Data Quality",
        "🤖 AI Wellbeing Predictor"
    ])
    
    with tab1:
        page_overview(df_filtered)
    
    with tab2:
        page_lifestyle(df_filtered)
    
    with tab3:
        page_segments(df_filtered)
    
    with tab4:
        page_correlations(df_filtered)
    
    with tab5:
        page_ml_predictor(df_filtered)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white;'>
        <p style='margin: 0; font-size: 16px; font-weight: 600;'>
            🧠 Mental Health & Lifestyle Analytics Dashboard v3.0
        </p>
        <p style='margin: 5px 0 0 0; font-size: 13px; opacity: 0.9;'>
            For educational purposes only | Consult healthcare professionals for medical advice
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
