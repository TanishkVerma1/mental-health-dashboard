import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import gc
warnings.filterwarnings('ignore')

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
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        color: white;
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .stMetric .metric-value {
        color: white !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #764ba2;
        font-weight: 600;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.4);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .highlight-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 20px rgba(168, 237, 234, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="✨ Loading magical data...")
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
        
        # Convert to appropriate types
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype(str)
        if 'Diet_Quality' in df.columns:
            df['Diet_Quality'] = df['Diet_Quality'].astype(str)
        if 'Social_Interaction_Freq' in df.columns:
            df['Social_Interaction_Freq'] = df['Social_Interaction_Freq'].astype(str)
        
        if 'Has_Close_Friends' in df.columns:
            df['Has_Close_Friends'] = df['Has_Close_Friends'].astype(str).str.lower().map({
                'true': True, 'false': False
            })
        
        # Numeric columns
        numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                       'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except FileNotFoundError:
        st.error("❌ CSV file not found")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
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
    """Create magical sidebar"""
    st.sidebar.markdown("# 🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    filters = {}
    
    with st.sidebar.expander("👥 Demographics", expanded=True):
        age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
        filters['age_range'] = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        filters['gender'] = st.multiselect("Gender", sorted(df['Gender'].unique().tolist()), 
                                          sorted(df['Gender'].unique().tolist()))
    
    with st.sidebar.expander("🧠 Mental Health"):
        filters['mental_health'] = st.multiselect("Status", 
                                                  sorted(df['Mental_Health_Status'].unique().tolist()),
                                                  sorted(df['Mental_Health_Status'].unique().tolist()))
    
    with st.sidebar.expander("🍎 Lifestyle"):
        filters['diet_quality'] = st.multiselect("Diet", sorted(df['Diet_Quality'].unique().tolist()),
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
# MAGICAL KPI CARDS
# ============================================================================

def display_magical_kpis(df):
    """Display animated KPI cards"""
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
# PAGE 1: OVERVIEW WITH MAGICAL CHARTS
# ============================================================================

def page_overview(df):
    """Overview with stunning visualizations"""
    st.markdown("# 📊 Executive Dashboard")
    st.markdown(f"<p style='text-align: center; color: #666; font-size: 1.1rem;'>Analyzing <strong>{len(df):,}</strong> individuals</p>", 
                unsafe_allow_html=True)
    
    display_magical_kpis(df)
    st.markdown("---")
    
    # Gauge Charts
    col1, col2 = st.columns([1, 1])
    
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
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#667eea", 'family': "Poppins"}
        )
        
        st.plotly_chart(fig, use_container_width=True, key="gauge_wellbeing")
        st.markdown("<p style='text-align: center; color: #666;'>🎯 <strong>Target:</strong> 7+ for optimal wellbeing</p>", 
                   unsafe_allow_html=True)
    
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
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f5576c", 'family': "Poppins"}
        )
        
        st.plotly_chart(fig, use_container_width=True, key="gauge_stress")
        st.markdown("<p style='text-align: center; color: #666;'>✅ <strong>Target:</strong> Below 5 for healthy stress</p>", 
                   unsafe_allow_html=True)
    
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
        font={'family': 'Poppins', 'size': 12},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="sunburst_mental")
    st.markdown("<p style='text-align: center; color: #666;'>💡 <strong>Insight:</strong> Click on segments to drill down into specific groups</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Violin and 3D Scatter
    col3, col4 = st.columns(2)
    
    with col3:
        fig = go.Figure()
        
        for diet in sorted(df['Diet_Quality'].unique()):
            diet_data = df[df['Diet_Quality'] == diet]['Overall_Wellbeing_Score']
            fig.add_trace(go.Violin(
                y=diet_data,
                name=diet,
                box_visible=True,
                meanline_visible=True,
                fillcolor='rgba(102, 126, 234, 0.5)',
                line_color='#667eea',
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Wellbeing Distribution by Diet Quality',
            yaxis_title='Wellbeing Score',
            xaxis_title='Diet Quality',
            height=400,
            showlegend=False,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="violin_diet")
        st.markdown("<p style='text-align: center; color: #666;'>🍎 Wider shapes = more people at that wellbeing level</p>", 
                   unsafe_allow_html=True)
    
    with col4:
        sample_df = df.sample(min(500, len(df)))
        fig = px.scatter_3d(
            sample_df,
            x='Age',
            y='Hours_of_Sleep',
            z='Overall_Wellbeing_Score',
            color='Stress_Level',
            size='Physical_Activity',
            color_continuous_scale='Viridis',
            title='3D Relationship: Age, Sleep & Wellbeing',
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
        st.markdown("<p style='text-align: center; color: #666;'>🔄 Drag to rotate | Scroll to zoom</p>", 
                   unsafe_allow_html=True)
    
    # Insights
    st.markdown("---")
    st.markdown("### 🔍 Key Insights")
    
    avg_wb = df['Overall_Wellbeing_Score'].mean()
    avg_stress = df['Stress_Level'].mean()
    avg_sleep = df['Hours_of_Sleep'].mean()
    best_diet = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].mean().idxmax()
    worst_mental = df['Mental_Health_Status'].value_counts().idxmax()
    
    insights_html = f"""
    <div class="insight-box">
        <h3 style="color: white; margin-top: 0;">📊 Current Population Insights</h3>
        <ul style="font-size: 1.05rem; line-height: 1.8;">
            <li><strong>Wellbeing Status:</strong> {avg_wb:.1f}/10 - {'🟢 Excellent' if avg_wb >= 7 else '🟡 Good' if avg_wb >= 5 else '🔴 Needs Attention'}</li>
            <li><strong>Stress Level:</strong> {avg_stress:.1f}/10 - {'🔴 High Alert' if avg_stress >= 7 else '🟡 Moderate' if avg_stress >= 5 else '🟢 Healthy'}</li>
            <li><strong>Best Diet for Wellbeing:</strong> {best_diet} shows highest wellbeing scores</li>
            <li><strong>Most Common Mental Health Status:</strong> {worst_mental}</li>
            <li><strong>Social Connection:</strong> {(df['Has_Close_Friends'].sum() / len(df) * 100):.0f}% have close friends</li>
        </ul>
    </div>
    """
    st.markdown(insights_html, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: LIFESTYLE ANALYSIS (FIXED)
# ============================================================================

def page_lifestyle(df):
    """Advanced lifestyle analysis"""
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
    z = np.polyfit(df['Screen_Time_per_Day'].dropna(), df['Overall_Wellbeing_Score'].dropna(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Screen_Time_per_Day'].min(), df['Screen_Time_per_Day'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode='lines',
        name='Trend',
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
    st.markdown(f"<p style='text-align: center; color: #666;'>📊 <strong>Correlation:</strong> {corr:.3f} - {'Negative relationship' if corr < -0.1 else 'Positive relationship' if corr > 0.1 else 'Weak relationship'}</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parallel Coordinates
    st.markdown("### 🌈 Multi-Dimensional Lifestyle Patterns")
    
    parallel_df = df[['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                      'Work_Hours_per_Day', 'Screen_Time_per_Day', 'Overall_Wellbeing_Score']].copy()
    
    # Normalize
    for col in parallel_df.columns:
        if col != 'Overall_Wellbeing_Score':
            parallel_df[col] = (parallel_df[col] - parallel_df[col].min()) / (parallel_df[col].max() - parallel_df[col].min()) * 10
    
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
    st.markdown("<p style='text-align: center; color: #666;'>💡 <strong>How to use:</strong> Drag on axes to filter | Green lines = high wellbeing | Red lines = low wellbeing</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Radar Charts (FIXED)
    st.markdown("### 🎯 Lifestyle Profile Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_wb = df[df['Overall_Wellbeing_Score'] >= 7]
        
        if len(high_wb) > 0:
            categories = ['Sleep', 'Low Stress', 'Activity', 'Social', 'Diet Quality']
            
            # Create mapping dictionaries
            social_map = {'Low': 3, 'Moderate': 6, 'High': 10}
            diet_map = {'Poor': 2, 'Fair': 4, 'Good': 6, 'Very Good': 8, 'Excellent': 10}
            
            # Convert to numeric before calculating mean
            social_numeric = high_wb['Social_Interaction_Freq'].map(social_map)
            diet_numeric = high_wb['Diet_Quality'].map(diet_map)
            
            high_values = [
                high_wb['Hours_of_Sleep'].mean() / 9 * 10,
                (10 - high_wb['Stress_Level'].mean()),
                high_wb['Physical_Activity'].mean() / high_wb['Physical_Activity'].max() * 10,
                social_numeric.mean(),
                diet_numeric.mean()
            ]
            
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
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10])
                ),
                showlegend=True,
                title='High Wellbeing Profile',
                height=400,
                font={'family': 'Poppins'},
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="radar_high")
    
    with col2:
        low_wb = df[df['Overall_Wellbeing_Score'] < 5]
        
        if len(low_wb) > 0:
            social_numeric_low = low_wb['Social_Interaction_Freq'].map(social_map)
            diet_numeric_low = low_wb['Diet_Quality'].map(diet_map)
            
            low_values = [
                low_wb['Hours_of_Sleep'].mean() / 9 * 10,
                (10 - low_wb['Stress_Level'].mean()),
                low_wb['Physical_Activity'].mean() / low_wb['Physical_Activity'].max() * 10,
                social_numeric_low.mean(),
                diet_numeric_low.mean()
            ]
            
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
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10])
                ),
                showlegend=True,
                title='Low Wellbeing Profile',
                height=400,
                font={'family': 'Poppins'},
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="radar_low")
    
    st.markdown("<p style='text-align: center; color: #666;'>📊 <strong>Comparison:</strong> Larger area = better lifestyle factors</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Heatmap
    st.markdown("### ⏰ Sleep vs Work Hours Impact Matrix")
    
    df_temp = df.copy()
    df_temp['Sleep_Bin'] = pd.cut(df_temp['Hours_of_Sleep'], bins=5, labels=['Very Low', 'Low', 'Medium', 'Good', 'Excellent'])
    df_temp['Work_Bin'] = pd.cut(df_temp['Work_Hours_per_Day'], bins=5, labels=['<6h', '6-7h', '7-8h', '8-9h', '9+h'])
    
    heatmap_data = df_temp.groupby(['Sleep_Bin', 'Work_Bin'])['Overall_Wellbeing_Score'].mean().unstack()
    
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
    st.markdown("<p style='text-align: center; color: #666;'>🟢 Green = High wellbeing | 🔴 Red = Low wellbeing</p>", 
               unsafe_allow_html=True)

# ============================================================================
# PAGE 3: SEGMENT DEEP DIVE
# ============================================================================

def page_segments(df):
    """Advanced segment analysis"""
    st.markdown("# 🔍 Segment Intelligence")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Deep dive into population segments</p>", 
                unsafe_allow_html=True)
    
    # Waterfall Chart
    st.markdown("### 💧 Wellbeing Contributors Analysis")
    
    baseline = df['Overall_Wellbeing_Score'].mean()
    
    factors = {}
    factors['Baseline'] = baseline
    factors['Good Sleep (7+h)'] = df[df['Hours_of_Sleep'] >= 7]['Overall_Wellbeing_Score'].mean() - baseline
    factors['Low Stress (<5)'] = df[df['Stress_Level'] < 5]['Overall_Wellbeing_Score'].mean() - baseline
    factors['Has Friends'] = df[df['Has_Close_Friends'] == True]['Overall_Wellbeing_Score'].mean() - baseline
    factors['High Activity'] = df[df['Physical_Activity'] > df['Physical_Activity'].median()]['Overall_Wellbeing_Score'].mean() - baseline
    
    # Get top 2 diet qualities
    top_diets = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].mean().nlargest(2).index.tolist()
    factors['Good Diet'] = df[df['Diet_Quality'].isin(top_diets)]['Overall_Wellbeing_Score'].mean() - baseline
    
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
    st.markdown("<p style='text-align: center; color: #666;'>📊 Shows how each factor contributes to overall wellbeing</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sankey Diagram
    st.markdown("### 🌊 Lifestyle → Wellbeing Flow")
    
    df_sankey = df.copy()
    df_sankey['Wellbeing_Cat'] = pd.cut(df_sankey['Overall_Wellbeing_Score'], 
                                        bins=[0, 4, 7, 10], 
                                        labels=['Low', 'Medium', 'High'])
    
    flows = []
    
    # Limit to top 3 diet categories for clarity
    top_3_diets = df_sankey['Diet_Quality'].value_counts().nlargest(3).index.tolist()
    
    for diet in top_3_diets:
        for wb in ['Low', 'Medium', 'High']:
            count = len(df_sankey[(df_sankey['Diet_Quality'] == diet) & (df_sankey['Wellbeing_Cat'] == wb)])
            if count > 0:
                flows.append({'source': diet, 'target': f'Wellbeing: {wb}', 'value': count})
    
    all_nodes = list(set([f['source'] for f in flows] + [f['target'] for f in flows]))
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=['#667eea', '#764ba2', '#f093fb', '#4CAF50', '#FFC107', '#F44336']
        ),
        link=dict(
            source=[node_dict[f['source']] for f in flows],
            target=[node_dict[f['target']] for f in flows],
            value=[f['value'] for f in flows],
            color='rgba(102, 126, 234, 0.3)'
        )
    )])
    
    fig.update_layout(
        title="Diet Quality → Wellbeing Flow",
        height=500,
        font={'family': 'Poppins', 'size': 12},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="sankey_flow")
    st.markdown("<p style='text-align: center; color: #666;'>💡 Thicker flows = more people following that path</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Box Plots
    st.markdown("### 📦 Statistical Comparison: Gender & Friendship Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        for gender in sorted(df['Gender'].unique()):
            gender_data = df[df['Gender'] == gender]['Overall_Wellbeing_Score']
            fig.add_trace(go.Box(
                y=gender_data,
                name=gender,
                boxmean='sd',
                marker_color='#667eea' if gender == sorted(df['Gender'].unique())[0] else '#f5576c'
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
        
        for friend_status in ['Has Friends', 'No Friends']:
            friend_data = df_temp[df_temp['Friends_Label'] == friend_status]['Overall_Wellbeing_Score']
            fig.add_trace(go.Box(
                y=friend_data,
                name=friend_status,
                boxmean='sd',
                marker_color='#4CAF50' if friend_status == 'Has Friends' else '#F44336'
            ))
        
        fig.update_layout(
            title='Wellbeing Distribution by Friendship',
            yaxis_title='Wellbeing Score',
            height=400,
            showlegend=True,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="box_friends")
    
    st.markdown("<p style='text-align: center; color: #666;'>📊 Box shows middle 50% | Line = median | Diamond = mean</p>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Segment Tables
    st.markdown("### 🏆 Top & Bottom Performing Segments")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🌟 Highest Wellbeing Segments")
        
        segments = []
        
        for gender in sorted(df['Gender'].unique()):
            subset = df[df['Gender'] == gender]
            segments.append({
                'Segment': f'Gender: {gender}',
                'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                'Count': len(subset),
                'Avg Stress': subset['Stress_Level'].mean()
            })
        
        for social in sorted(df['Social_Interaction_Freq'].unique()):
            subset = df[df['Social_Interaction_Freq'] == social]
            segments.append({
                'Segment': f'Social: {social}',
                'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                'Count': len(subset),
                'Avg Stress': subset['Stress_Level'].mean()
            })
        
        for friends in [True, False]:
            subset = df[df['Has_Close_Friends'] == friends]
            label = 'Has Friends' if friends else 'No Friends'
            segments.append({
                'Segment': label,
                'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
                'Count': len(subset),
                'Avg Stress': subset['Stress_Level'].mean()
            })
        
        segments_df = pd.DataFrame(segments).sort_values('Avg Wellbeing', ascending=False).head(5)
        segments_df['Avg Wellbeing'] = segments_df['Avg Wellbeing'].round(2)
        segments_df['Avg Stress'] = segments_df['Avg Stress'].round(2)
        
        st.dataframe(segments_df, hide_index=True, use_container_width=True)
    
    with col4:
        st.markdown("#### ⚠️ Highest Stress Segments")
        
        segments_stress = pd.DataFrame(segments).sort_values('Avg Stress', ascending=False).head(5)
        segments_stress['Avg Wellbeing'] = segments_stress['Avg Wellbeing'].round(2)
        segments_stress['Avg Stress'] = segments_stress['Avg Stress'].round(2)
        
        st.dataframe(segments_stress, hide_index=True, use_container_width=True)

# ============================================================================
# PAGE 4: CORRELATIONS
# ============================================================================

def page_correlations(df):
    """Advanced correlation analysis"""
    st.markdown("# 📈 Correlation Intelligence")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Discover hidden relationships in the data</p>", 
                unsafe_allow_html=True)
    
    numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                    'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
    
    corr_df = df[numeric_cols].corr()
    
    # Correlation Heatmap
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
            <li><strong>+1 (Dark Blue):</strong> Perfect positive correlation</li>
            <li><strong>-1 (Dark Red):</strong> Perfect negative correlation</li>
            <li><strong>0 (White):</strong> No correlation</li>
            <li><strong>Hover</strong> over cells for exact values</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Correlations
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
    
    # Data Quality
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

# ============================================================================
# PAGE 5: ML PREDICTOR (Continued in next message due to length)
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
    """ML predictor page"""
    st.markdown("# 🤖 AI Wellbeing Predictor")
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Powered by Random Forest Machine Learning</p>", 
                unsafe_allow_html=True)
    
    model, label_encoders, feature_cols, rmse, r2, y_test, y_pred = train_ml_model(df)
    
    # Model Performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 R² Score", f"{r2:.3f}")
    with col2:
        st.metric("📊 RMSE", f"{rmse:.3f}")
    with col3:
        accuracy = max(0, (1 - (rmse / df['Overall_Wellbeing_Score'].std())) * 100)
        st.metric("✅ Accuracy", f"{accuracy:.0f}%")
    with col4:
        st.metric("📈 Predictions", f"{len(y_test)}")
    
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
            showscale=True
        ),
        text=np.round(feature_importance['Importance'], 3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Importance',
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
    st.markdown("### 📊 Model Accuracy")
    
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
                opacity=0.6
            )
        ))
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="actual_pred")
    
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
                opacity=0.6
            )
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="green")
        
        fig.update_layout(
            title='Prediction Errors',
            xaxis_title='Predicted',
            yaxis_title='Error',
            height=400,
            font={'family': 'Poppins'},
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="residuals")
    
    st.markdown("---")
    
    # Interactive Predictor
    st.markdown("### 🎮 Try the Predictor")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal**")
            input_age = st.slider("Age", 20, 35, 27)
            input_gender = st.selectbox("Gender", sorted(df['Gender'].unique().tolist()))
            input_friends = st.selectbox("Friends?", ['Yes', 'No'])
        
        with col2:
            st.markdown("**😴 Health**")
            input_sleep = st.slider("Sleep", 5.0, 9.0, 7.0, 0.5)
            input_stress = st.slider("Stress", 1, 10, 5)
            input_activity = st.slider("Activity", 
                                      int(df['Physical_Activity'].min()), 
                                      int(df['Physical_Activity'].max()), 
                                      int(df['Physical_Activity'].mean()))
        
        with col3:
            st.markdown("**💼 Lifestyle**")
            input_work = st.slider("Work", 5.0, 11.0, 8.0, 0.5)
            input_screen = st.slider("Screen", 0.0, 12.0, 4.0, 0.5)
            input_diet = st.selectbox("Diet", sorted(df['Diet_Quality'].unique().tolist()))
        
        col4, col5 = st.columns(2)
        with col4:
            input_social = st.selectbox("Social", sorted(df['Social_Interaction_Freq'].unique().tolist()))
        with col5:
            input_substance = st.selectbox("Substance", sorted(df['Substance_Use'].unique().tolist()))
        
        input_physical = st.selectbox("Physical Health", sorted(df['Physical_Health_Condition'].unique().tolist()))
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")
    
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
        st.markdown("### 🎯 Your Prediction")
        
        if prediction >= 7:
            color, status, emoji = "#4CAF50", "Excellent", "🌟"
        elif prediction >= 5:
            color, status, emoji = "#FFC107", "Good", "👍"
        else:
            color, status, emoji = "#F44336", "Needs Attention", "⚠️"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    border-radius: 20px; border: 4px solid {color}; margin: 20px 0;'>
            <h1 style='font-size: 72px; margin: 0; color: {color};'>{prediction:.1f}</h1>
            <p style='font-size: 20px; color: #666; margin: 10px 0;'>out of 10</p>
            <h2 style='margin: 15px 0; color: {color};'>{emoji} {status}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recs = []
        if input_sleep < 7:
            recs.append("😴 **Sleep More:** Aim for 7-8 hours")
        if input_stress >= 7:
            recs.append("🧘 **Manage Stress:** Try meditation or exercise")
        if input_screen > 6:
            recs.append("📱 **Reduce Screen Time:** Take regular breaks")
        if input_friends == 'No':
            recs.append("👥 **Build Connections:** Social relationships matter")
        
        if not recs:
            recs.append("✅ **Great Job!** Keep it up!")
        
        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec}")
        
        # Comparison
        st.markdown("---")
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            avg = df['Overall_Wellbeing_Score'].mean()
            st.metric("vs Average", f"{prediction:.1f}", f"{prediction - avg:+.1f}")
        
        with col_comp2:
            percentile = (df['Overall_Wellbeing_Score'] < prediction).sum() / len(df) * 100
            st.metric("Percentile", f"{percentile:.0f}%")
        
        with col_comp3:
            similar = len(df[(df['Overall_Wellbeing_Score'] >= prediction - 0.5) & 
                            (df['Overall_Wellbeing_Score'] <= prediction + 0.5)])
            st.metric("Similar People", f"{similar}")

# ============================================================================
# MAIN
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
        st.warning("⚠️ No data matches filters")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Records", f"{len(df_filtered):,} / {len(df):,}")
    
    st.sidebar.markdown("---")
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        "📥 Download CSV",
        csv,
        "data.csv",
        "text/csv",
        use_container_width=True
    )
    
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **Mental Health Analytics v3.0**
        
        - 15+ Interactive Charts
        - AI Predictions
        - Real-time Filtering
        
        Built with ❤️
        """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🏃 Lifestyle",
        "🔍 Segments",
        "📈 Correlations",
        "🤖 AI Predictor"
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
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white;'>
        <p style='margin: 0; font-weight: 600;'>🧠 Mental Health Analytics v3.0</p>
        <p style='margin: 5px 0 0 0; font-size: 13px;'>For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
