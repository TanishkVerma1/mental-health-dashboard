import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
# IMPROVED CSS - BETTER READABILITY
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f8f9fa;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #4F46E5;
    }
    
    .stMetric label {
        color: #6B7280 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    h1 {
        color: #111827;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        font-size: 1.8rem;
    }
    
    h3 {
        color: #4B5563;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .insight-box h3, .insight-box h4 {
        color: white !important;
        margin-top: 0;
    }
    
    .explanation-box {
        background: #F3F4F6;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #4F46E5;
        margin-top: 12px;
        color: #374151;
    }
    
    .explanation-box strong {
        color: #111827;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        border-radius: 8px;
        color: #6B7280;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4F46E5 !important;
        color: white !important;
    }
    
    .stButton>button {
        background: #4F46E5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4F46E5 0%, #7C3AED 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_and_preprocess_data():
    """Load and preprocess data"""
    try:
        df = pd.read_csv('data/Mental_Health_and_Lifestyle_Research.csv')
        df.columns = df.columns.str.strip()
        
        categorical_cols = ['Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
        
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
    """Create sidebar"""
    st.sidebar.title("🎛️ Control Panel")
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
# KPI CARDS
# ============================================================================

def display_kpis(df):
    """Display KPI cards"""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        avg_wb = df['Overall_Wellbeing_Score'].mean()
        st.metric("🎯 Wellbeing", f"{avg_wb:.1f}/10", 
                 f"{avg_wb - 5.5:+.1f}")
    
    with col2:
        avg_stress = df['Stress_Level'].mean()
        st.metric("😰 Stress", f"{avg_stress:.1f}/10",
                 f"{5.5 - avg_stress:+.1f}")
    
    with col3:
        avg_sleep = df['Hours_of_Sleep'].mean()
        st.metric("😴 Sleep", f"{avg_sleep:.1f}h",
                 f"{avg_sleep - 7:+.1f}h")
    
    with col4:
        avg_screen = df['Screen_Time_per_Day'].mean()
        st.metric("📱 Screen", f"{avg_screen:.1f}h",
                 f"{4 - avg_screen:+.1f}h")
    
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
    """Overview dashboard"""
    st.title("📊 Executive Dashboard")
    st.markdown(f"**Analyzing {len(df):,} individuals**")
    st.markdown("---")
    
    display_kpis(df)
    st.markdown("---")
    
    # Row 1: Gauge Charts
    col1, col2 = st.columns(2)
    
    with col1:
        avg_wellbeing = df['Overall_Wellbeing_Score'].mean()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_wellbeing,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Wellbeing Score", 'font': {'size': 20}},
            delta={'reference': 5.5},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "#4F46E5"},
                'steps': [
                    {'range': [0, 3], 'color': "#FEE2E2"},
                    {'range': [3, 5], 'color': "#FEF3C7"},
                    {'range': [5, 7], 'color': "#D1FAE5"},
                    {'range': [7, 10], 'color': "#A7F3D0"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 7}
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True, key="gauge_wb")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> The gauge displays the average wellbeing score across all filtered individuals. 
        Scores above 7 (green zone) indicate excellent wellbeing, while scores below 5 (red/yellow zones) suggest areas needing attention.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_stress = df['Stress_Level'].mean()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_stress,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Stress Level", 'font': {'size': 20}},
            delta={'reference': 5.5},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "#DC2626"},
                'steps': [
                    {'range': [0, 3], 'color': "#A7F3D0"},
                    {'range': [3, 5], 'color': "#D1FAE5"},
                    {'range': [5, 7], 'color': "#FEF3C7"},
                    {'range': [7, 10], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 3}
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True, key="gauge_stress")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This gauge measures average stress levels. Lower is better! 
        Stress below 3 (green) is healthy, while levels above 7 (red) indicate high stress requiring intervention.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Distribution Charts
    st.subheader("📊 Population Distribution Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Wellbeing distribution with statistics
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['Overall_Wellbeing_Score'],
            nbinsx=20,
            name='Distribution',
            marker_color='#4F46E5',
            opacity=0.7
        ))
        
        # Add mean line
        mean_wb = df['Overall_Wellbeing_Score'].mean()
        fig.add_vline(x=mean_wb, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_wb:.1f}", annotation_position="top")
        
        fig.update_layout(
            title='Wellbeing Score Distribution',
            xaxis_title='Wellbeing Score',
            yaxis_title='Number of People',
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="hist_wb")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> This histogram shows how wellbeing scores are distributed across the population. 
        The red dashed line marks the average. A right-skewed distribution (more bars on the right) indicates generally good wellbeing.
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Mental Health Pie Chart
        mental_counts = df['Mental_Health_Status'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=mental_counts.index,
            values=mental_counts.values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Mental Health Status Breakdown',
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key="pie_mental")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> This donut chart shows the proportion of each mental health status in the population. 
        Larger slices indicate more common conditions. Use this to identify which mental health issues are most prevalent.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 3: Treemap - Hierarchical View
    st.subheader("🗺️ Hierarchical Population View")
    
    treemap_data = df.groupby(['Gender', 'Diet_Quality', 'Social_Interaction_Freq']).size().reset_index(name='count')
    
    fig = px.treemap(
        treemap_data,
        path=['Gender', 'Diet_Quality', 'Social_Interaction_Freq'],
        values='count',
        color='count',
        color_continuous_scale='Viridis',
        title='Population Breakdown: Gender → Diet → Social Interaction'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="treemap")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 How to use:</strong> This treemap shows the hierarchical breakdown of the population. 
    Larger rectangles = more people. Click on any rectangle to zoom in and explore that segment in detail. 
    The color intensity shows the size of each group.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("🔍 Key Insights")
    
    avg_wb = df['Overall_Wellbeing_Score'].mean()
    avg_stress = df['Stress_Level'].mean()
    avg_sleep = df['Hours_of_Sleep'].mean()
    best_diet = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].mean().idxmax()
    pct_friends = (df['Has_Close_Friends'].sum() / len(df)) * 100
    
    insights_html = f"""
    <div class="insight-box">
        <h3>📊 Current Population Summary</h3>
        <ul style="font-size: 1.05rem; line-height: 1.8;">
            <li><strong>Average Wellbeing:</strong> {avg_wb:.1f}/10 - {'🟢 Excellent (Above 7)' if avg_wb >= 7 else '🟡 Good (5-7)' if avg_wb >= 5 else '🔴 Needs Attention (Below 5)'}</li>
            <li><strong>Average Stress:</strong> {avg_stress:.1f}/10 - {'🔴 High (Above 7)' if avg_stress >= 7 else '🟡 Moderate (5-7)' if avg_stress >= 5 else '🟢 Healthy (Below 5)'}</li>
            <li><strong>Sleep Pattern:</strong> {avg_sleep:.1f} hours average - {'✅ Adequate (7+ hours)' if avg_sleep >= 7 else '⚠️ Below recommended (Less than 7 hours)'}</li>
            <li><strong>Best Diet for Wellbeing:</strong> {best_diet} diet shows the highest wellbeing scores</li>
            <li><strong>Social Connection:</strong> {pct_friends:.0f}% have close friends - {'Strong social network' if pct_friends >= 70 else 'Limited social connections'}</li>
        </ul>
    </div>
    """
    st.markdown(insights_html, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: LIFESTYLE INTELLIGENCE (COMPLETELY REDESIGNED)
# ============================================================================

def page_lifestyle(df):
    """Redesigned lifestyle analysis with clear charts"""
    st.title("🏃 Lifestyle Intelligence")
    st.markdown("**Understand how daily habits impact mental wellbeing**")
    st.markdown("---")
    
    # Chart 1: Simple Bar Chart - Average Wellbeing by Key Factors
    st.subheader("📊 Wellbeing Scores Across Lifestyle Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Diet Quality Impact
        diet_wb = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].agg(['mean', 'count']).reset_index()
        diet_wb = diet_wb.sort_values('mean', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=diet_wb['Diet_Quality'],
            y=diet_wb['mean'],
            text=diet_wb['mean'].round(2),
            textposition='outside',
            marker=dict(
                color=diet_wb['mean'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Wellbeing")
            ),
            hovertemplate='<b>%{x}</b><br>Avg Wellbeing: %{y:.2f}<br>Count: %{customdata}<extra></extra>',
            customdata=diet_wb['count']
        ))
        
        fig.update_layout(
            title='Average Wellbeing by Diet Quality',
            xaxis_title='Diet Quality',
            yaxis_title='Average Wellbeing Score',
            height=400,
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_diet")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This bar chart compares average wellbeing scores across different diet qualities. 
        Taller bars = better wellbeing. The color gradient (green = high, red = low) makes it easy to spot the best diet choices.
        <br><strong>💡 Insight:</strong> People with better diet quality consistently report higher wellbeing scores.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Social Interaction Impact
        social_wb = df.groupby('Social_Interaction_Freq')['Overall_Wellbeing_Score'].agg(['mean', 'count']).reset_index()
        social_order = {'Low': 0, 'Moderate': 1, 'High': 2}
        social_wb['order'] = social_wb['Social_Interaction_Freq'].map(social_order)
        social_wb = social_wb.sort_values('order')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=social_wb['Social_Interaction_Freq'],
            y=social_wb['mean'],
            text=social_wb['mean'].round(2),
            textposition='outside',
            marker=dict(
                color=['#EF4444', '#F59E0B', '#10B981'],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Avg Wellbeing: %{y:.2f}<br>Count: %{customdata}<extra></extra>',
            customdata=social_wb['count']
        ))
        
        fig.update_layout(
            title='Wellbeing by Social Interaction Frequency',
            xaxis_title='Social Interaction Level',
            yaxis_title='Average Wellbeing Score',
            height=400,
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_social")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This chart reveals the relationship between social interaction frequency and wellbeing. 
        The progression from red (low) to green (high) shows the trend clearly.
        <br><strong>💡 Insight:</strong> Higher social interaction is strongly associated with better mental wellbeing.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 2: Grouped Bar Chart - Stress Levels by Multiple Factors
    st.subheader("😰 Stress Level Comparison")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Stress by Sleep Categories
        df_temp = df.copy()
        df_temp['Sleep_Category'] = pd.cut(df_temp['Hours_of_Sleep'], 
                                           bins=[0, 6, 7, 8, 10], 
                                           labels=['<6h', '6-7h', '7-8h', '8+h'])
        
        sleep_stress = df_temp.groupby('Sleep_Category')['Stress_Level'].agg(['mean', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sleep_stress['Sleep_Category'],
            y=sleep_stress['mean'],
            text=sleep_stress['mean'].round(2),
            textposition='outside',
            marker=dict(
                color=sleep_stress['mean'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Stress")
            ),
            hovertemplate='<b>%{x}</b><br>Avg Stress: %{y:.2f}<br>Count: %{customdata}<extra></extra>',
            customdata=sleep_stress['count']
        ))
        
        fig.update_layout(
            title='Average Stress by Sleep Duration',
            xaxis_title='Sleep Duration',
            yaxis_title='Average Stress Level',
            height=400,
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_sleep_stress")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This chart shows how sleep duration affects stress levels. 
        Lower bars (green) = less stress, which is good. Higher bars (red) = more stress.
        <br><strong>💡 Insight:</strong> People sleeping 7-8 hours typically have the lowest stress levels.
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Stress by Work Hours
        df_temp['Work_Category'] = pd.cut(df_temp['Work_Hours_per_Day'], 
                                         bins=[0, 7, 8, 9, 12], 
                                         labels=['<7h', '7-8h', '8-9h', '9+h'])
        
        work_stress = df_temp.groupby('Work_Category')['Stress_Level'].agg(['mean', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=work_stress['Work_Category'],
            y=work_stress['mean'],
            text=work_stress['mean'].round(2),
            textposition='outside',
            marker=dict(
                color=['#10B981', '#F59E0B', '#F97316', '#EF4444'],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Avg Stress: %{y:.2f}<br>Count: %{customdata}<extra></extra>',
            customdata=work_stress['count']
        ))
        
        fig.update_layout(
            title='Stress Levels by Work Hours',
            xaxis_title='Daily Work Hours',
            yaxis_title='Average Stress Level',
            height=400,
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_work_stress")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This chart demonstrates the relationship between work hours and stress. 
        The color progression from green to red shows increasing stress with longer work hours.
        <br><strong>💡 Insight:</strong> Working more than 9 hours per day is associated with significantly higher stress.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 3: Line Chart - Wellbeing Trends
    st.subheader("📈 Wellbeing Trends Across Age Groups")
    
    age_bins = pd.cut(df['Age'], bins=5)
    age_analysis = df.groupby(age_bins).agg({
        'Overall_Wellbeing_Score': 'mean',
        'Stress_Level': 'mean',
        'Hours_of_Sleep': 'mean'
    }).reset_index()
    
    age_analysis['Age_Range'] = age_analysis['Age'].astype(str)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=age_analysis['Age_Range'],
        y=age_analysis['Overall_Wellbeing_Score'],
        mode='lines+markers',
        name='Wellbeing',
        line=dict(color='#10B981', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=age_analysis['Age_Range'],
        y=age_analysis['Stress_Level'],
        mode='lines+markers',
        name='Stress',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Wellbeing and Stress Trends by Age Group',
        xaxis_title='Age Range',
        yaxis_title='Score (1-10)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="line_age")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This line chart tracks how wellbeing (green) and stress (red) change across different age groups. 
    Lines going up = increasing, lines going down = decreasing.
    <br><strong>💡 Insight:</strong> Look for age groups where wellbeing is high and stress is low - these are the "sweet spots."
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 4: Scatter with Size - Screen Time Impact
    st.subheader("📱 Screen Time Impact on Wellbeing")
    
    fig = px.scatter(
        df,
        x='Screen_Time_per_Day',
        y='Overall_Wellbeing_Score',
        size='Physical_Activity',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        title='Screen Time vs Wellbeing (Bubble size = Physical Activity)',
        labels={
            'Screen_Time_per_Day': 'Daily Screen Time (hours)',
            'Overall_Wellbeing_Score': 'Wellbeing Score',
            'Physical_Activity': 'Physical Activity',
            'Stress_Level': 'Stress Level'
        },
        size_max=20,
        opacity=0.6
    )
    
    # Add trendline
    z = np.polyfit(df['Screen_Time_per_Day'].dropna(), df['Overall_Wellbeing_Score'].dropna(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Screen_Time_per_Day'].min(), df['Screen_Time_per_Day'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode='lines',
        name='Trend Line',
        line=dict(color='black', width=3, dash='dash')
    ))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="scatter_screen")
    
    corr = df[['Screen_Time_per_Day', 'Overall_Wellbeing_Score']].corr().iloc[0, 1]
    
    st.markdown(f"""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> Each bubble represents a person. Position shows screen time (x-axis) and wellbeing (y-axis). 
    Bubble size = physical activity level. Color = stress (green = low, red = high). The dashed line shows the overall trend.
    <br><strong>💡 Insight:</strong> Correlation = {corr:.3f}. {'Negative correlation suggests more screen time may reduce wellbeing' if corr < -0.1 else 'Weak relationship between screen time and wellbeing' if abs(corr) < 0.1 else 'Positive correlation suggests screen time may improve wellbeing'}.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: SEGMENT INTELLIGENCE (COMPLETELY REDESIGNED)
# ============================================================================

def page_segments(df):
    """Redesigned segment analysis"""
    st.title("🔍 Segment Intelligence")
    st.markdown("**Compare different population groups to identify patterns**")
    st.markdown("---")
    
    # Chart 1: Side-by-Side Comparison - Gender
    st.subheader("⚥ Gender Comparison")
    
    gender_stats = df.groupby('Gender').agg({
        'Overall_Wellbeing_Score': 'mean',
        'Stress_Level': 'mean',
        'Hours_of_Sleep': 'mean',
        'Physical_Activity': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    metrics = ['Overall_Wellbeing_Score', 'Stress_Level', 'Hours_of_Sleep', 'Physical_Activity']
    metric_names = ['Wellbeing', 'Stress', 'Sleep (hrs)', 'Activity']
    
    for i, gender in enumerate(gender_stats['Gender']):
        fig.add_trace(go.Bar(
            name=gender,
            x=metric_names,
            y=gender_stats.iloc[i][metrics].values,
            text=gender_stats.iloc[i][metrics].round(2).values,
            textposition='outside',
            marker_color=['#4F46E5', '#EC4899'][i]
        ))
    
    fig.update_layout(
        title='Gender Comparison Across Key Metrics',
        xaxis_title='Metric',
        yaxis_title='Average Score',
        barmode='group',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, key="bar_gender_comp")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This grouped bar chart compares males and females across four key metrics. 
    Bars side-by-side make it easy to spot differences. Taller bars = higher values.
    <br><strong>💡 How to read:</strong> For Wellbeing, Sleep, and Activity - higher is better. For Stress - lower is better.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 2: Stacked Bar - Mental Health by Social Interaction
    st.subheader("🧠 Mental Health Distribution by Social Interaction")
    
    mental_social = df.groupby(['Social_Interaction_Freq', 'Mental_Health_Status']).size().reset_index(name='count')
    mental_social_pivot = mental_social.pivot(index='Social_Interaction_Freq', 
                                              columns='Mental_Health_Status', 
                                              values='count').fillna(0)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    for i, col in enumerate(mental_social_pivot.columns):
        fig.add_trace(go.Bar(
            name=col,
            x=mental_social_pivot.index,
            y=mental_social_pivot[col],
            marker_color=colors[i % len(colors)],
            text=mental_social_pivot[col].astype(int),
            textposition='inside'
        ))
    
    fig.update_layout(
        title='Mental Health Status Distribution by Social Interaction Level',
        xaxis_title='Social Interaction Frequency',
        yaxis_title='Number of People',
        barmode='stack',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True, key="stacked_mental_social")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This stacked bar chart shows how mental health conditions are distributed across different social interaction levels. 
    Each color represents a different mental health status. The height of each segment shows how many people have that condition.
    <br><strong>💡 Insight:</strong> Compare the proportions across Low, Moderate, and High social interaction to see which conditions are more common in isolated vs. social individuals.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 3: Heatmap - Wellbeing by Multiple Factors
    st.subheader("🔥 Wellbeing Heatmap: Diet × Social Interaction")
    
    heatmap_data = df.groupby(['Diet_Quality', 'Social_Interaction_Freq'])['Overall_Wellbeing_Score'].mean().unstack()
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        text=np.round(heatmap_data.values, 2),
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        colorbar=dict(title="Wellbeing<br>Score"),
        hovertemplate='Diet: %{y}<br>Social: %{x}<br>Wellbeing: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Wellbeing Score by Diet Quality and Social Interaction',
        xaxis_title='Social Interaction Frequency',
        yaxis_title='Diet Quality',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True, key="heatmap_diet_social")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This heatmap reveals wellbeing scores for every combination of diet quality and social interaction. 
    Green cells = high wellbeing, Red cells = low wellbeing. The numbers show exact scores.
    <br><strong>💡 How to use:</strong> Find the greenest cell to identify the best combination of diet and social interaction for optimal wellbeing.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 4: Funnel Chart - Wellbeing Segments
    st.subheader("📊 Population Segments by Wellbeing Level")
    
    df_temp = df.copy()
    df_temp['Wellbeing_Category'] = pd.cut(df_temp['Overall_Wellbeing_Score'], 
                                           bins=[0, 3, 5, 7, 10], 
                                           labels=['Poor (0-3)', 'Fair (3-5)', 'Good (5-7)', 'Excellent (7-10)'])
    
    wellbeing_counts = df_temp['Wellbeing_Category'].value_counts().sort_index(ascending=False)
    
    fig = go.Figure(go.Funnel(
        y=wellbeing_counts.index,
        x=wellbeing_counts.values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=['#10B981', '#84CC16', '#F59E0B', '#EF4444']),
        connector={"line": {"color": "#6B7280", "width": 2}}
    ))
    
    fig.update_layout(
        title='Population Distribution by Wellbeing Category',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True, key="funnel_wellbeing")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This funnel chart shows how many people fall into each wellbeing category. 
    The width of each section represents the number of people. Percentages show the proportion of the total population.
    <br><strong>💡 Insight:</strong> A healthy population should have most people in "Good" or "Excellent" categories (top sections).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 5: Comparison Table with Highlights
    st.subheader("📋 Detailed Segment Comparison Table")
    
    segments = []
    
    # By Gender
    for gender in sorted(df['Gender'].unique()):
        subset = df[df['Gender'] == gender]
        segments.append({
            'Segment Type': 'Gender',
            'Segment': gender,
            'Count': len(subset),
            'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
            'Avg Stress': subset['Stress_Level'].mean(),
            'Avg Sleep': subset['Hours_of_Sleep'].mean(),
            '% With Friends': (subset['Has_Close_Friends'].sum() / len(subset) * 100)
        })
    
    # By Social Interaction
    for social in sorted(df['Social_Interaction_Freq'].unique()):
        subset = df[df['Social_Interaction_Freq'] == social]
        segments.append({
            'Segment Type': 'Social',
            'Segment': social,
            'Count': len(subset),
            'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
            'Avg Stress': subset['Stress_Level'].mean(),
            'Avg Sleep': subset['Hours_of_Sleep'].mean(),
            '% With Friends': (subset['Has_Close_Friends'].sum() / len(subset) * 100)
        })
    
    # By Friends
    for friends in [True, False]:
        subset = df[df['Has_Close_Friends'] == friends]
        label = 'Has Friends' if friends else 'No Friends'
        segments.append({
            'Segment Type': 'Friendship',
            'Segment': label,
            'Count': len(subset),
            'Avg Wellbeing': subset['Overall_Wellbeing_Score'].mean(),
            'Avg Stress': subset['Stress_Level'].mean(),
            'Avg Sleep': subset['Hours_of_Sleep'].mean(),
            '% With Friends': (subset['Has_Close_Friends'].sum() / len(subset) * 100)
        })
    
    segments_df = pd.DataFrame(segments)
    segments_df = segments_df.round(2)
    segments_df = segments_df.sort_values('Avg Wellbeing', ascending=False)
    
    # Style the dataframe
    st.dataframe(
        segments_df.style.background_gradient(subset=['Avg Wellbeing'], cmap='RdYlGn', vmin=0, vmax=10)
                        .background_gradient(subset=['Avg Stress'], cmap='RdYlGn_r', vmin=0, vmax=10)
                        .background_gradient(subset=['% With Friends'], cmap='Blues', vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This table provides a detailed comparison of all segments with color-coded cells. 
    Green = good, Red = bad. The color intensity helps you quickly identify the best and worst performing segments.
    <br><strong>💡 How to use:</strong> Sort by any column to find top performers. Look for patterns - do certain segment types consistently perform better?
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4: CORRELATION INTELLIGENCE (REDESIGNED)
# ============================================================================

def page_correlations(df):
    """Redesigned correlation analysis"""
    st.title("📈 Correlation Intelligence")
    st.markdown("**Discover meaningful relationships between variables**")
    st.markdown("---")
    
    numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                    'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
    
    corr_df = df[numeric_cols].corr()
    
    # Chart 1: Enhanced Correlation Heatmap with Better Scale
    st.subheader("🔥 Correlation Heatmap")
    
    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    corr_masked = corr_df.mask(mask)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale=[
            [0, '#DC2626'],      # Strong negative - Red
            [0.25, '#F87171'],   # Moderate negative - Light red
            [0.4, '#FEF3C7'],    # Weak negative - Yellow
            [0.5, '#FFFFFF'],    # No correlation - White
            [0.6, '#D1FAE5'],    # Weak positive - Light green
            [0.75, '#34D399'],   # Moderate positive - Green
            [1, '#059669']       # Strong positive - Dark green
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_masked.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12, "color": "black"},
        colorbar=dict(
            title="Correlation<br>Coefficient",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['Strong<br>Negative<br>(-1)', 'Moderate<br>Negative', 'None<br>(0)', 'Moderate<br>Positive', 'Strong<br>Positive<br>(+1)']
        ),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Correlation Matrix (Lower Triangle Only)',
        height=600,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This heatmap shows how strongly each pair of variables is related. 
    <ul>
        <li><strong>Dark Green (+0.5 to +1):</strong> Strong positive relationship - when one increases, the other increases</li>
        <li><strong>Light Green (0 to +0.5):</strong> Weak positive relationship</li>
        <li><strong>White (0):</strong> No relationship</li>
        <li><strong>Light Red (0 to -0.5):</strong> Weak negative relationship</li>
        <li><strong>Dark Red (-0.5 to -1):</strong> Strong negative relationship - when one increases, the other decreases</li>
    </ul>
    <strong>💡 Why it matters:</strong> Strong correlations (dark colors) reveal important relationships. For example, if sleep and wellbeing are strongly correlated (dark green), improving sleep could improve wellbeing.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 2: Top Correlations Bar Chart
    st.subheader("🔝 Strongest Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Top Positive Correlations")
        
        corr_pairs = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                corr_pairs.append({
                    'Pair': f"{corr_df.columns[i]}<br>↔<br>{corr_df.columns[j]}",
                    'Variable 1': corr_df.columns[i],
                    'Variable 2': corr_df.columns[j],
                    'Correlation': corr_df.iloc[i, j]
                })
        
        corr_pairs_df = pd.DataFrame(corr_pairs)
        top_positive = corr_pairs_df.nlargest(5, 'Correlation')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_positive['Pair'],
            x=top_positive['Correlation'],
            orientation='h',
            marker=dict(
                color=top_positive['Correlation'],
                colorscale='Greens',
                showscale=False
            ),
            text=top_positive['Correlation'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 5 Positive Correlations',
            xaxis_title='Correlation Coefficient',
            height=350,
            xaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_pos_corr")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> These pairs of variables move together. 
        Longer bars = stronger relationship. Values closer to 1.0 indicate very strong positive relationships.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⛔ Top Negative Correlations")
        
        top_negative = corr_pairs_df.nsmallest(5, 'Correlation')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_negative['Pair'],
            x=top_negative['Correlation'],
            orientation='h',
            marker=dict(
                color=np.abs(top_negative['Correlation']),
                colorscale='Reds',
                showscale=False
            ),
            text=top_negative['Correlation'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 5 Negative Correlations',
            xaxis_title='Correlation Coefficient',
            height=350,
            xaxis_range=[-1, 0]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="bar_neg_corr")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> These pairs of variables move in opposite directions. 
        Longer bars (more negative) = stronger inverse relationship. Values closer to -1.0 indicate very strong negative relationships.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 3: Scatter Matrix for Key Variables
    st.subheader("🔍 Detailed Relationship Explorer")
    
    key_vars = ['Overall_Wellbeing_Score', 'Stress_Level', 'Hours_of_Sleep', 'Physical_Activity']
    
    fig = px.scatter_matrix(
        df[key_vars],
        dimensions=key_vars,
        color=df['Overall_Wellbeing_Score'],
        color_continuous_scale='RdYlGn',
        title='Scatter Matrix: Key Variables',
        labels={col: col.replace('_', ' ') for col in key_vars},
        height=700
    )
    
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    st.plotly_chart(fig, use_container_width=True, key="scatter_matrix")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This scatter matrix shows the relationship between every pair of key variables. 
    Each small chart is a scatter plot comparing two variables. Points are colored by wellbeing score (green = high, red = low).
    <br><strong>💡 How to use:</strong> Look for patterns in the scatter plots:
    <ul>
        <li><strong>Upward slope:</strong> Positive correlation (both increase together)</li>
        <li><strong>Downward slope:</strong> Negative correlation (one increases, other decreases)</li>
        <li><strong>Random cloud:</strong> No correlation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart 4: Correlation Network (if strong correlations exist)
    st.subheader("🕸️ Correlation Network")
    
    # Filter for strong correlations only
    strong_corr = []
    for i in range(len(corr_df.columns)):
        for j in range(i+1, len(corr_df.columns)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) > 0.3:  # Only show correlations > 0.3
                strong_corr.append({
                    'source': corr_df.columns[i],
                    'target': corr_df.columns[j],
                    'value': abs(corr_val),
                    'correlation': corr_val
                })
    
    if strong_corr:
        # Create a simple network visualization using a chord-like diagram
        all_vars = list(set([c['source'] for c in strong_corr] + [c['target'] for c in strong_corr]))
        var_dict = {var: idx for idx, var in enumerate(all_vars)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_vars,
                color=['#4F46E5'] * len(all_vars)
            ),
            link=dict(
                source=[var_dict[c['source']] for c in strong_corr],
                target=[var_dict[c['target']] for c in strong_corr],
                value=[c['value'] for c in strong_corr],
                color=['rgba(79, 70, 229, 0.3)'] * len(strong_corr),
                customdata=[c['correlation'] for c in strong_corr],
                hovertemplate='%{source.label} ↔ %{target.label}<br>Correlation: %{customdata:.3f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title='Strong Correlations Network (|r| > 0.3)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key="network_corr")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> This network diagram shows only the strong relationships (correlation > 0.3). 
        Thicker connections = stronger correlations. Hover over connections to see exact correlation values.
        <br><strong>💡 Insight:</strong> Variables with many connections are "hub" variables that influence or are influenced by many other factors.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No strong correlations (|r| > 0.3) found in the current filtered data.")
    
    st.markdown("---")
    
    # Data Quality Section
    st.subheader("🔍 Data Quality Metrics")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("📊 Total Records", f"{len(df):,}")
    
    with col4:
        st.metric("📈 Numeric Variables", len(numeric_cols))
    
    with col5:
        completeness = ((df.size - df.isna().sum().sum()) / df.size) * 100
        st.metric("✅ Data Completeness", f"{completeness:.1f}%")
    
    with col6:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")

# ============================================================================
# PAGE 5: AI PREDICTOR (COMPLETELY REDESIGNED)
# ============================================================================

@st.cache_resource(show_spinner="Training AI model...")
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
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    del X_train, X_test, y_train, df_ml
    gc.collect()
    
    return model, label_encoders, feature_cols, rmse, mae, r2, y_test, y_pred

def page_ml_predictor(df):
    """Redesigned AI predictor page"""
    st.title("🤖 AI Wellbeing Predictor & Analyzer")
    st.markdown("**Use machine learning to predict wellbeing and get personalized recommendations**")
    st.markdown("---")
    
    model, label_encoders, feature_cols, rmse, mae, r2, y_test, y_pred = train_ml_model(df)
    
    # Model Performance Section
    st.subheader("📊 Model Performance Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 R² Score", f"{r2:.3f}", 
                 help="How well the model explains the data (0-1, higher is better)")
    
    with col2:
        st.metric("📊 RMSE", f"{rmse:.3f}",
                 help="Average prediction error in wellbeing points")
    
    with col3:
        st.metric("📏 MAE", f"{mae:.3f}",
                 help="Mean absolute error")
    
    with col4:
        accuracy = max(0, (1 - (rmse / df['Overall_Wellbeing_Score'].std())) * 100)
        st.metric("✅ Accuracy", f"{accuracy:.0f}%",
                 help="Relative accuracy estimate")
    
    with col5:
        st.metric("🧪 Test Size", f"{len(y_test)}",
                 help="Number of predictions tested")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🎯 What Factors Matter Most?")
    
    feature_importance = pd.DataFrame({
        'Feature': [col.replace('_encoded', '').replace('_', ' ').title() for col in feature_cols],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        orientation='h',
        marker=dict(
            color=feature_importance['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=np.round(feature_importance['Importance'], 3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Importance: Which Factors Predict Wellbeing Best?',
        xaxis_title='Importance Score',
        yaxis_title='',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True, key="feature_importance")
    
    st.markdown("""
    <div class="explanation-box">
    <strong>📖 What this shows:</strong> This chart ranks all factors by how much they influence wellbeing predictions. 
    Longer bars = more important factors. The AI model uses these factors to make predictions.
    <br><strong>💡 Key insight:</strong> Focus on improving the top 3-5 factors for the biggest impact on wellbeing.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Accuracy Visualization
    st.subheader("📈 Model Accuracy Analysis")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Actual vs Predicted
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
                colorbar=dict(title="Actual<br>Score"),
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ))
        
        # Perfect prediction line
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
            title='Actual vs Predicted Wellbeing Scores',
            xaxis_title='Actual Wellbeing Score',
            yaxis_title='Predicted Wellbeing Score',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key="actual_pred")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> Each dot is a prediction. Points closer to the red line = more accurate predictions. 
        The model is performing well if most points cluster around the line.
        </div>
        """, unsafe_allow_html=True)
    
    with col_chart2:
        # Error Distribution
        errors = y_test - y_pred
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            marker_color='#4F46E5',
            opacity=0.7,
            name='Error Distribution'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Perfect (0 error)", annotation_position="top")
        
        fig.update_layout(
            title='Prediction Error Distribution',
            xaxis_title='Prediction Error (Actual - Predicted)',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        ))
        
        st.plotly_chart(fig, use_container_width=True, key="error_dist")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 Interpretation:</strong> This shows how far off predictions typically are. 
        A bell curve centered at 0 (red line) means the model is unbiased. Narrow distribution = more accurate predictions.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Predictor
    st.subheader("🎮 Try the AI Predictor")
    st.markdown("**Enter your information to get a personalized wellbeing prediction and recommendations**")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal Information**")
            input_age = st.slider("Age", 20, 35, 27)
            input_gender = st.selectbox("Gender", sorted(df['Gender'].unique().tolist()))
            input_friends = st.radio("Do you have close friends?", ['Yes', 'No'])
        
        with col2:
            st.markdown("**😴 Health & Wellness**")
            input_sleep = st.slider("Hours of Sleep per Night", 5.0, 9.0, 7.0, 0.5)
            input_stress = st.slider("Stress Level (1-10)", 1, 10, 5)
            input_activity = st.slider("Physical Activity Score", 
                                      int(df['Physical_Activity'].min()), 
                                      int(df['Physical_Activity'].max()), 
                                      int(df['Physical_Activity'].mean()))
        
        with col3:
            st.markdown("**💼 Daily Habits**")
            input_work = st.slider("Work Hours per Day", 5.0, 11.0, 8.0, 0.5)
            input_screen = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, 0.5)
            input_diet = st.selectbox("Diet Quality", sorted(df['Diet_Quality'].unique().tolist()))
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            input_social = st.selectbox("Social Interaction Frequency", 
                                       sorted(df['Social_Interaction_Freq'].unique().tolist()))
        
        with col5:
            input_substance = st.selectbox("Substance Use", 
                                          sorted(df['Substance_Use'].unique().tolist()))
        
        with col6:
            input_physical = st.selectbox("Physical Health Condition", 
                                         sorted(df['Physical_Health_Condition'].unique().tolist()))
        
        submitted = st.form_submit_button("🔮 Predict My Wellbeing Score", 
                                         use_container_width=True, 
                                         type="primary")
    
    if submitted:
        # Prepare input
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
        
        # Display Result
        st.markdown("---")
        st.markdown("### 🎯 Your Predicted Wellbeing Score")
        
        if prediction >= 7:
            color, status, emoji, bg_color = "#10B981", "Excellent", "🌟", "#D1FAE5"
        elif prediction >= 5:
            color, status, emoji, bg_color = "#F59E0B", "Good", "👍", "#FEF3C7"
        else:
            color, status, emoji, bg_color = "#EF4444", "Needs Attention", "⚠️", "#FEE2E2"
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            st.markdown(f"""
            <div style='text-align: center; padding: 40px; background: {bg_color}; 
                        border-radius: 15px; border: 4px solid {color}; margin: 20px 0;'>
                <h1 style='font-size: 72px; margin: 0; color: {color};'>{prediction:.1f}</h1>
                <p style='font-size: 20px; color: #374151; margin: 10px 0; font-weight: 600;'>out of 10</p>
                <h2 style='margin: 15px 0; color: {color}; font-size: 32px;'>{emoji} {status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge for prediction
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Your Wellbeing Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 3], 'color': "#FEE2E2"},
                    {'range': [3, 5], 'color': "#FEF3C7"},
                    {'range': [5, 7], 'color': "#D1FAE5"},
                    {'range': [7, 10], 'color': "#A7F3D0"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 7}
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True, key="prediction_gauge")
        
        st.markdown("---")
        
        # Personalized Recommendations
        st.markdown("### 💡 Personalized Action Plan")
        
        recommendations = []
        
        if input_sleep < 7:
            recommendations.append({
                'priority': 'High',
                'icon': '😴',
                'title': 'Improve Sleep Quality',
                'current': f'{input_sleep:.1f} hours',
                'target': '7-8 hours',
                'advice': 'Establish a consistent bedtime routine, avoid screens 1 hour before bed, and create a dark, cool sleeping environment.',
                'impact': f'+{(7 - input_sleep) * 0.3:.1f} potential wellbeing points'
            })
        
        if input_stress >= 7:
            recommendations.append({
                'priority': 'High',
                'icon': '🧘',
                'title': 'Stress Management',
                'current': f'{input_stress}/10 stress level',
                'target': 'Below 5/10',
                'advice': 'Practice daily meditation (10-15 min), try deep breathing exercises, consider talking to a counselor, and identify stress triggers.',
                'impact': f'+{(input_stress - 5) * 0.4:.1f} potential wellbeing points'
            })
        
        if input_screen > 6:
            recommendations.append({
                'priority': 'Medium',
                'icon': '📱',
                'title': 'Reduce Screen Time',
                'current': f'{input_screen:.1f} hours/day',
                'target': '4-6 hours/day',
                'advice': 'Use screen time tracking apps, take 20-20-20 breaks (every 20 min, look 20 feet away for 20 sec), and replace screen time with outdoor activities.',
                'impact': f'+{(input_screen - 4) * 0.2:.1f} potential wellbeing points'
            })
        
        if input_activity < df['Physical_Activity'].median():
            recommendations.append({
                'priority': 'High',
                'icon': '🏃',
                'title': 'Increase Physical Activity',
                'current': f'{input_activity} activity score',
                'target': f'{int(df["Physical_Activity"].median())}+ activity score',
                'advice': 'Start with 30 minutes of moderate exercise 3-4 times per week. Walking, cycling, or swimming are great options. Gradually increase intensity.',
                'impact': '+0.5-1.0 potential wellbeing points'
            })
        
        if input_friends == 'No':
            recommendations.append({
                'priority': 'High',
                'icon': '👥',
                'title': 'Build Social Connections',
                'current': 'No close friends',
                'target': 'Develop meaningful friendships',
                'advice': 'Join clubs or groups aligned with your interests, volunteer, attend community events, or reconnect with old friends. Quality matters more than quantity.',
                'impact': '+1.0-1.5 potential wellbeing points'
            })
        
        if input_work > 9:
            recommendations.append({
                'priority': 'Medium',
                'icon': '⚖️',
                'title': 'Work-Life Balance',
                'current': f'{input_work:.1f} hours/day',
                'target': '8 hours/day or less',
                'advice': 'Set clear work boundaries, learn to delegate, take regular breaks, and discuss workload with your manager if needed.',
                'impact': f'+{(input_work - 8) * 0.3:.1f} potential wellbeing points'
            })
        
        if input_diet not in ['Excellent', 'Very Good']:
            recommendations.append({
                'priority': 'Medium',
                'icon': '🍎',
                'title': 'Improve Diet Quality',
                'current': f'{input_diet} diet',
                'target': 'Excellent or Very Good',
                'advice': 'Increase fruits and vegetables, reduce processed foods, stay hydrated (8 glasses water/day), and consider meal planning.',
                'impact': '+0.5-0.8 potential wellbeing points'
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'Positive',
                'icon': '✅',
                'title': 'Excellent Lifestyle!',
                'current': 'All factors optimal',
                'target': 'Maintain current habits',
                'advice': 'You\'re doing great! Keep maintaining these healthy habits. Consider helping others by sharing your wellness strategies.',
                'impact': 'Continue thriving!'
            })
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Positive': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = {'High': '#EF4444', 'Medium': '#F59E0B', 'Positive': '#10B981'}.get(rec['priority'], '#6B7280')
            
            st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 10px; margin: 15px 0; 
                        border-left: 5px solid {priority_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h4 style='margin: 0 0 10px 0; color: #111827;'>
                    {rec['icon']} {i}. {rec['title']} 
                    <span style='background: {priority_color}; color: white; padding: 3px 10px; 
                                 border-radius: 5px; font-size: 12px; margin-left: 10px;'>{rec['priority']} Priority</span>
                </h4>
                <p style='margin: 5px 0; color: #6B7280;'><strong>Current:</strong> {rec['current']} → <strong>Target:</strong> {rec['target']}</p>
                <p style='margin: 10px 0 5px 0; color: #374151;'>{rec['advice']}</p>
                <p style='margin: 5px 0 0 0; color: {priority_color}; font-weight: 600;'><strong>Potential Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison with Population
        st.markdown("### 📊 How You Compare to Others")
        
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        
        with col_comp1:
            avg_wb = df['Overall_Wellbeing_Score'].mean()
            diff = prediction - avg_wb
            st.metric("vs Population Average", 
                     f"{prediction:.1f}", 
                     f"{diff:+.1f}",
                     delta_color="normal")
        
        with col_comp2:
            percentile = (df['Overall_Wellbeing_Score'] < prediction).sum() / len(df) * 100
            st.metric("Your Percentile", 
                     f"{percentile:.0f}%",
                     help="You're doing better than this % of people")
        
        with col_comp3:
            similar = len(df[(df['Overall_Wellbeing_Score'] >= prediction - 0.5) & 
                            (df['Overall_Wellbeing_Score'] <= prediction + 0.5)])
            st.metric("Similar People", 
                     f"{similar}",
                     help="People with similar wellbeing scores")
        
        with col_comp4:
            if prediction >= 7:
                rank = "Top 25%"
            elif prediction >= 5:
                rank = "Middle 50%"
            else:
                rank = "Bottom 25%"
            st.metric("Your Rank", rank)
        
        # Comparison Chart
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['Overall_Wellbeing_Score'],
            nbinsx=20,
            name='Population',
            marker_color='#4F46E5',
            opacity=0.6
        ))
        
        fig.add_vline(x=prediction, line_dash="dash", line_color="red", line_width=3,
                     annotation_text=f"Your Score: {prediction:.1f}", annotation_position="top")
        
        fig.update_layout(
            title='Your Score vs Population Distribution',
            xaxis_title='Wellbeing Score',
            yaxis_title='Number of People',
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="comparison_hist")
        
        st.markdown("""
        <div class="explanation-box">
        <strong>📖 What this shows:</strong> The blue bars show how wellbeing scores are distributed across the entire population. 
        The red line shows where YOUR predicted score falls. Being to the right of the peak means you're above average!
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    df = load_and_preprocess_data()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 30px 0 20px 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>🧠 Mental Health Analytics Platform</h1>
        <p style='font-size: 1.2rem; color: #6B7280;'>
            AI-Powered Insights for Mental Wellbeing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    filters = create_sidebar(df)
    df_filtered = apply_filters(df, filters)
    
    if len(df_filtered) == 0:
        st.warning("⚠️ No data matches your current filter selections. Please adjust the filters.")
        st.stop()
    
    # Sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Active Records", f"{len(df_filtered):,} / {len(df):,}")
    
    # Export
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export Data")
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        "Download Filtered Data (CSV)",
        csv,
        "mental_health_data.csv",
        "text/csv",
        use_container_width=True
    )
    
    # About
    with st.sidebar.expander("ℹ️ About This Platform"):
        st.markdown("""
        **Mental Health Analytics v4.0**
        
        This platform uses advanced data science and machine learning to:
        - Analyze mental health patterns
        - Identify key wellbeing factors
        - Predict wellbeing scores
        - Provide personalized recommendations
        
        **Built with:**
        - Streamlit
        - Plotly
        - Scikit-learn
        - Pandas
        
        **Note:** This is for educational purposes. 
        Consult healthcare professionals for medical advice.
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard",
        "🏃 Lifestyle Intelligence",
        "🔍 Segment Analysis",
        "📈 Correlation Intelligence",
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: #F3F4F6; border-radius: 10px;'>
        <p style='margin: 0; color: #6B7280; font-size: 14px;'>
            🧠 Mental Health Analytics Platform v4.0 | Built with ❤️ using Streamlit
        </p>
        <p style='margin: 5px 0 0 0; color: #9CA3AF; font-size: 12px;'>
            For educational and research purposes only | Consult healthcare professionals for medical advice
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
