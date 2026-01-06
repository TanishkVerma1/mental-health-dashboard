import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Mental Health & Lifestyle Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 10px;
        border-bottom: 3px solid #4CAF50;
    }
    h2 {
        color: #34495e;
        margin-top: 20px;
    }
    h3 {
        color: #7f8c8d;
    }
    .insight-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_data
def load_data():
    """Load the CSV data with error handling"""
    try:
        df = pd.read_csv('data/Mental_Health_and_Lifestyle_Research.csv')
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please ensure 'Mental_Health_and_Lifestyle_Research.csv' is in the 'data/' folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

@st.cache_data
def preprocess_data(df):
    """Preprocess the dataset: handle missing values, convert types"""
    df = df.copy()
    
    # Handle missing values with "Unknown" or appropriate defaults
    categorical_cols = ['Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'Unknown')
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                    'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle categorical columns
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip()
    
    if 'Social_Interaction_Freq' in df.columns:
        df['Social_Interaction_Freq'] = df['Social_Interaction_Freq'].astype(str).str.strip()
    
    if 'Diet_Quality' in df.columns:
        df['Diet_Quality'] = df['Diet_Quality'].astype(str).str.strip()
    
    if 'Has_Close_Friends' in df.columns:
        df['Has_Close_Friends'] = df['Has_Close_Friends'].astype(str).str.strip()
        df['Has_Close_Friends'] = df['Has_Close_Friends'].map({'True': True, 'False': False, 'true': True, 'false': False})
    
    return df

def apply_filters(df, filters):
    """Apply sidebar filters to the dataframe"""
    filtered_df = df.copy()
    
    # Age filter
    if 'age_range' in filters:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= filters['age_range'][0]) & 
            (filtered_df['Age'] <= filters['age_range'][1])
        ]
    
    # Gender filter
    if 'gender' in filters and filters['gender']:
        filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
    
    # Mental Health Status filter
    if 'mental_health' in filters and filters['mental_health']:
        filtered_df = filtered_df[filtered_df['Mental_Health_Status'].isin(filters['mental_health'])]
    
    # Diet Quality filter
    if 'diet_quality' in filters and filters['diet_quality']:
        filtered_df = filtered_df[filtered_df['Diet_Quality'].isin(filters['diet_quality'])]
    
    # Social Interaction filter
    if 'social_interaction' in filters and filters['social_interaction']:
        filtered_df = filtered_df[filtered_df['Social_Interaction_Freq'].isin(filters['social_interaction'])]
    
    # Has Close Friends filter
    if 'has_friends' in filters and filters['has_friends'] != 'All':
        friend_value = True if filters['has_friends'] == 'True' else False
        filtered_df = filtered_df[filtered_df['Has_Close_Friends'] == friend_value]
    
    # Substance Use filter
    if 'substance_use' in filters and filters['substance_use']:
        filtered_df = filtered_df[filtered_df['Substance_Use'].isin(filters['substance_use'])]
    
    # Physical Health Condition filter
    if 'physical_health' in filters and filters['physical_health']:
        filtered_df = filtered_df[filtered_df['Physical_Health_Condition'].isin(filters['physical_health'])]
    
    return filtered_df

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

def create_sidebar_filters(df):
    """Create sidebar filters and return filter dictionary"""
    st.sidebar.title("🎛️ Dashboard Filters")
    st.sidebar.markdown("---")
    
    # Initialize session state for reset functionality
    if 'reset_filters' not in st.session_state:
        st.session_state.reset_filters = False
    
    filters = {}
    
    # Age Range
    st.sidebar.subheader("📊 Demographics")
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    filters['age_range'] = st.sidebar.slider(
        "Age Range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        key='age_slider'
    )
    
    # Gender
    gender_options = sorted(df['Gender'].unique().tolist())
    filters['gender'] = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
        key='gender_select'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Mental Health")
    
    # Mental Health Status
    mental_health_options = sorted(df['Mental_Health_Status'].unique().tolist())
    filters['mental_health'] = st.sidebar.multiselect(
        "Mental Health Status",
        options=mental_health_options,
        default=mental_health_options,
        key='mental_health_select'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🍎 Lifestyle Factors")
    
    # Diet Quality
    diet_options = sorted(df['Diet_Quality'].unique().tolist())
    filters['diet_quality'] = st.sidebar.multiselect(
        "Diet Quality",
        options=diet_options,
        default=diet_options,
        key='diet_select'
    )
    
    # Social Interaction
    social_options = sorted(df['Social_Interaction_Freq'].unique().tolist())
    filters['social_interaction'] = st.sidebar.multiselect(
        "Social Interaction Frequency",
        options=social_options,
        default=social_options,
        key='social_select'
    )
    
    # Has Close Friends
    filters['has_friends'] = st.sidebar.selectbox(
        "Has Close Friends",
        options=['All', 'True', 'False'],
        key='friends_select'
    )
    
    # Substance Use
    substance_options = sorted(df['Substance_Use'].unique().tolist())
    filters['substance_use'] = st.sidebar.multiselect(
        "Substance Use",
        options=substance_options,
        default=substance_options,
        key='substance_select'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Physical Health")
    
    # Physical Health Condition
    physical_health_options = sorted(df['Physical_Health_Condition'].unique().tolist())
    filters['physical_health'] = st.sidebar.multiselect(
        "Physical Health Condition",
        options=physical_health_options,
        default=physical_health_options,
        key='physical_health_select'
    )
    
    st.sidebar.markdown("---")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.reset_filters = True
        st.rerun()
    
    # Info
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use filters to explore specific segments of the population.")
    
    return filters

# ============================================================================
# KPI SECTION
# ============================================================================

def display_kpis(df):
    """Display key performance indicators"""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        avg_wellbeing = df['Overall_Wellbeing_Score'].mean()
        st.metric(
            label="🎯 Avg Wellbeing",
            value=f"{avg_wellbeing:.1f}",
            delta=f"{avg_wellbeing - 5.5:.1f} from mid"
        )
    
    with col2:
        avg_stress = df['Stress_Level'].mean()
        st.metric(
            label="😰 Avg Stress",
            value=f"{avg_stress:.1f}",
            delta=f"{5.5 - avg_stress:.1f} from mid",
            delta_color="inverse"
        )
    
    with col3:
        avg_sleep = df['Hours_of_Sleep'].mean()
        st.metric(
            label="😴 Avg Sleep (hrs)",
            value=f"{avg_sleep:.1f}",
            delta=f"{avg_sleep - 7:.1f} from 7h"
        )
    
    with col4:
        avg_screen = df['Screen_Time_per_Day'].mean()
        st.metric(
            label="📱 Avg Screen Time (hrs)",
            value=f"{avg_screen:.1f}",
            delta=f"{4 - avg_screen:.1f} from 4h",
            delta_color="inverse"
        )
    
    with col5:
        avg_work = df['Work_Hours_per_Day'].mean()
        st.metric(
            label="💼 Avg Work Hours",
            value=f"{avg_work:.1f}",
            delta=f"{avg_work - 8:.1f} from 8h"
        )
    
    with col6:
        pct_friends = (df['Has_Close_Friends'].sum() / len(df)) * 100
        st.metric(
            label="👥 % With Close Friends",
            value=f"{pct_friends:.1f}%"
        )

# ============================================================================
# PAGE A: OVERVIEW
# ============================================================================

def overview_page(df):
    """Display overview page with KPIs and distributions"""
    st.header("📊 Overview Dashboard")
    st.markdown("**High-level metrics and distributions of mental health and lifestyle factors**")
    st.markdown("---")
    
    # KPIs
    display_kpis(df)
    
    st.markdown("---")
    
    # Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Wellbeing Score Distribution")
        fig_wellbeing = px.histogram(
            df,
            x='Overall_Wellbeing_Score',
            nbins=20,
            title='Distribution of Overall Wellbeing Scores',
            labels={'Overall_Wellbeing_Score': 'Wellbeing Score', 'count': 'Number of People'},
            color_discrete_sequence=['#4CAF50']
        )
        fig_wellbeing.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_wellbeing, use_container_width=True)
        st.markdown("**📖 How to read:** This histogram shows how wellbeing scores are distributed. Higher scores (right side) indicate better wellbeing.")
    
    with col2:
        st.subheader("😰 Stress Level Distribution")
        fig_stress = px.histogram(
            df,
            x='Stress_Level',
            nbins=10,
            title='Distribution of Stress Levels',
            labels={'Stress_Level': 'Stress Level', 'count': 'Number of People'},
            color_discrete_sequence=['#FF5722']
        )
        fig_stress.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_stress, use_container_width=True)
        st.markdown("**📖 How to read:** This shows stress level distribution. Higher values (right side) mean higher stress.")
    
    st.markdown("---")
    
    # Mental Health Status
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🧠 Mental Health Status Breakdown")
        mental_health_counts = df['Mental_Health_Status'].value_counts().reset_index()
        mental_health_counts.columns = ['Status', 'Count']
        
        fig_mental = px.bar(
            mental_health_counts,
            x='Status',
            y='Count',
            title='Mental Health Status Distribution',
            labels={'Status': 'Mental Health Status', 'Count': 'Number of People'},
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig_mental.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mental, use_container_width=True)
        st.markdown("**📖 How to read:** Each bar represents the number of people with each mental health status.")
    
    with col4:
        st.subheader("😴 Sleep Hours Distribution")
        fig_sleep = px.histogram(
            df,
            x='Hours_of_Sleep',
            nbins=15,
            title='Distribution of Sleep Hours',
            labels={'Hours_of_Sleep': 'Hours of Sleep', 'count': 'Number of People'},
            color_discrete_sequence=['#2196F3']
        )
        fig_sleep.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_sleep, use_container_width=True)
        st.markdown("**📖 How to read:** Shows how many hours people typically sleep. 7-8 hours is generally recommended.")
    
    st.markdown("---")
    
    # Key Takeaways
    st.subheader("🔍 Key Takeaways")
    
    avg_wellbeing = df['Overall_Wellbeing_Score'].mean()
    avg_stress = df['Stress_Level'].mean()
    avg_sleep = df['Hours_of_Sleep'].mean()
    pct_friends = (df['Has_Close_Friends'].sum() / len(df)) * 100
    most_common_mental = df['Mental_Health_Status'].mode()[0]
    
    takeaways = f"""
    <div class="insight-box">
    <h4>📌 Current Dataset Insights ({len(df)} people):</h4>
    <ul>
        <li><strong>Average Wellbeing:</strong> {avg_wellbeing:.1f}/10 - {'Good' if avg_wellbeing >= 6 else 'Needs attention'}</li>
        <li><strong>Average Stress:</strong> {avg_stress:.1f}/10 - {'High stress levels' if avg_stress >= 6 else 'Moderate stress'}</li>
        <li><strong>Sleep Pattern:</strong> {avg_sleep:.1f} hours average - {'Below recommended' if avg_sleep < 7 else 'Adequate'}</li>
        <li><strong>Social Connection:</strong> {pct_friends:.1f}% have close friends - {'Strong social network' if pct_friends >= 70 else 'Limited social connections'}</li>
        <li><strong>Most Common Mental Health Status:</strong> {most_common_mental}</li>
    </ul>
    </div>
    """
    st.markdown(takeaways, unsafe_allow_html=True)

# ============================================================================
# PAGE B: LIFESTYLE DRIVERS
# ============================================================================

def lifestyle_drivers_page(df):
    """Display lifestyle factors and their relationships"""
    st.header("🏃 Lifestyle Drivers Analysis")
    st.markdown("**Explore how lifestyle factors relate to mental health and wellbeing**")
    st.markdown("---")
    
    # Screen Time vs Wellbeing
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Screen Time vs Wellbeing")
        correlation = df[['Screen_Time_per_Day', 'Overall_Wellbeing_Score']].corr().iloc[0, 1]
        
        fig_screen = px.scatter(
            df,
            x='Screen_Time_per_Day',
            y='Overall_Wellbeing_Score',
            title=f'Screen Time vs Wellbeing (Correlation: {correlation:.3f})',
            labels={'Screen_Time_per_Day': 'Screen Time (hours/day)', 'Overall_Wellbeing_Score': 'Wellbeing Score'},
            color='Stress_Level',
            color_continuous_scale='RdYlGn_r',
            opacity=0.6
        )
        fig_screen.update_layout(height=400)
        st.plotly_chart(fig_screen, use_container_width=True)
        
        interpretation = "negative" if correlation < -0.1 else "positive" if correlation > 0.1 else "weak"
        st.markdown(f"**📖 Interpretation:** There is a **{interpretation}** relationship between screen time and wellbeing. Each dot represents a person, colored by stress level.")
    
    with col2:
        st.subheader("😴 Sleep vs Stress")
        correlation_sleep = df[['Hours_of_Sleep', 'Stress_Level']].corr().iloc[0, 1]
        
        fig_sleep_stress = px.scatter(
            df,
            x='Hours_of_Sleep',
            y='Stress_Level',
            title=f'Sleep Hours vs Stress Level (Correlation: {correlation_sleep:.3f})',
            labels={'Hours_of_Sleep': 'Hours of Sleep', 'Stress_Level': 'Stress Level'},
            color='Overall_Wellbeing_Score',
            color_continuous_scale='Viridis',
            opacity=0.6
        )
        fig_sleep_stress.update_layout(height=400)
        st.plotly_chart(fig_sleep_stress, use_container_width=True)
        
        interpretation_sleep = "negative" if correlation_sleep < -0.1 else "positive" if correlation_sleep > 0.1 else "weak"
        st.markdown(f"**📖 Interpretation:** There is a **{interpretation_sleep}** relationship. Better sleep typically correlates with lower stress.")
    
    st.markdown("---")
    
    # Work Hours vs Stress
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💼 Work Hours vs Stress")
        correlation_work = df[['Work_Hours_per_Day', 'Stress_Level']].corr().iloc[0, 1]
        
        fig_work = px.scatter(
            df,
            x='Work_Hours_per_Day',
            y='Stress_Level',
            title=f'Work Hours vs Stress (Correlation: {correlation_work:.3f})',
            labels={'Work_Hours_per_Day': 'Work Hours per Day', 'Stress_Level': 'Stress Level'},
            color='Overall_Wellbeing_Score',
            color_continuous_scale='RdYlGn',
            opacity=0.6
        )
        fig_work.update_layout(height=400)
        st.plotly_chart(fig_work, use_container_width=True)
        st.markdown(f"**📖 Interpretation:** Correlation of {correlation_work:.3f} suggests {'longer work hours may increase stress' if correlation_work > 0.1 else 'work hours have minimal impact on stress'}.")
    
    with col4:
        st.subheader("🏃 Physical Activity vs Wellbeing")
        correlation_activity = df[['Physical_Activity', 'Overall_Wellbeing_Score']].corr().iloc[0, 1]
        
        fig_activity = px.scatter(
            df,
            x='Physical_Activity',
            y='Overall_Wellbeing_Score',
            title=f'Physical Activity vs Wellbeing (Correlation: {correlation_activity:.3f})',
            labels={'Physical_Activity': 'Physical Activity Score', 'Overall_Wellbeing_Score': 'Wellbeing Score'},
            color='Stress_Level',
            color_continuous_scale='RdYlGn_r',
            opacity=0.6
        )
        fig_activity.update_layout(height=400)
        st.plotly_chart(fig_activity, use_container_width=True)
        st.markdown(f"**📖 Interpretation:** {'Higher physical activity is associated with better wellbeing' if correlation_activity > 0.1 else 'Physical activity shows limited correlation with wellbeing'}.")
    
    st.markdown("---")
    
    # Diet Quality Analysis
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("🍎 Wellbeing by Diet Quality")
        
        diet_wellbeing = df.groupby('Diet_Quality')['Overall_Wellbeing_Score'].agg(['mean', 'std', 'count']).reset_index()
        diet_wellbeing = diet_wellbeing.sort_values('mean', ascending=False)
        
        fig_diet = px.bar(
            diet_wellbeing,
            x='Diet_Quality',
            y='mean',
            title='Average Wellbeing by Diet Quality',
            labels={'Diet_Quality': 'Diet Quality', 'mean': 'Average Wellbeing Score'},
            color='mean',
            color_continuous_scale='Greens',
            text='mean'
        )
        fig_diet.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_diet.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_diet, use_container_width=True)
        st.markdown("**📖 How to read:** Higher bars indicate better average wellbeing for that diet quality category.")
    
    with col6:
        st.subheader("👥 Stress by Social Interaction")
        
        social_stress = df.groupby('Social_Interaction_Freq')['Stress_Level'].agg(['mean', 'std', 'count']).reset_index()
        social_order = {'Low': 0, 'Moderate': 1, 'High': 2}
        social_stress['order'] = social_stress['Social_Interaction_Freq'].map(social_order)
        social_stress = social_stress.sort_values('order')
        
        fig_social = px.bar(
            social_stress,
            x='Social_Interaction_Freq',
            y='mean',
            title='Average Stress by Social Interaction Frequency',
            labels={'Social_Interaction_Freq': 'Social Interaction Frequency', 'mean': 'Average Stress Level'},
            color='mean',
            color_continuous_scale='Reds_r',
            text='mean'
        )
        fig_social.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_social.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_social, use_container_width=True)
        st.markdown("**📖 How to read:** Lower bars are better (less stress). Social interaction frequency may impact stress levels.")
    
    st.markdown("---")
    
    # Box plots for deeper analysis
    st.subheader("📦 Distribution Analysis")
    
    col7, col8 = st.columns(2)
    
    with col7:
        fig_box_diet = px.box(
            df,
            x='Diet_Quality',
            y='Overall_Wellbeing_Score',
            title='Wellbeing Distribution by Diet Quality',
            labels={'Diet_Quality': 'Diet Quality', 'Overall_Wellbeing_Score': 'Wellbeing Score'},
            color='Diet_Quality'
        )
        fig_box_diet.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_box_diet, use_container_width=True)
        st.markdown("**📖 How to read:** Box plots show the spread of data. The box represents the middle 50% of values, with the line showing the median.")
    
    with col8:
        fig_box_social = px.box(
            df,
            x='Social_Interaction_Freq',
            y='Stress_Level',
            title='Stress Distribution by Social Interaction',
            labels={'Social_Interaction_Freq': 'Social Interaction', 'Stress_Level': 'Stress Level'},
            color='Social_Interaction_Freq'
        )
        fig_box_social.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_box_social, use_container_width=True)
        st.markdown("**📖 How to read:** Compare the median lines and box positions to see how stress varies across social interaction levels.")

# ============================================================================
# PAGE C: SEGMENT COMPARISON
# ============================================================================

def segment_comparison_page(df):
    """Compare different population segments"""
    st.header("🔍 Segment Comparison")
    st.markdown("**Compare wellbeing and stress across different population groups**")
    st.markdown("---")
    
    # Gender Comparison
    st.subheader("⚥ Gender Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_wellbeing = df.groupby('Gender')['Overall_Wellbeing_Score'].mean().reset_index()
        
        fig_gender_wb = px.bar(
            gender_wellbeing,
            x='Gender',
            y='Overall_Wellbeing_Score',
            title='Average Wellbeing by Gender',
            labels={'Gender': 'Gender', 'Overall_Wellbeing_Score': 'Avg Wellbeing Score'},
            color='Overall_Wellbeing_Score',
            color_continuous_scale='Viridis',
            text='Overall_Wellbeing_Score'
        )
        fig_gender_wb.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_gender_wb.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_gender_wb, use_container_width=True)
    
    with col2:
        gender_stress = df.groupby('Gender')['Stress_Level'].mean().reset_index()
        
        fig_gender_stress = px.bar(
            gender_stress,
            x='Gender',
            y='Stress_Level',
            title='Average Stress by Gender',
            labels={'Gender': 'Gender', 'Stress_Level': 'Avg Stress Level'},
            color='Stress_Level',
            color_continuous_scale='Reds',
            text='Stress_Level'
        )
        fig_gender_stress.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_gender_stress.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_gender_stress, use_container_width=True)
    
    st.markdown("---")
    
    # Close Friends Comparison
    st.subheader("👥 Impact of Close Friends")
    
    col3, col4 = st.columns(2)
    
    with col3:
        friends_wellbeing = df.groupby('Has_Close_Friends')['Overall_Wellbeing_Score'].mean().reset_index()
        friends_wellbeing['Has_Close_Friends'] = friends_wellbeing['Has_Close_Friends'].map({True: 'Has Friends', False: 'No Close Friends'})
        
        fig_friends_wb = px.bar(
            friends_wellbeing,
            x='Has_Close_Friends',
            y='Overall_Wellbeing_Score',
            title='Wellbeing: With vs Without Close Friends',
            labels={'Has_Close_Friends': 'Friendship Status', 'Overall_Wellbeing_Score': 'Avg Wellbeing Score'},
            color='Overall_Wellbeing_Score',
            color_continuous_scale='Greens',
            text='Overall_Wellbeing_Score'
        )
        fig_friends_wb.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_friends_wb.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_friends_wb, use_container_width=True)
    
    with col4:
        friends_stress = df.groupby('Has_Close_Friends')['Stress_Level'].mean().reset_index()
        friends_stress['Has_Close_Friends'] = friends_stress['Has_Close_Friends'].map({True: 'Has Friends', False: 'No Close Friends'})
        
        fig_friends_stress = px.bar(
            friends_stress,
            x='Has_Close_Friends',
            y='Stress_Level',
            title='Stress: With vs Without Close Friends',
            labels={'Has_Close_Friends': 'Friendship Status', 'Stress_Level': 'Avg Stress Level'},
            color='Stress_Level',
            color_continuous_scale='Reds',
            text='Stress_Level'
        )
        fig_friends_stress.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_friends_stress.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_friends_stress, use_container_width=True)
    
    st.markdown("---")
    
    # Substance Use Comparison
    st.subheader("🚬 Substance Use Impact")
    
    col5, col6 = st.columns(2)
    
    with col5:
        substance_wellbeing = df.groupby('Substance_Use')['Overall_Wellbeing_Score'].mean().reset_index()
        substance_wellbeing = substance_wellbeing.sort_values('Overall_Wellbeing_Score', ascending=False)
        
        fig_substance_wb = px.bar(
            substance_wellbeing,
            x='Substance_Use',
            y='Overall_Wellbeing_Score',
            title='Average Wellbeing by Substance Use',
            labels={'Substance_Use': 'Substance Use', 'Overall_Wellbeing_Score': 'Avg Wellbeing Score'},
            color='Overall_Wellbeing_Score',
            color_continuous_scale='Blues',
            text='Overall_Wellbeing_Score'
        )
        fig_substance_wb.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_substance_wb.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_substance_wb, use_container_width=True)
    
    with col6:
        substance_stress = df.groupby('Substance_Use')['Stress_Level'].mean().reset_index()
        substance_stress = substance_stress.sort_values('Stress_Level', ascending=True)
        
        fig_substance_stress = px.bar(
            substance_stress,
            x='Substance_Use',
            y='Stress_Level',
            title='Average Stress by Substance Use',
            labels={'Substance_Use': 'Substance Use', 'Stress_Level': 'Avg Stress Level'},
            color='Stress_Level',
            color_continuous_scale='Oranges',
            text='Stress_Level'
        )
        fig_substance_stress.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_substance_stress.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_substance_stress, use_container_width=True)
    
    st.markdown("---")
    
    # Box plot comparisons
    st.subheader("📊 Detailed Distribution Comparisons")
    
    col7, col8 = st.columns(2)
    
    with col7:
        fig_box_gender = px.box(
            df,
            x='Gender',
            y='Overall_Wellbeing_Score',
            title='Wellbeing Distribution by Gender',
            labels={'Gender': 'Gender', 'Overall_Wellbeing_Score': 'Wellbeing Score'},
            color='Gender'
        )
        fig_box_gender.update_layout(height=400)
        st.plotly_chart(fig_box_gender, use_container_width=True)
    
    with col8:
        df_friends_labeled = df.copy()
        df_friends_labeled['Friends_Label'] = df_friends_labeled['Has_Close_Friends'].map({True: 'Has Friends', False: 'No Friends'})
        
        fig_box_friends = px.box(
            df_friends_labeled,
            x='Friends_Label',
            y='Stress_Level',
            title='Stress Distribution by Friendship Status',
            labels={'Friends_Label': 'Friendship Status', 'Stress_Level': 'Stress Level'},
            color='Friends_Label'
        )
        fig_box_friends.update_layout(height=400)
        st.plotly_chart(fig_box_friends, use_container_width=True)
    
    st.markdown("---")
    
    # Top Segments Analysis
    st.subheader("🏆 Top Performing Segments")
    
    col9, col10 = st.columns(2)
    
    with col9:
        st.markdown("**🌟 Top 3 Highest Wellbeing Segments**")
        
        # Create segments
        segments_wb = []
        
        # By Gender
        for gender in df['Gender'].unique():
            avg_wb = df[df['Gender'] == gender]['Overall_Wellbeing_Score'].mean()
            count = len(df[df['Gender'] == gender])
            segments_wb.append({'Segment': f'Gender: {gender}', 'Avg Wellbeing': avg_wb, 'Count': count})
        
        # By Friends
        for friends in [True, False]:
            avg_wb = df[df['Has_Close_Friends'] == friends]['Overall_Wellbeing_Score'].mean()
            count = len(df[df['Has_Close_Friends'] == friends])
            label = 'Has Friends' if friends else 'No Friends'
            segments_wb.append({'Segment': label, 'Avg Wellbeing': avg_wb, 'Count': count})
        
        # By Social Interaction
        for social in df['Social_Interaction_Freq'].unique():
            avg_wb = df[df['Social_Interaction_Freq'] == social]['Overall_Wellbeing_Score'].mean()
            count = len(df[df['Social_Interaction_Freq'] == social])
            segments_wb.append({'Segment': f'Social: {social}', 'Avg Wellbeing': avg_wb, 'Count': count})
        
        segments_wb_df = pd.DataFrame(segments_wb).sort_values('Avg Wellbeing', ascending=False).head(3)
        segments_wb_df['Avg Wellbeing'] = segments_wb_df['Avg Wellbeing'].round(2)
        
        st.dataframe(segments_wb_df, use_container_width=True, hide_index=True)
    
    with col10:
        st.markdown("**⚠️ Top 3 Highest Stress Segments**")
        
        # Create segments
        segments_stress = []
        
        # By Gender
        for gender in df['Gender'].unique():
            avg_stress = df[df['Gender'] == gender]['Stress_Level'].mean()
            count = len(df[df['Gender'] == gender])
            segments_stress.append({'Segment': f'Gender: {gender}', 'Avg Stress': avg_stress, 'Count': count})
        
        # By Friends
        for friends in [True, False]:
            avg_stress = df[df['Has_Close_Friends'] == friends]['Stress_Level'].mean()
            count = len(df[df['Has_Close_Friends'] == friends])
            label = 'Has Friends' if friends else 'No Friends'
            segments_stress.append({'Segment': label, 'Avg Stress': avg_stress, 'Count': count})
        
        # By Social Interaction
        for social in df['Social_Interaction_Freq'].unique():
            avg_stress = df[df['Social_Interaction_Freq'] == social]['Stress_Level'].mean()
            count = len(df[df['Social_Interaction_Freq'] == social])
            segments_stress.append({'Segment': f'Social: {social}', 'Avg Stress': avg_stress, 'Count': count})
        
        segments_stress_df = pd.DataFrame(segments_stress).sort_values('Avg Stress', ascending=False).head(3)
        segments_stress_df['Avg Stress'] = segments_stress_df['Avg Stress'].round(2)
        
        st.dataframe(segments_stress_df, use_container_width=True, hide_index=True)
    
    st.markdown("**📖 How to read:** These tables show which population segments have the best wellbeing and highest stress levels.")

# ============================================================================
# PAGE D: CORRELATIONS & DATA QUALITY
# ============================================================================

def correlations_data_quality_page(df):
    """Display correlation analysis and data quality metrics"""
    st.header("📈 Correlations & Data Quality")
    st.markdown("**Understand relationships between variables and assess data completeness**")
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    
    numeric_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                    'Work_Hours_per_Day', 'Overall_Wellbeing_Score', 'Screen_Time_per_Day']
    
    corr_df = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect='auto',
        title='Correlation Matrix of Numeric Variables',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("""
    **📖 How to read this heatmap:**
    - Values range from -1 to +1
    - **+1** = Perfect positive correlation (when one increases, the other increases)
    - **-1** = Perfect negative correlation (when one increases, the other decreases)
    - **0** = No correlation
    - **Red colors** = Negative correlation
    - **Blue colors** = Positive correlation
    """)
    
    st.markdown("---")
    
    # Key Correlations
    st.subheader("🔑 Key Correlation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strongest Positive Correlations:**")
        
        # Get upper triangle of correlation matrix
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
        
        st.dataframe(top_positive, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Strongest Negative Correlations:**")
        
        top_negative = corr_pairs_df.nsmallest(5, 'Correlation')
        top_negative['Correlation'] = top_negative['Correlation'].round(3)
        
        st.dataframe(top_negative, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Data Quality Section
    st.subheader("🔍 Data Quality Assessment")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("**📊 Missing Values Summary**")
        
        # Calculate missing values
        missing_data = []
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            # Also count "Unknown" as missing for categorical columns
            if col in ['Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']:
                unknown_count = (df[col] == 'Unknown').sum()
                total_missing = missing_count + unknown_count
                total_missing_pct = (total_missing / len(df)) * 100
                missing_data.append({
                    'Column': col,
                    'Missing/Unknown': total_missing,
                    'Percentage': f"{total_missing_pct:.1f}%"
                })
            else:
                missing_data.append({
                    'Column': col,
                    'Missing/Unknown': missing_count,
                    'Percentage': f"{missing_pct:.1f}%"
                })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df[missing_df['Missing/Unknown'] > 0].sort_values('Missing/Unknown', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values detected!")
    
    with col4:
        st.markdown("**📈 Missing Values Visualization**")
        
        if len(missing_df) > 0:
            fig_missing = px.bar(
                missing_df,
                x='Column',
                y='Missing/Unknown',
                title='Missing/Unknown Values by Column',
                labels={'Column': 'Column Name', 'Missing/Unknown': 'Count'},
                color='Missing/Unknown',
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.info("No missing values to visualize.")
    
    st.markdown("---")
    
    # Dataset Statistics
    st.subheader("📋 Dataset Statistics")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Total Columns", len(df.columns))
    
    with col6:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'bool']).columns))
    
    with col7:
        completeness = ((df.size - df.isna().sum().sum()) / df.size) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    st.markdown("---")
    
    # Descriptive Statistics
    st.subheader("📊 Descriptive Statistics")
    
    desc_stats = df[numeric_cols].describe().T
    desc_stats = desc_stats.round(2)
    desc_stats = desc_stats.reset_index()
    desc_stats.columns = ['Variable', 'Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max']
    
    st.dataframe(desc_stats, use_container_width=True, hide_index=True)
    
    st.markdown("**📖 How to read:** This table shows statistical summaries for all numeric variables. Use it to understand the range and distribution of each variable.")

# ============================================================================
# PAGE E: ML PREDICTION MODEL
# ============================================================================

def ml_prediction_page(df):
    """Machine Learning model for predicting wellbeing"""
    st.header("🤖 AI Wellbeing Predictor")
    st.markdown("**Use machine learning to predict Overall Wellbeing Score based on lifestyle factors**")
    st.markdown("---")
    
    # Model explanation
    with st.expander("ℹ️ About This Model", expanded=False):
        st.markdown("""
        This prediction model uses **Random Forest Regression** to estimate a person's Overall Wellbeing Score 
        based on their lifestyle and health factors.
        
        **How it works:**
        1. The model is trained on the current dataset
        2. It learns patterns between lifestyle factors and wellbeing
        3. You can input your own values to get a personalized wellbeing prediction
        
        **Input Features:**
        - Age, Sleep Hours, Stress Level, Physical Activity
        - Work Hours, Screen Time, Diet Quality, Social Interaction
        - Gender, Close Friends, Substance Use, Physical Health Condition
        
        **Note:** This is a demonstration model for educational purposes. Real health assessments should be done by professionals.
        """)
    
    st.markdown("---")
    
    # Prepare data for ML
    df_ml = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['Gender', 'Diet_Quality', 'Social_Interaction_Freq', 
                           'Mental_Health_Status', 'Substance_Use', 'Physical_Health_Condition']
    
    for col in categorical_features:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
    
    # Convert boolean to int
    if 'Has_Close_Friends' in df_ml.columns:
        df_ml['Has_Close_Friends_encoded'] = df_ml['Has_Close_Friends'].astype(int)
    
    # Select features for model
    feature_cols = ['Age', 'Hours_of_Sleep', 'Stress_Level', 'Physical_Activity', 
                   'Work_Hours_per_Day', 'Screen_Time_per_Day',
                   'Gender_encoded', 'Diet_Quality_encoded', 'Social_Interaction_Freq_encoded',
                   'Has_Close_Friends_encoded', 'Substance_Use_encoded', 'Physical_Health_Condition_encoded']
    
    # Remove any columns that don't exist
    feature_cols = [col for col in feature_cols if col in df_ml.columns]
    
    X = df_ml[feature_cols]
    y = df_ml['Overall_Wellbeing_Score']
    
    # Remove any rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with st.spinner("🔄 Training AI model..."):
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    # Display model performance
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
        st.caption("How well model fits data (0-1, higher is better)")
    
    with col2:
        st.metric("RMSE", f"{rmse:.3f}")
        st.caption("Average prediction error")
    
    with col3:
        st.metric("MAE", f"{mae:.3f}")
        st.caption("Mean absolute error")
    
    with col4:
        accuracy_pct = max(0, (1 - (rmse / y.std())) * 100)
        st.metric("Accuracy", f"{accuracy_pct:.1f}%")
        st.caption("Relative accuracy estimate")
    
    st.markdown("---")
    
    # Feature importance
    col5, col6 = st.columns([1, 1])
    
    with col5:
        st.subheader("🎯 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': [col.replace('_encoded', '') for col in feature_cols],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Which Factors Matter Most?',
            labels={'Importance': 'Importance Score', 'Feature': 'Factor'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("**📖 Interpretation:** Higher bars indicate factors that have more influence on wellbeing predictions.")
    
    with col6:
        st.subheader("📈 Prediction Accuracy")
        
        # Scatter plot of actual vs predicted
        comparison_df = pd.DataFrame({
            'Actual Wellbeing': y_test,
            'Predicted Wellbeing': y_pred
        })
        
        fig_scatter = px.scatter(
            comparison_df,
            x='Actual Wellbeing',
            y='Predicted Wellbeing',
            title='Actual vs Predicted Wellbeing Scores',
            labels={'Actual Wellbeing': 'Actual Score', 'Predicted Wellbeing': 'Predicted Score'},
            opacity=0.6
        )
        
        # Add perfect prediction line
        fig_scatter.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("**📖 Interpretation:** Points closer to the red line indicate more accurate predictions.")
    
    st.markdown("---")
    
    # Interactive Prediction Tool
    st.subheader("🎮 Try It Yourself: Predict Your Wellbeing")
    st.markdown("**Enter your information below to get a personalized wellbeing prediction:**")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown("**👤 Personal Info**")
        input_age = st.slider("Age", 20, 35, 27)
        input_gender = st.selectbox("Gender", options=df['Gender'].unique())
        input_friends = st.selectbox("Have Close Friends?", options=['Yes', 'No'])
    
    with col8:
        st.markdown("**😴 Health & Lifestyle**")
        input_sleep = st.slider("Hours of Sleep", 5.0, 9.0, 7.0, 0.5)
        input_stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        input_activity = st.slider("Physical Activity Score", 
                                   int(df['Physical_Activity'].min()), 
                                   int(df['Physical_Activity'].max()), 
                                   int(df['Physical_Activity'].mean()))
    
    with col9:
        st.markdown("**💼 Work & Screen**")
        input_work = st.slider("Work Hours per Day", 5.0, 11.0, 8.0, 0.5)
        input_screen = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, 0.5)
        input_diet = st.selectbox("Diet Quality", options=sorted(df['Diet_Quality'].unique()))
    
    col10, col11 = st.columns(2)
    
    with col10:
        input_social = st.selectbox("Social Interaction Frequency", 
                                    options=sorted(df['Social_Interaction_Freq'].unique()))
    
    with col11:
        input_substance = st.selectbox("Substance Use", 
                                       options=sorted(df['Substance_Use'].unique()))
    
    input_physical_health = st.selectbox("Physical Health Condition", 
                                         options=sorted(df['Physical_Health_Condition'].unique()))
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict My Wellbeing Score", type="primary", use_container_width=True):
        # Prepare input data
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
            'Physical_Health_Condition_encoded': label_encoders['Physical_Health_Condition'].transform([input_physical_health])[0]
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display result with styling
        st.markdown("---")
        st.markdown("### 🎯 Your Predicted Wellbeing Score")
        
        # Create a gauge-like display
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            # Determine color based on score
            if prediction >= 7:
                color = "🟢"
                status = "Excellent"
                advice_color = "#4CAF50"
            elif prediction >= 5:
                color = "🟡"
                status = "Good"
                advice_color = "#FFC107"
            else:
                color = "🔴"
                status = "Needs Attention"
                advice_color = "#F44336"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background-color: #f0f2f6; border-radius: 15px; border: 3px solid {advice_color}'>
                <h1 style='font-size: 72px; margin: 0; color: {advice_color}'>{prediction:.1f}</h1>
                <h3 style='margin: 10px 0; color: #666'>out of 10</h3>
                <h2 style='margin: 10px 0; color: {advice_color}'>{color} {status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Personalized recommendations
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
# EXPORT FUNCTIONALITY
# ============================================================================

def export_filtered_data(df):
    """Create download button for filtered data"""
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_mental_health_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load and preprocess data
    df_raw = load_data()
    df = preprocess_data(df_raw)
    
    # Header
    st.title("🧠 Mental Health & Lifestyle Research Dashboard")
    st.markdown("**Comprehensive analysis of mental health factors and lifestyle patterns**")
    st.markdown("---")
    
    # Sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    df_filtered = apply_filters(df, filters)
    
    # Check if filtered data is empty
    if len(df_filtered) == 0:
        st.warning("⚠️ No data matches the current filters. Please adjust your filter selections.")
        st.stop()
    
    # Display filtered count
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Filtered Records", f"{len(df_filtered):,} / {len(df):,}")
    
    # Export button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export Data")
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # About section in sidebar
    with st.sidebar.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Mental Health & Lifestyle Dashboard**
        
        This dashboard analyzes the relationship between lifestyle factors and mental health outcomes.
        
        **Data Source:** Mental Health and Lifestyle Research Dataset
        
        **Features:**
        - Interactive filtering
        - Real-time analytics
        - ML predictions
        - Data export
        
        **Version:** 1.0.0
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏃 Lifestyle Drivers",
        "🔍 Segment Comparison",
        "📈 Correlations & Data Quality",
        "🤖 AI Wellbeing Predictor"
    ])
    
    with tab1:
        overview_page(df_filtered)
    
    with tab2:
        lifestyle_drivers_page(df_filtered)
    
    with tab3:
        segment_comparison_page(df_filtered)
    
    with tab4:
        correlations_data_quality_page(df_filtered)
    
    with tab5:
        ml_prediction_page(df_filtered)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Mental Health & Lifestyle Research Dashboard | Built with Streamlit</p>
        <p>💡 For educational and research purposes only. Consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()