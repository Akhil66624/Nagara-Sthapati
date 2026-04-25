import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from pyspark.sql import SparkSession

# Page configuration
st.set_page_config(
    page_title="Nagara-Sthapati: Smart City Infrastructure Planner",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Spark session
@st.cache_resource
def get_spark():
    return SparkSession.builder.getOrCreate()

spark = get_spark()

# Databricks API configuration
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Gemma model endpoint
GEMMA_ENDPOINT = "databricks-gemma-3-12b"

def query_gemma(prompt, max_tokens=2000, temperature=0.7):
    """Query the Gemma 3 12B model via Foundation Model API"""
    url = f"{DATABRICKS_HOST}/serving-endpoints/{GEMMA_ENDPOINT}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert urban planner and data analyst specializing in smart city infrastructure. Analyze data and provide insights about city infrastructure, traffic, energy, waste management, and environmental metrics."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying model: {str(e)}"

@st.cache_data(ttl=300)
def load_hourly_data():
    """Load hourly aggregated sensor data"""
    df = spark.table("workspace.smart_city.gold_sensor_metrics_by_city_hour").toPandas()
    df['event_hour'] = pd.to_datetime(df['event_hour'])
    return df

@st.cache_data(ttl=300)
def load_daily_data():
    """Load daily aggregated sensor data"""
    df = spark.table("workspace.smart_city.gold_sensor_metrics_by_city_day").toPandas()
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

def generate_infographic_insights(data, time_period="hourly"):
    """Generate AI insights for infographics"""
    
    # Prepare data summary for the model
    latest_data = data.sort_values('event_hour' if time_period == "hourly" else 'event_date', ascending=False).head(24)
    
    data_summary = f"""
Smart City Infrastructure Data Summary ({time_period}):

Cities: {', '.join(data['city_id'].unique())}
Time period: Last 24 {time_period} records

Traffic Metrics:
- Average vehicles: {latest_data['avg_vehicles'].mean():.0f}
- Peak vehicle count: {latest_data['max_vehicles'].max():.0f}

Energy Metrics:
- Total energy consumption: {latest_data['total_energy_kwh'].sum():.2f} kWh
- Average energy per reading: {latest_data['avg_energy_kwh'].mean():.2f} kWh

Waste Management:
- Average occupancy rate: {latest_data['avg_occupancy_rate'].mean():.1f}%

Environmental:
- Average noise level: {latest_data['avg_noise_level'].mean():.1f} dB

Provide a brief 3-4 sentence analysis highlighting key trends, patterns, and any concerns or recommendations.
"""
    
    return query_gemma(data_summary, max_tokens=300, temperature=0.5)

def analyze_proposal(proposal_text, data):
    """Analyze user's development proposal using Gemma model"""
    
    # Prepare comprehensive data context
    data_context = f"""
Current Smart City Infrastructure Status:

Available Cities: {', '.join(data['city_id'].unique())}

Current Infrastructure Metrics (Recent Averages):
Traffic:
- Daily average vehicles: {data.groupby('city_id')['avg_vehicles'].mean().to_dict()}
- Peak traffic hours: Varies by city

Energy Consumption:
- Daily average: {data.groupby('city_id')['total_energy_kwh'].mean().to_dict()} kWh

Waste Management:
- Average bin occupancy: {data.groupby('city_id')['avg_occupancy_rate'].mean().to_dict()}%

Environmental:
- Noise levels: {data.groupby('city_id')['avg_noise_level'].mean().to_dict()} dB

User's Development Proposal:
{proposal_text}

As an expert urban planner, analyze this proposal considering the current infrastructure data. Provide:
1. Impact Assessment: How will this affect traffic, energy, waste, and environment?
2. Advantages: What are the key benefits?
3. Challenges/Concerns: What issues might arise?
4. Recommendations: Specific suggestions for implementation
5. Data-Driven Insights: Use the provided metrics to support your analysis

Format your response in clear sections with markdown.
"""
    
    return query_gemma(data_context, max_tokens=2000, temperature=0.7)

# ==================== MAIN APP ====================

st.title("🏙️ Nagara-Sthapati: Smart City Infrastructure Planner")
st.markdown("*AI-Powered Urban Planning & Analytics*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    view_mode = st.radio("Select View", ["📊 City Infographics", "📝 Proposal Analysis"])
    
    st.divider()
    
    if view_mode == "📊 City Infographics":
        time_granularity = st.selectbox("Time Granularity", ["Hourly", "Daily"])
    
    st.divider()
    st.markdown("### About")
    st.info("This app uses Gemma 3 12B model to analyze smart city infrastructure data and provide insights for urban planning decisions.")

# Load data
try:
    if view_mode == "📊 City Infographics":
        if time_granularity == "Hourly":
            data = load_hourly_data()
            time_col = 'event_hour'
        else:
            data = load_daily_data()
            time_col = 'event_date'
    else:
        data = load_daily_data()  # Use daily data for proposals
        time_col = 'event_date'
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ==================== FEATURE 1: CITY INFOGRAPHICS ====================
if view_mode == "📊 City Infographics":
    st.header(f"📊 Smart City Infrastructure Dashboard ({time_granularity})")
    
    # City selector
    cities = data['city_id'].unique()
    selected_city = st.selectbox("Select City", cities)
    
    # Filter data for selected city
    city_data = data[data['city_id'] == selected_city].sort_values(time_col, ascending=False)
    
    # Latest metrics
    latest = city_data.iloc[0]
    previous = city_data.iloc[1] if len(city_data) > 1 else latest
    
    st.subheader(f"Latest Metrics for {selected_city}")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚗 Avg Vehicles",
            f"{latest['avg_vehicles']:.0f}",
            f"{((latest['avg_vehicles'] - previous['avg_vehicles']) / previous['avg_vehicles'] * 100):.1f}%" if previous['avg_vehicles'] > 0 else "N/A"
        )
    
    with col2:
        st.metric(
            "⚡ Energy (kWh)",
            f"{latest['total_energy_kwh']:.0f}",
            f"{((latest['total_energy_kwh'] - previous['total_energy_kwh']) / previous['total_energy_kwh'] * 100):.1f}%" if previous['total_energy_kwh'] > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            "🗑️ Waste Occupancy",
            f"{latest['avg_occupancy_rate']:.1f}%",
            f"{(latest['avg_occupancy_rate'] - previous['avg_occupancy_rate']):.1f}%" if previous['avg_occupancy_rate'] > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            "🔊 Noise Level",
            f"{latest['avg_noise_level']:.1f} dB",
            f"{(latest['avg_noise_level'] - previous['avg_noise_level']):.1f} dB" if previous['avg_noise_level'] > 0 else "N/A"
        )
    
    st.divider()
    
    # AI Insights
    st.subheader("🤖 AI-Generated Insights")
    with st.spinner("Analyzing data with Gemma 3 12B..."):
        insights = generate_infographic_insights(city_data.head(24), "hourly" if time_granularity == "Hourly" else "daily")
        st.info(insights)
    
    st.divider()
    
    # Visualizations
    st.subheader("📈 Trend Analysis")
    
    # Prepare data for last 7 days/hours
    recent_data = city_data.head(168 if time_granularity == "Hourly" else 7)
    
    # Traffic trend
    col1, col2 = st.columns(2)
    
    with col1:
        fig_traffic = px.line(
            recent_data,
            x=time_col,
            y='avg_vehicles',
            title=f'Traffic Trend - {selected_city}',
            labels={'avg_vehicles': 'Average Vehicles', time_col: 'Time'}
        )
        fig_traffic.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_traffic, use_container_width=True)
    
    with col2:
        fig_energy = px.line(
            recent_data,
            x=time_col,
            y='total_energy_kwh',
            title=f'Energy Consumption - {selected_city}',
            labels={'total_energy_kwh': 'Total Energy (kWh)', time_col: 'Time'}
        )
        fig_energy.update_traces(line_color='#ff7f0e', line_width=3)
        st.plotly_chart(fig_energy, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_waste = px.line(
            recent_data,
            x=time_col,
            y='avg_occupancy_rate',
            title=f'Waste Bin Occupancy - {selected_city}',
            labels={'avg_occupancy_rate': 'Occupancy Rate (%)', time_col: 'Time'}
        )
        fig_waste.update_traces(line_color='#2ca02c', line_width=3)
        st.plotly_chart(fig_waste, use_container_width=True)
    
    with col4:
        fig_noise = px.line(
            recent_data,
            x=time_col,
            y='avg_noise_level',
            title=f'Noise Levels - {selected_city}',
            labels={'avg_noise_level': 'Noise Level (dB)', time_col: 'Time'}
        )
        fig_noise.update_traces(line_color='#d62728', line_width=3)
        st.plotly_chart(fig_noise, use_container_width=True)
    
    # Sensor type breakdown
    st.subheader("📊 Metrics by Sensor Type")
    sensor_summary = city_data.groupby('sensor_type').agg({
        'avg_vehicles': 'mean',
        'total_energy_kwh': 'sum',
        'avg_occupancy_rate': 'mean',
        'avg_noise_level': 'mean',
        'unique_sensors': 'sum'
    }).reset_index()
    
    st.dataframe(sensor_summary, use_container_width=True)

# ==================== FEATURE 2: PROPOSAL ANALYSIS ====================
else:
    st.header("📝 Development Proposal Analysis")
    
    st.markdown("""
    Submit your urban development proposal or city planning idea. The AI will analyze it against 
    current infrastructure data and provide comprehensive insights.
    """)
    
    # Proposal input
    proposal = st.text_area(
        "Enter your development proposal:",
        placeholder="Example: I propose building a new shopping mall in Istanbul's central district with 500 parking spaces, solar panels on the roof, and a waste recycling facility...",
        height=150
    )
    
    # City context selector
    selected_cities = st.multiselect(
        "Select cities for context (analysis will consider these cities' data)",
        data['city_id'].unique(),
        default=list(data['city_id'].unique()[:1])
    )
    
    if st.button("🚀 Analyze Proposal", type="primary"):
        if not proposal:
            st.warning("Please enter a proposal first!")
        else:
            # Filter data for selected cities
            context_data = data[data['city_id'].isin(selected_cities)]
            
            with st.spinner("Analyzing your proposal with Gemma 3 12B... This may take a moment."):
                analysis = analyze_proposal(proposal, context_data)
            
            st.success("Analysis Complete!")
            
            # Display analysis
            st.markdown("### 📋 AI Analysis Report")
            st.markdown(analysis)
            
            st.divider()
            
            # Show relevant data context
            st.subheader("📊 Supporting Data Context")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cities Analyzed", len(selected_cities))
                st.metric("Data Points", len(context_data))
            
            with col2:
                st.metric("Avg Daily Traffic", f"{context_data['avg_vehicles'].mean():.0f}")
                st.metric("Avg Energy (kWh)", f"{context_data['total_energy_kwh'].mean():.0f}")
            
            with col3:
                st.metric("Avg Waste Occupancy", f"{context_data['avg_occupancy_rate'].mean():.1f}%")
                st.metric("Avg Noise Level", f"{context_data['avg_noise_level'].mean():.1f} dB")
            
            # Comparative visualization
            st.subheader("📈 Current Infrastructure Trends")
            
            comparison_data = context_data.groupby(['city_id', 'sensor_type']).agg({
                'avg_vehicles': 'mean',
                'total_energy_kwh': 'mean',
                'avg_occupancy_rate': 'mean',
                'avg_noise_level': 'mean'
            }).reset_index()
            
            fig_comparison = px.bar(
                comparison_data,
                x='city_id',
                y='avg_vehicles',
                color='sensor_type',
                title='Average Traffic by City and Sensor Type',
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

# Footer
st.divider()
st.markdown("*Powered by Databricks + Gemma 3 12B | Data Pipeline: Smart City Infrastructure*")
