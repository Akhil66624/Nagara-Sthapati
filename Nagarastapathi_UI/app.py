"""
Smart City Infrastructure Planner App
A comprehensive Streamlit dashboard for urban planning and development analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Smart City Infrastructure Planner",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Spark session and load data
@st.cache_resource
def get_spark_session():
    """Get or create Spark session"""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark session: {e}")
        return None

@st.cache_data(ttl=300)
def load_data(granularity='hourly'):
    """Load sensor metrics data from gold layer"""
    spark = get_spark_session()
    if spark is None:
        return pd.DataFrame()
    
    try:
        if granularity == 'hourly':
            table_name = "workspace.smart_city.gold_sensor_metrics_by_city_hour"
        else:
            table_name = "workspace.smart_city.gold_sensor_metrics_by_city_day"
        
        df = spark.table(table_name).toPandas()
        
        # Convert timestamp columns to datetime
        if 'event_hour' in df.columns:
            df['event_hour'] = pd.to_datetime(df['event_hour'])
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'])
        
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def get_ai_insights(data_summary, question_type="general"):
    """Generate AI insights using Databricks Foundation Model API"""
    try:
        # Get Databricks token and host
        from dbutils import DBUtils
        dbutils = DBUtils(get_spark_session())
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        
        endpoint = f"{host}/serving-endpoints/databricks-gemma-3-12b/invocations"
        
        if question_type == "general":
            system_prompt = """You are an expert urban planner and data analyst specializing in smart city infrastructure. 
            Analyze the provided sensor data and provide concise, actionable insights about traffic patterns, energy usage, 
            waste management, and environmental conditions. Focus on trends, anomalies, and recommendations."""
        else:  # proposal analysis
            system_prompt = """You are an expert urban planner evaluating development proposals. 
            Analyze the proposal in the context of the provided smart city sensor data. 
            Provide a structured assessment covering:
            1. Impact Assessment (traffic, energy, waste, environment)
            2. Key Advantages
            3. Potential Challenges/Concerns
            4. Recommendations
            5. Data-Driven Insights
            
            Be specific, cite data trends, and provide actionable recommendations."""
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_summary}
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"AI service temporarily unavailable (Status: {response.status_code})"
            
    except Exception as e:
        return f"Unable to generate AI insights: {str(e)}"

def calculate_trend(current, previous):
    """Calculate percentage change"""
    if previous == 0 or pd.isna(previous):
        return 0
    return ((current - previous) / previous) * 100

# =====================
# SIDEBAR NAVIGATION
# =====================
st.sidebar.title("🏙️ Smart City Planner")
page = st.sidebar.radio(
    "Navigation",
    ["📊 City Infographics", "📝 Development Proposal Analyzer"]
)

# =====================
# PAGE 1: CITY INFOGRAPHICS DASHBOARD
# =====================
if page == "📊 City Infographics":
    st.markdown('<div class="main-header">🏙️ Smart City Infrastructure Dashboard</div>', unsafe_allow_html=True)
    
    # Controls
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_city = st.selectbox("Select City", ["Istanbul", "Ankara", "All Cities"])
    with col2:
        time_granularity = st.radio("Time Granularity", ["Hourly", "Daily"], horizontal=True)
    
    # Load data based on granularity
    granularity = 'hourly' if time_granularity == "Hourly" else 'daily'
    df = load_data(granularity)
    
    if df.empty:
        st.error("No data available. Please check the pipeline and data tables.")
        st.stop()
    
    # Filter by city
    if selected_city != "All Cities":
        df_filtered = df[df['city_id'] == selected_city].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_city}")
        st.stop()
    
    # Get latest period and previous period for comparison
    if granularity == 'hourly':
        time_col = 'event_hour'
        latest_time = df_filtered[time_col].max()
        previous_time = latest_time - timedelta(hours=1)
    else:
        time_col = 'event_date'
        latest_time = df_filtered[time_col].max()
        previous_time = latest_time - timedelta(days=1)
    
    current_data = df_filtered[df_filtered[time_col] == latest_time]
    previous_data = df_filtered[df_filtered[time_col] == previous_time]
    
    # Aggregate metrics
    current_metrics = {
        'traffic': current_data['total_vehicles'].sum(),
        'energy': current_data['total_energy_kwh'].sum(),
        'waste': current_data['avg_occupancy_rate'].mean(),
        'noise': current_data['avg_noise_level'].mean()
    }
    
    previous_metrics = {
        'traffic': previous_data['total_vehicles'].sum() if not previous_data.empty else 0,
        'energy': previous_data['total_energy_kwh'].sum() if not previous_data.empty else 0,
        'waste': previous_data['avg_occupancy_rate'].mean() if not previous_data.empty else 0,
        'noise': previous_data['avg_noise_level'].mean() if not previous_data.empty else 0
    }
    
    # Display metrics with trends
    st.subheader("📈 Real-Time Metrics")
    cols = st.columns(4)
    
    metrics_config = [
        ("🚦 Traffic", current_metrics['traffic'], previous_metrics['traffic'], "vehicles", ""),
        ("⚡ Energy", current_metrics['energy'], previous_metrics['energy'], "kWh", ""),
        ("🗑️ Waste", current_metrics['waste'], previous_metrics['waste'], "% occupancy", ".1f"),
        ("🔊 Noise", current_metrics['noise'], previous_metrics['noise'], "dB", ".1f")
    ]
    
    for i, (label, current, previous, unit, fmt) in enumerate(metrics_config):
        trend = calculate_trend(current, previous)
        delta_color = "normal" if abs(trend) < 5 else ("off" if trend > 5 else "inverse")
        
        with cols[i]:
            if fmt:
                st.metric(
                    label=label,
                    value=f"{current:{fmt}} {unit}",
                    delta=f"{trend:+.1f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(
                    label=label,
                    value=f"{int(current):,} {unit}",
                    delta=f"{trend:+.1f}%",
                    delta_color=delta_color
                )
    
    # AI Insights
    st.subheader("🤖 AI-Powered Insights")
    with st.spinner("Analyzing data patterns..."):
        data_summary = f"""
        City: {selected_city}
        Time Period: Latest {time_granularity.lower()} data
        
        Current Metrics:
        - Traffic: {int(current_metrics['traffic']):,} vehicles (trend: {calculate_trend(current_metrics['traffic'], previous_metrics['traffic']):+.1f}%)
        - Energy: {int(current_metrics['energy']):,} kWh (trend: {calculate_trend(current_metrics['energy'], previous_metrics['energy']):+.1f}%)
        - Waste Occupancy: {current_metrics['waste']:.1f}% (trend: {calculate_trend(current_metrics['waste'], previous_metrics['waste']):+.1f}%)
        - Noise Level: {current_metrics['noise']:.1f} dB (trend: {calculate_trend(current_metrics['noise'], previous_metrics['noise']):+.1f}%)
        
        Recent Trends (last 24 {time_granularity.lower()} periods):
        - Total sensors monitored: {current_data['unique_sensors'].sum() if not current_data.empty else 0}
        - Total records processed: {current_data['record_count'].sum() if not current_data.empty else 0}
        """
        
        insights = get_ai_insights(data_summary, "general")
        st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("📊 Trend Analysis")
    
    # Prepare time series data (last 7 days or 24 hours)
    if granularity == 'hourly':
        cutoff_time = latest_time - timedelta(hours=24)
    else:
        cutoff_time = latest_time - timedelta(days=7)
    
    df_recent = df_filtered[df_filtered[time_col] >= cutoff_time].copy()
    df_recent = df_recent.sort_values(time_col)
    
    # Group by time for aggregate view
    df_chart = df_recent.groupby(time_col).agg({
        'total_vehicles': 'sum',
        'total_energy_kwh': 'sum',
        'avg_occupancy_rate': 'mean',
        'avg_noise_level': 'mean'
    }).reset_index()
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_traffic = px.line(df_chart, x=time_col, y='total_vehicles',
                             title='🚦 Traffic Volume Over Time',
                             labels={time_col: 'Time', 'total_vehicles': 'Total Vehicles'})
        fig_traffic.update_traces(line_color='#FF6B6B')
        st.plotly_chart(fig_traffic, use_container_width=True)
        
        fig_waste = px.line(df_chart, x=time_col, y='avg_occupancy_rate',
                           title='🗑️ Waste Container Occupancy',
                           labels={time_col: 'Time', 'avg_occupancy_rate': 'Occupancy Rate (%)'})
        fig_waste.update_traces(line_color='#95E1D3')
        st.plotly_chart(fig_waste, use_container_width=True)
    
    with col2:
        fig_energy = px.line(df_chart, x=time_col, y='total_energy_kwh',
                            title='⚡ Energy Consumption',
                            labels={time_col: 'Time', 'total_energy_kwh': 'Energy (kWh)'})
        fig_energy.update_traces(line_color='#F38181')
        st.plotly_chart(fig_energy, use_container_width=True)
        
        fig_noise = px.line(df_chart, x=time_col, y='avg_noise_level',
                           title='🔊 Noise Levels',
                           labels={time_col: 'Time', 'avg_noise_level': 'Noise Level (dB)'})
        fig_noise.update_traces(line_color='#AA96DA')
        st.plotly_chart(fig_noise, use_container_width=True)
    
    # Sensor type breakdown
    st.subheader("🎯 Sensor Type Breakdown")
    sensor_breakdown = df_recent.groupby('sensor_type').agg({
        'record_count': 'sum',
        'unique_sensors': 'sum',
        'total_vehicles': 'sum',
        'total_energy_kwh': 'sum',
        'avg_occupancy_rate': 'mean',
        'avg_noise_level': 'mean'
    }).reset_index()
    
    st.dataframe(sensor_breakdown, use_container_width=True, hide_index=True)

# =====================
# PAGE 2: DEVELOPMENT PROPOSAL ANALYZER
# =====================
elif page == "📝 Development Proposal Analyzer":
    st.markdown('<div class="main-header">📝 Urban Development Proposal Analyzer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Submit your urban development proposal for AI-powered analysis based on real smart city sensor data.
    The system will evaluate potential impacts on traffic, energy, waste management, and environmental factors.
    """)
    
    # Proposal input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        proposal_text = st.text_area(
            "Development Proposal",
            placeholder="Example: Construct a new shopping mall with 200,000 sq ft retail space, 1,500 parking spots, "
                       "and solar panels on the roof in the Kadıköy district of Istanbul...",
            height=200
        )
    
    with col2:
        context_cities = st.multiselect(
            "Context Cities (for data comparison)",
            ["Istanbul", "Ankara"],
            default=["Istanbul"]
        )
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard"
        )
    
    analyze_button = st.button("🔍 Analyze Proposal", type="primary", use_container_width=True)
    
    if analyze_button and proposal_text:
        with st.spinner("Analyzing proposal against smart city data..."):
            # Load relevant data for context cities
            df = load_data('daily')  # Use daily data for broader trends
            
            context_data = []
            for city in context_cities:
                city_data = df[df['city_id'] == city]
                if not city_data.empty:
                    # Get recent averages (last 30 days)
                    recent_cutoff = city_data['event_date'].max() - timedelta(days=30)
                    recent_data = city_data[city_data['event_date'] >= recent_cutoff]
                    
                    summary = {
                        'city': city,
                        'avg_daily_traffic': recent_data['total_vehicles'].mean(),
                        'avg_daily_energy': recent_data['total_energy_kwh'].mean(),
                        'avg_waste_occupancy': recent_data['avg_occupancy_rate'].mean(),
                        'avg_noise_level': recent_data['avg_noise_level'].mean(),
                        'total_sensors': recent_data['unique_sensors'].sum(),
                        'date_range': f"{recent_data['event_date'].min().date()} to {recent_data['event_date'].max().date()}"
                    }
                    context_data.append(summary)
            
            # Build analysis context
            context_summary = f"""
            Urban Development Proposal:
            {proposal_text}
            
            Smart City Context Data (Last 30 Days):
            """
            
            for ctx in context_data:
                context_summary += f"""
                
                {ctx['city']}:
                - Average Daily Traffic: {int(ctx['avg_daily_traffic']):,} vehicles
                - Average Daily Energy Consumption: {int(ctx['avg_daily_energy']):,} kWh
                - Average Waste Container Occupancy: {ctx['avg_waste_occupancy']:.1f}%
                - Average Noise Level: {ctx['avg_noise_level']:.1f} dB
                - Active Sensors: {int(ctx['total_sensors'])}
                - Data Period: {ctx['date_range']}
                """
            
            context_summary += f"""
            
            Analysis Depth: {analysis_depth}
            
            Please provide a {analysis_depth.lower()} analysis of this development proposal considering:
            1. Traffic impact and infrastructure requirements
            2. Energy demand and sustainability considerations
            3. Waste management implications
            4. Environmental and noise impact
            5. Recommendations for optimal implementation
            """
            
            # Get AI analysis
            analysis = get_ai_insights(context_summary, "proposal")
            
            # Display analysis
            st.markdown("---")
            st.subheader("📋 Analysis Results")
            st.markdown(f'<div class="insight-box">{analysis}</div>', unsafe_allow_html=True)
            
            # Display supporting data
            st.markdown("---")
            st.subheader("📊 Supporting Data Context")
            
            cols = st.columns(len(context_data))
            for i, ctx in enumerate(context_data):
                with cols[i]:
                    st.markdown(f"**{ctx['city']}** (30-day avg)")
                    st.metric("🚦 Daily Traffic", f"{int(ctx['avg_daily_traffic']):,}")
                    st.metric("⚡ Daily Energy", f"{int(ctx['avg_daily_energy']):,} kWh")
                    st.metric("🗑️ Waste Occupancy", f"{ctx['avg_waste_occupancy']:.1f}%")
                    st.metric("🔊 Noise Level", f"{ctx['avg_noise_level']:.1f} dB")
            
            # Comparative visualization
            if len(context_data) > 0:
                st.markdown("---")
                st.subheader("📈 Comparative Metrics")
                
                comparison_df = pd.DataFrame(context_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Traffic', x=comparison_df['city'], 
                                    y=comparison_df['avg_daily_traffic'], marker_color='#FF6B6B'))
                fig.update_layout(title='Average Daily Traffic by City', yaxis_title='Vehicles')
                st.plotly_chart(fig, use_container_width=True)
    
    elif analyze_button:
        st.warning("Please enter a development proposal to analyze.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🏙️ <strong>Nagara-Sthapati Smart City Infrastructure Planner</strong></p>
        <p>Powered by Databricks Lakehouse Platform | Data refreshed every 5 minutes</p>
    </div>
""", unsafe_allow_html=True)
