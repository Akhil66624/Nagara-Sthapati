# Nagara-Sthapati: Smart City Infrastructure Planner

## 🎯 Overview

This Streamlit app provides AI-powered urban planning analytics using your smart city sensor data and the Gemma 3 12B reasoning model.

## ✨ Features

### 1. **City Infographics Dashboard** 📊
- Real-time metrics: Traffic, Energy, Waste, Environment
- Hourly and Daily trend analysis
- AI-generated insights powered by Gemma 3 12B
- Interactive visualizations with Plotly

### 2. **Development Proposal Analyzer** 📝
- Submit urban development proposals as text
- AI analyzes impact on infrastructure
- Provides advantages, challenges, and recommendations
- Data-driven insights with supporting visualizations

## 🏗️ Architecture

```
Smart City Data Pipeline (Lakeflow)
    ↓
Bronze → Silver → Gold Tables
    ↓
Streamlit App ← Gemma 3 12B Model (Foundation Model API)
    ↓
User Interface
```

## 📦 Installation & Setup

### Prerequisites
- Databricks workspace with serverless compute enabled
- Access to Foundation Model APIs (Gemma 3 12B)
- Smart City Infrastructure Pipeline running (with gold tables)

### Step 1: Install Dependencies

The app requires these Python packages (already listed in `requirements.txt`):
- streamlit
- pandas
- plotly
- requests
- pyspark

**Note**: These packages are typically pre-installed in Databricks ML Runtime. If running elsewhere, install via:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Data Availability

Ensure your pipeline gold tables exist and have data:
```sql
SELECT COUNT(*) FROM workspace.smart_city.gold_sensor_metrics_by_city_hour;
SELECT COUNT(*) FROM workspace.smart_city.gold_sensor_metrics_by_city_day;
```

### Step 3: Run the Streamlit App

#### Option A: Run from Databricks Notebook

1. Create a new Python notebook
2. In cell 1, run:
```python
%pip install streamlit plotly
```

3. In cell 2, run:
```python
import subprocess
import sys

# Run Streamlit app
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "/Workspace/Users/am23s045@smail.iitm.ac.in/Nagara-Sthapati/smart_city_planner_app.py",
    "--server.port", "8501",
    "--server.headless", "true"
])
```

4. Access the app at the provided URL (Databricks will show a proxy URL)

#### Option B: Run from Terminal (if SSH access available)

```bash
cd /Workspace/Users/am23s045@smail.iitm.ac.in/Nagara-Sthapati
streamlit run smart_city_planner_app.py
```

## 🚀 Using the App

### Feature 1: City Infographics

1. **Select View**: Choose "📊 City Infographics" from sidebar
2. **Time Granularity**: Choose "Hourly" or "Daily"
3. **Select City**: Pick a city from the dropdown
4. **View Insights**:
   - Latest metrics with trend indicators
   - AI-generated analysis from Gemma 3 12B
   - Interactive trend charts
   - Sensor type breakdown

### Feature 2: Proposal Analysis

1. **Select View**: Choose "📝 Proposal Analysis" from sidebar
2. **Enter Proposal**: Type your development idea/proposal
   - Example: *"Build a new metro station in Istanbul near the business district with 2000 daily capacity, solar-powered, and green roof gardens"*
3. **Select Cities**: Choose which cities' data to use for context
4. **Click Analyze**: The AI will provide:
   - Impact assessment
   - Advantages and challenges
   - Implementation recommendations
   - Supporting data visualizations

## 🤖 How It Works

### Gemma 3 12B Integration

The app uses Databricks Foundation Model API to query Gemma 3 12B:

```python
def query_gemma(prompt, max_tokens=2000, temperature=0.7):
    url = f"{DATABRICKS_HOST}/serving-endpoints/databricks-gemma-3-12b/invocations"
    # API call with system prompt and user context
```

**System Prompt**: Configured as an expert urban planner and data analyst
**Context**: App provides current infrastructure metrics from your gold tables
**Analysis**: Model reasons over data and generates insights

### Data Flow

1. **Load Data**: App queries gold tables (cached for 5 minutes)
2. **Prepare Context**: Summarizes metrics relevant to the query
3. **Query Gemma**: Sends context + user input to the model
4. **Display Results**: Renders AI response + supporting visualizations

## 📊 Data Tables Used

- `workspace.smart_city.gold_sensor_metrics_by_city_hour`
  - Hourly aggregated metrics
  - 122,147 rows processed

- `workspace.smart_city.gold_sensor_metrics_by_city_day`
  - Daily aggregated metrics
  - 5,117 rows processed

## 🔧 Customization

### Modify Model Parameters

In `smart_city_planner_app.py`, adjust the `query_gemma` function:

```python
# More creative responses
query_gemma(prompt, temperature=0.9)

# More focused/deterministic responses
query_gemma(prompt, temperature=0.3)

# Longer responses
query_gemma(prompt, max_tokens=3000)
```

### Add New Metrics

To display additional metrics:

1. Ensure they exist in your gold tables
2. Add to the data loading functions
3. Create new visualizations in the appropriate section

### Change Model

To use a different model (e.g., Gemini 2.5 Flash):

```python
GEMMA_ENDPOINT = "databricks-gemini-2-5-flash"  # Change this line
```

## 🐛 Troubleshooting

### Error: "Cannot connect to Foundation Model API"
- **Solution**: Verify Foundation Model API is enabled in your workspace settings
- Check that `databricks-gemma-3-12b` endpoint is available

### Error: "Table not found"
- **Solution**: Ensure pipeline has run successfully
- Verify gold tables exist: `SHOW TABLES IN workspace.smart_city`

### Error: "Streamlit module not found"
- **Solution**: Install dependencies: `%pip install streamlit plotly`

### Slow Response from AI
- **Cause**: Large data context or complex prompts
- **Solution**: 
  - Reduce `max_tokens` parameter
  - Filter data to smaller time windows
  - Use daily data instead of hourly for proposals

## 📈 Performance Tips

1. **Cache Duration**: Adjust `@st.cache_data(ttl=300)` to balance freshness vs. speed
2. **Data Filtering**: Pre-filter data in SQL queries for faster loading
3. **Async Queries**: For multiple AI calls, consider async implementation

## 🔐 Security Notes

- API tokens are automatically retrieved from Databricks context
- No hardcoded credentials in the app
- Data stays within Databricks security perimeter

## 📝 Next Steps

- **Deploy as Databricks App**: Consider deploying as a managed Databricks App for production use
- **Add User Authentication**: Implement role-based access if needed
- **Extend Models**: Try other Foundation Models (Llama, Mistral, etc.)
- **Add More Features**: 
  - Historical comparison
  - Predictive analytics
  - Multi-city comparison
  - Export reports to PDF

## 🆘 Support

For issues or questions:
- Check Databricks documentation: [Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/)
- Review Streamlit docs: [streamlit.io/docs](https://docs.streamlit.io/)

---

**Built with ❤️ using Databricks, Streamlit, and Gemma 3 12B**
