# 🚀 Nagara Sthapati: A City Policy Simulation Platform

## 📌 Overview
This project builds a **City Digital Twin on Databricks** that enables urban planners to simulate and optimize policy decisions such as pricing for transport, water, and fuel.

The system models how citizens respond to policy changes and provides **real-time insights into demand, revenue, and infrastructure stress**.

---

## 🎯 Problem Statement
Urban policy decisions (e.g., fare pricing, fuel taxation) are often made without understanding:
- behavioral response of citizens  
- impact on infrastructure (traffic, congestion)  
- trade-offs between revenue and system efficiency  

---

## 💡 Solution
We developed an **AI-powered simulation platform** that:
- models city behavior using synthetic but realistic data  
- predicts demand using machine learning  
- simulates system-level interactions  
- recommends optimal policies  

---

## 🏗️ Architecture

- **Data Layer**: Synthetic city data stored in Delta tables  
- **ML Layer**: Random Forest model trained using MLflow  
- **Simulation Layer**: Converts policy inputs → system outputs  
- **Application Layer**: Streamlit UI for real-time interaction  

---

## ⚙️ Components

### 1. Data Generation (`01_data_generation.ipynb`)
- Generates synthetic city data with:
  - population drift  
  - congestion  
  - festival/holiday effects  
- Stored as Delta table: `city_data`

---

### 2. Model Training (`02_model_training.ipynb`)
- Trains demand prediction model using:
  - price  
  - congestion  
  - population  
  - temporal signals  
- Logs experiments using MLflow  
- Registers model: `city_demand_forecast`

---

### 3. Simulation Engine (`03_simulation.ipynb`)
- Inputs: policy variables (prices, taxes)  
- Outputs:
  - demand per service  
  - revenue  
- Captures system effects (traffic, behavior)

---

### 4. Streamlit App (`app.py`)
Interactive UI for planners to:
- adjust policy variables  
- visualize demand & revenue  
- simulate real-time impact  

---

## 🧠 Key Features

- ✅ Real-time simulation  
- ✅ ML-driven demand prediction  
- ✅ Policy experimentation  
- ✅ Interactive visualization  
- ✅ Reproducible via Databricks + MLflow  

---

## 📊 Sample Outputs

- Demand by service (Metro, Bus, etc.)
- Revenue impact
- Policy sensitivity

---

## 🧪 How to Run

### Step 1: Run Data Generation
```
01_data_generation.ipynb
```

### Step 2: Train Model
```
02_model_training.ipynb
```

### Step 3: Register Model in MLflow
- Navigate to MLflow UI  
- Register model as `city_demand_forecast`

### Step 4: Run Simulation / App
```
streamlit run app.py
```

---

## 🚀 Future Enhancements

- Real smart city sensor integration  
- Multi-objective optimization (Pareto frontier)  
- Causal inference for policy impact  
- Real-time streaming simulation  

---

## 🏆 Impact

This system enables planners to:
- test policies before implementation  
- understand trade-offs  
- move toward **data-driven governance**
