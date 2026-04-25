#!pip install streamlit
import streamlit as st
import mlflow.pyfunc
import pandas as pd
import plotly.express as px


model_version_uri = f"models:/workspace.default.city_demand_forecast/1"
model = mlflow.pyfunc.load_model(model_version_uri)

# model = mlflow.pyfunc.load_model("models:/workspace.default.city_demand_forecast/Production")

st.title("🌆 City Policy Simulator")

metro_price = st.slider("Metro Price", 10.0,100.0, 30.0)
bus_price = st.slider("Bus Price", 5.0, 50.0, 10.0)
fuel_price = st.slider("Fuel Price", 50.0, 150.0, 80.0)
water_price = st.slider("Water Price", 1.0, 20.0, 5.0)

fuel_tax = st.slider("Fuel Tax", 0.0, 0.3, 0.1)
festival = st.selectbox("Festival Week", [0,1])
holiday = st.selectbox("Holiday", [0,1])

policy = {
    "metro_price": metro_price,
    "bus_price": bus_price,
    "private_transport_price": fuel_price,
    "water_price": water_price,
    "fuel_tax": fuel_tax,
    "festival": festival,
    "holiday": holiday
}

def simulate(policy):
    results = []
    for s in ["metro","bus","private_transport","water"]:
        price = policy[f"{s}_price"]

        df = pd.DataFrame([{
            "price": price,
            "population": 1.1e7,
            "congestion": 0.65 + 0.3 * fuel_tax,
            "signal_load": 0.6,
            "festival": festival,
            "holiday": holiday
        }])

        demand = model.predict(df)[0]
        results.append({"service": s, "demand": demand, "revenue": demand*price})

    return pd.DataFrame(results)

df_res = simulate(policy)

st.plotly_chart(px.bar(df_res, x="service", y="demand"))
st.plotly_chart(px.bar(df_res, x="service", y="revenue"))
## saved
