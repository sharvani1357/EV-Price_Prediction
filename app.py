import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils import get_demand_data, get_region_data, predict_demand, get_model_evaluation_data, get_world_summary_data

# Set page configuration
st.set_page_config(
    page_title="EV Energy Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #757575;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .highlight-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("⚡ Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Page", ["Dashboard", "Predict Demand", "Model Evaluation"])

st.sidebar.markdown("---")
st.sidebar.info("This application predicts Electric Vehicle (EV) energy demand to optimize grid load and charging station operations.")


# ==========================================
# PAGE 1: DEMAND PREDICTION DASHBOARD
# ==========================================
if page == "Dashboard":
    st.markdown('<p class="main-header">📊 Demand Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Overview of current and forecasted EV energy demand across regions</p>', unsafe_allow_html=True)
    
    # Generate placeholder data
    df_time = get_demand_data()
    df_region = get_region_data()
    
    # Summary Cards
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    current_demand = df_time['Demand_MW'].iloc[-1]
    avg_demand = df_time['Demand_MW'].mean()
    peak_demand = df_time['Demand_MW'].max()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Demand</h4>
            <h2 style="color: #1E88E5;">{current_demand:.1f} MW</h2>
            <p style="color: green;">↑ 2.4% from last hour</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Demand (24h)</h4>
            <h2 style="color: #FF9800;">{avg_demand:.1f} MW</h2>
            <p style="color: gray;">Active across 4 regions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Peak Demand (24h)</h4>
            <h2 style="color: #F44336;">{peak_demand:.1f} MW</h2>
            <p style="color: #F44336;">Expected at 18:00</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Interactive Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📉 Demand Over Time (24h Forecast)")
        fig_line = px.line(df_time, x='Time', y='Demand_MW', 
                           markers=True, 
                           line_shape='spline',
                           color_discrete_sequence=['#1E88E5'])
        fig_line.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Time of Day", yaxis_title="Energy Demand (MW)")
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_chart2:
        st.markdown("### 🗺️ Demand by Region")
        fig_bar = px.bar(df_region, x='Region', y='Demand_MW',
                         color='Region',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_bar.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Region/Station", yaxis_title="Energy Demand (MW)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 🌍 Global EV Charging Station Distribution")
    
    df_world = get_world_summary_data()
    if not df_world.empty:
        # Create a choropleth map using Plotly
        fig_map = px.choropleth(
            df_world,
            locations="country", 
            locationmode="country names",
            color="Demand_Category",
            hover_name="country",
            hover_data={
                "Demand_Category": True,
                "station_count": True,
                "port_count": True,
                "Demand_Score": ":.1f"
            },
            color_discrete_map={
                'High': '#D32F2F',    # Red
                'Medium': '#FBC02D',  # Yellow
                'Low': '#388E3C'      # Green
            },
            title="Categorized EV Demand by Country (Based on Infrastructure Density)"
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
            legend_title="Demand Level"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Display some key stats 
        with st.expander("View Global Network Stats"):
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Stations Globally", f"{df_world['station_count'].sum():,}")
            with col_stat2:
                st.metric("Total Ports Globally", f"{df_world['port_count'].sum():,}")
            with col_stat3:
                st.metric("Fast Stations Globally", f"{df_world['fast_station_count'].sum():,}")
    else:
        st.info("World summary data not found. Please ensure 'world_summary.csv' is in the data directory.")


# ==========================================
# PAGE 2: USER INPUT PAGE
# ==========================================
elif page == "Predict Demand":
    st.markdown('<p class="main-header">⚡ Predict Energy Demand</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter parameters to forecast EV charging station load</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 📝 Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.selectbox("📍 Charging Station Location", ["Downtown Hub", "Northside Mall", "Airport Express", "Highway 101 Rest Stop", "Residential District A"])
            time_of_day = st.time_input("🕒 Time of Day", value=datetime.strptime("17:00", "%H:%M").time())
            weather = st.selectbox("🌤️ Weather Conditions", ["Clear", "Rainy", "Snow", "Extreme Heat", "Extreme Cold"])
            
        with col2:
            num_evs = st.slider("🚗 Expected Number of EVs", min_value=1, max_value=500, value=50)
            station_capacity = st.number_input("🔌 Station Total Capacity (MW)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
            day_type = st.radio("📅 Day Type", ["Weekday", "Weekend", "Holiday"], horizontal=True)

    st.markdown("---")
    
    if st.button("🚀 Predict Demand", use_container_width=True):
        input_data = {
            "Location": location,
            "Time": str(time_of_day),
            "Weather": weather,
            "Num_EVs": num_evs,
            "Capacity": station_capacity,
            "DayType": day_type
        }
        
        with st.spinner("Running prediction model..."):
            predicted_demand, confidence = predict_demand(input_data)
            
        st.markdown(f"""
        <div class="highlight-box">
            <h3 style="color: #1E88E5; margin-top: 0;">🎯 Prediction Result</h3>
            <p style="font-size: 1.2rem; margin-bottom: 5px;">Estimated Energy Demand for <strong>{location}</strong> at <strong>{time_of_day.strftime('%H:%M')}</strong>:</p>
            <h1 style="color: #333; margin-top: 0; margin-bottom: 10px;">{predicted_demand:.2f} MW</h1>
            <p style="color: #666; margin-bottom: 0;">Model Confidence: <strong>{confidence}%</strong></p>
            <progress value="{confidence}" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
        if predicted_demand > (station_capacity * 0.9):
            st.warning("⚠️ Warning: Predicted demand is approaching or exceeding station capacity. Load balancing recommended.")
        else:
            st.success("✅ Station capacity is sufficient for predicted demand.")


# ==========================================
# PAGE 3: MODEL EVALUATION PAGE
# ==========================================
elif page == "Model Evaluation":
    st.markdown('<p class="main-header">🧪 Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Performance comparison of machine learning models</p>', unsafe_allow_html=True)
    
    eval_df = get_model_evaluation_data()
    
    st.markdown("### 📊 Metrics Comparison")
    
    # Highlight Best Model
    best_model = eval_df.loc[eval_df['R² Score'].idxmax()]['Model']
    st.success(f"🏆 Best Performing Model: **{best_model}** (Based on Highest R² Score)")
    
    # Display dataframe with styling
    st.dataframe(
        eval_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#C8E6C9')
                     .highlight_max(subset=['R² Score'], color='#C8E6C9'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### 📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error Metrics Chart
        error_df = pd.melt(eval_df, id_vars=['Model'], value_vars=['MAE', 'RMSE'], 
                           var_name='Metric', value_name='Error Value')
        fig_error = px.bar(error_df, x='Model', y='Error Value', color='Metric', barmode='group',
                           title="Error Metrics (Lower is Better)")
        st.plotly_chart(fig_error, use_container_width=True)
        
    with col2:
        # R2 Score Chart
        fig_r2 = px.bar(eval_df, x='Model', y='R² Score', color='Model',
                        title="R² Score (Higher is Better)",
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_r2.update_layout(yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)
        
    import json
    import os
    best_config_path = os.path.join("models", "best_model_config.json")
    active_model = "Artificial Neural Network (ANN)"
    if os.path.exists(best_config_path):
        try:
            with open(best_config_path, "r") as f:
                config = json.load(f)
                active_model = config.get("best_model_name", active_model)
        except:
            pass
            
    st.info(f"💡 Note: The currently active model servicing the 'Predict Demand' page is: **{active_model}**.")
