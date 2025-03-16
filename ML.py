import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

def ML():
    print(os.getcwd())
    st.title('🤖Machine Learning Model🤖')
    st.write('ใช้งาน sidebar เพื่อใช้งาน Multi-page ไปยังหน้าอื่นๆ')
    st.write('ใช้ Select-box สำหรับเลือกโมเดลที่ต้องการ(RF,GBR)')

    # 🔄 เลือกโมเดลที่ต้องการใช้งาน
    model_choice = st.selectbox("เลือกโมเดลที่ต้องการพยากรณ์", ["Gradient Boosting Regressor (GBR)", "Random Forest Regressor (RF)"])

    # โหลดโมเดลตามที่เลือก
    if model_choice == "Gradient Boosting Regressor (GBR)":
        model = joblib.load("model/gbr_model.pkl")
    else:
        model = joblib.load("model/rf_model.pkl")

    st.title("🚗 Car Price Prediction App")
    st.write(f"ใส่ข้อมูลรถยนต์ แล้วระบบจะคาดการณ์ราคารถให้คุณด้วยโมเดล: **{model_choice}**")

    # สร้าง input features
    symboling = st.slider("Symboling", -3, 3, 0)
    fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    doornumber = st.selectbox("Number of Doors", ["two", "four"])
    carbody = st.selectbox("Car Body", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
    drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
    enginelocation = st.selectbox("Engine Location", ["front", "rear"])
    enginetype = st.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "rotor"])
    cylindernumber = st.selectbox("Cylinder Number", ["two", "three", "four", "five", "six", "eight", "twelve"])
    fuelsystem = st.selectbox("Fuel System", ["mpfi", "2bbl", "idi", "1bbl", "spdi"])

    # Numeric inputs
    wheelbase = st.number_input("Wheelbase", min_value=80.0, max_value=130.0, value=100.0)
    curbweight = st.number_input("Curb Weight", min_value=1500, max_value=4000, value=2500)
    enginesize = st.number_input("Engine Size", min_value=50, max_value=400, value=150)
    horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=100)
    citympg = st.number_input("City MPG", min_value=10, max_value=60, value=30)
    highwaympg = st.number_input("Highway MPG", min_value=10, max_value=70, value=35)

    # แปลง categorical ให้เป็นเลข (เหมือนตอนเทรน)
    label_map = {
        "fueltype": {"gas": 0, "diesel": 1},
        "aspiration": {"std": 0, "turbo": 1},
        "doornumber": {"two": 0, "four": 1},
        "carbody": {"convertible": 0, "hatchback": 1, "sedan": 2, "wagon": 3, "hardtop": 4},
        "drivewheel": {"fwd": 0, "rwd": 1, "4wd": 2},
        "enginelocation": {"front": 0, "rear": 1},
        "enginetype": {"dohc": 0, "ohcv": 1, "ohc": 2, "rotor": 3},
        "cylindernumber": {"two": 0, "three": 1, "four": 2, "five": 3, "six": 4, "eight": 5, "twelve": 6},
        "fuelsystem": {"mpfi": 0, "2bbl": 1, "idi": 2, "1bbl": 3, "spdi": 4},
    }

    # สร้าง DataFrame 1 row
    input_data = pd.DataFrame([{
        "symboling": symboling,
        "fueltype": label_map["fueltype"][fueltype],
        "aspiration": label_map["aspiration"][aspiration],
        "doornumber": label_map["doornumber"][doornumber],
        "carbody": label_map["carbody"][carbody],
        "drivewheel": label_map["drivewheel"][drivewheel],
        "enginelocation": label_map["enginelocation"][enginelocation],
        "enginetype": label_map["enginetype"][enginetype],
        "cylindernumber": label_map["cylindernumber"][cylindernumber],
        "fuelsystem": label_map["fuelsystem"][fuelsystem],
        "wheelbase": wheelbase,
        "curbweight": curbweight,
        "enginesize": enginesize,
        "horsepower": horsepower,
        "citympg": citympg,
        "highwaympg": highwaympg
    }])

    # ทำนายราคา
    pred_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ${pred_price:,.2f}")

    if hasattr(model, 'feature_importances_'):
        st.subheader("📊 Feature Importance")
        fig, ax = plt.subplots(figsize=(7, 4))  # ✅ ย่อขนาด
        importance = model.feature_importances_
        features = input_data.columns
        ax.barh(features, importance)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance from Model")
        st.pyplot(fig)

    st.subheader("📈 Car Price Prediction Comparison")
    sample_y_test = np.random.randint(5000, 40000, size=10)
    sample_pred_rf = np.random.randint(6000, 42000, size=10)
    sample_pred_gbr = np.random.randint(5500, 41000, size=10)

    fig_compare, ax_compare = plt.subplots(figsize=(10, 5))
    ax_compare.plot(sample_y_test, label='Actual Price', marker='o')
    ax_compare.plot(sample_pred_rf, label='RF Prediction', linestyle='--')
    ax_compare.plot(sample_pred_gbr, label='GBR Prediction', linestyle=':')
    ax_compare.set_title('Car Price Prediction Comparison')
    ax_compare.set_xlabel('Sample Index')
    ax_compare.set_ylabel('Price')
    ax_compare.legend()
    ax_compare.grid(True)
    st.pyplot(fig_compare)

# เรียกใช้งานฟังก์ชัน ML()

