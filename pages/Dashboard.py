import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np
import tensorflow as tf

# ======= Config ======
st.set_page_config(page_title="Dashboard | T.P.", layout="wide")

# ======= Logout Utility =======
def logout():
    st.session_state.update({
        "otp_sent": False,
        "authenticated": False,
        "email": "",
        "otp": None,
        "otp_attempts": 0,
        "otp_timestamp": 0
    })
    st.switch_page("app.py")

# ======= Auth Check =======
if not st.session_state.get("authenticated", False):
    st.warning('**⚠️ :red[Unauthorized Access]**')
    st.switch_page('app.py')
    st.stop()

# ====== SIDEBAR CONTEXT ======
st.sidebar.success(f"✅ Logged in as: {st.session_state.get('email', 'Unknown')}")
st.sidebar.button("🔓 Logout", on_click=logout)

# ====== MAIN DASHBOARD ======
st.title(":green[Tensor Pay: Income Prediction & Profiling System]")
st.markdown("---")

# ======= Load Dataset =======
excel_path = 'datasets/income_census_data_with_salary.csv'  

try:
    df = pd.read_csv(excel_path)
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ======= Dataset Preview =======
with st.expander('📊 Income Trend'):
    st.subheader("📁 Income Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Simple Altair Chart: Salary vs Education
    salary_chart = alt.Chart(df).mark_boxplot().encode(
        x=alt.X("education:N", title="Education Level"),
        y=alt.Y("salary:Q", title="Salary"),
        color="education:N"
    ).properties(
        width=800,
        height=400,
        title="📈 Salary Distribution by Education Level"
    )

    st.altair_chart(salary_chart, use_container_width=True)

# ======= Prediction Section =======
with st.expander("📥 Predict Your Salary"):
    st.subheader("👤 Enter Your Profile Info")

    age = st.slider("Age", 18, 70, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp', 'Government', 'Unemployed'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Masters', 'Doctorate', 'Assoc', 'Some-college'])
    education_num = st.slider("Years of Education", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Other-service'])
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex = st.radio("Gender", ['Male', 'Female'])
    hours_per_week = st.slider("Working Hours Per Week", 1, 100, 40)

    if st.button("🎯 Predict My Salary"):
        try:
            # Load encoders and scaler
            encoders = joblib.load("models/encoders.pkl")
            scaler = joblib.load("models/scaler.pkl")
            model = tf.keras.models.load_model("models/salary_predictor.h5")  # ✅ Fixed here

            # Encode categorical features
            input_dict = {
                "age": age,
                "workclass": encoders['workclass'].transform([workclass])[0],
                "education": encoders['education'].transform([education])[0],
                "education-num": education_num,
                "marital-status": encoders['marital-status'].transform([marital_status])[0],
                "occupation": encoders['occupation'].transform([occupation])[0],
                "relationship": encoders['relationship'].transform([relationship])[0],
                "race": encoders['race'].transform([race])[0],
                "sex": encoders['sex'].transform([sex])[0],
                "hours-per-week": hours_per_week
            }

            # Prepare input
            input_array = np.array([list(input_dict.values())])
            input_scaled = scaler.transform(input_array)

            predicted_salary = model.predict(input_scaled)[0][0]  # ✅ For Keras, grab [0][0]

            st.success(f"💰 Predicted Salary: ₹ {int(predicted_salary):,} per year")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
            st.exception(e)
