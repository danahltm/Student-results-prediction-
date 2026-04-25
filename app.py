
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Success Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>


.stApp {
    background-color: #F6F0FF;
    background-image:
        linear-gradient(#D9CCFF 1px, transparent 1px),
        linear-gradient(90deg, #EEE6FF 1px, transparent 1px);
    background-size: 100% 28px, 28px 100%;
}


html, body, [class*="css"], p, span, label, div {
    color: #2E004F !important;
    font-weight: 700 !important;
}

/* Title */
h1, h2, h3 {
    color: #2E004F !important;
    font-weight: 900 !important;
}

/* Icons */
.stApp::before {
    content: '📚 ✏️ 🎓 📖 📓 🖊️';
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-size: 44px;
    opacity: 0.35;
    color: #FFFFFF;
}

/* Card */
.card {
     background: linear-gradient(135deg, #6A1B9A, #FFFFFF);
    padding: 20px;
    border-radius: 20px;
    border: 2px solid #6A1B9A;
    box-shadow: 0 6px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    color: #FFFFFF;
}

/* Inputs */
input, textarea, select {
    color: #FFFFFF !important;
}

/* Streamlit inputs */
div[data-baseweb="input"] {
    background-color: white !important;
    border: 2px solid #6A1B9A !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #6A1B9A, #FFFFFF);
    color: white !important;
    border-radius: 15px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

model = joblib.load("logistic_model.pkl")

st.markdown("<h1>⋆˚꩜ Study Success Predictor ꩜˚⋆</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='card'>
✨ Success is built with consistency, not luck.
</div>
""", unsafe_allow_html=True)

# --- 3 INPUTS ONLY ---
study_hours = st.number_input("Study Hours", 0, 168, 20)
attendance = st.number_input("Attendance (%)", 0, 100, 75)
past_score = st.number_input("Past Score", 0, 100, 65)

# --- FIXED FEATURES ---
gender_val = 1
parent_val = 1
internet_val = 1
activities = 0.5

if st.button("Predict.✦݁˖"):

    features = np.array([[
        gender_val,
        study_hours,
        attendance,
        past_score,
        parent_val,
        internet_val,
        activities
    ]])

    proba = model.predict_proba(features)[0][1]
    is_pass = proba >= 0.5

    if is_pass:
        st.success(f"🎉 PASS! Success Rate: {round(proba*100, 1)}%")
        st.balloons()

    else:
        st.error(f"❌ FAIL - Success Rate: {round(proba*100, 1)}%")

        st.markdown("""
### 📌 Improvement Plan:
- Increase study hours 📚  
- Improve attendance 🏫  
- Review past exams 📝  
- Stay consistent daily 🔥  
""")
