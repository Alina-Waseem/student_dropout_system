import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Student Dropout Risk System",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.main {
    background-color: #f5f7fb;
}

h1, h2, h3 {
    color: #1f2933;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

.metric-box {
    background: linear-gradient(135deg, #273f75, #4d73b0);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.metric-box h2 {
    color: white;
}

.footer {
    text-align: center;
    color: gray;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("dropout_model.pkl")


# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
<div class="card">
    <h1>🎓 Student Dropout Early Warning System</h1>
    <p>
    AI-powered dashboard for identifying at-risk students early
    and enabling timely academic interventions.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📂 Upload Data")
st.sidebar.info("Upload student CSV file for analysis")

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file",
    type=["csv"]
)


# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file:

    data = pd.read_csv(uploaded_file)
    data["student_id"] = data.index

    X = data.drop("Class", axis=1, errors="ignore")


    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)


    data["risk_score"] = probs
    data["predicted_dropout"] = preds


    # Risk Label
    def risk_label(p):
        if p >= 0.7:
            return "High"
        elif p >= 0.4:
            return "Medium"
        else:
            return "Low"


    data["risk_level"] = data["risk_score"].apply(risk_label)


    # -------------------------------
    # Summary Metrics
    # -------------------------------
    total = len(data)
    high = len(data[data["risk_level"] == "High"])
    medium = len(data[data["risk_level"] == "Medium"])
    low = len(data[data["risk_level"] == "Low"])


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{total}</h2>
            <p>Total Students</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{high}</h2>
            <p>High Risk</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{medium}</h2>
            <p>Medium Risk</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{low}</h2>
            <p>Low Risk</p>
        </div>
        """, unsafe_allow_html=True)


    st.write("")


    # -------------------------------
    # High Risk Table
    # -------------------------------
    st.markdown("""
    <div class="card">
        <h3>🚨 Top 20 High-Risk Students</h3>
    </div>
    """, unsafe_allow_html=True)

    top = data.sort_values(
        "risk_score",
        ascending=False
    ).head(20)

    st.dataframe(
        top,
        use_container_width=True
    )


    st.write("")


    # -------------------------------
    # Student Profile
    # -------------------------------
    st.markdown("""
    <div class="card">
        <h3>🔍 Individual Student Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    sid = st.selectbox(
        "Select Student ID",
        data["student_id"]
    )

    student = data[data["student_id"] == sid]


    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card">
            <h4>📌 Risk Summary</h4>
        """, unsafe_allow_html=True)

        st.write(student[[
            "student_id",
            "risk_score",
            "risk_level",
            "predicted_dropout"
        ]])

        st.markdown("</div>", unsafe_allow_html=True)


    # -------------------------------
    # Feature Importance
    # -------------------------------
    with colB:

        st.markdown("""
        <div class="card">
            <h4>📊 Key Risk Factors</h4>
        """, unsafe_allow_html=True)


        prep = model["prep"]
        rf = model["model"]


        num_features = prep.transformers_[0][2]
        cat_features = prep.transformers_[1][1]["encoder"].get_feature_names_out()

        features = list(num_features) + list(cat_features)


        importances = rf.feature_importances_


        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(
            "Importance",
            ascending=False
        ).head(8)


        fig, ax = plt.subplots()
        ax.barh(
            feat_df["Feature"],
            feat_df["Importance"]
        )

        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Top Predictors")

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)



# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
    <hr>
    <p>
    Developed by Alina Waseem | AI Hackathon Project  
    Student Dropout Prediction System
    </p>
</div>
""", unsafe_allow_html=True)
