import streamlit as st
import pandas as pd
import pickle

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Housing Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD MODEL & ENCODERS
# ==========================================

model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */

.stApp{
    background-color:#F8F6F0;
}


/* Header */

.main-header{
    background:linear-gradient(90deg,#234F2E,#5B4636);
    padding:30px;
    border-radius:18px;
    text-align:center;
    color:white;
    box-shadow:0px 6px 15px rgba(0,0,0,0.15);
}

.main-header h1{
    font-size:42px;
    margin-bottom:5px;
}

.main-header p{
    font-size:18px;
    color:#EFE8DA;
}


/* Cards */

.card{
    background:white;
    padding:20px;
    border-radius:18px;
    border-left:8px solid #C7A44C;
    box-shadow:0px 6px 12px rgba(0,0,0,0.08);
    margin-bottom:20px;
}


/* Section Title */

.section-title{
    color:#234F2E;
    font-size:28px;
    font-weight:bold;
    margin-bottom:10px;
}


/* Prediction Card */

.prediction{
    background:#FFF8E8;
    border:3px solid #C7A44C;
    border-radius:18px;
    padding:30px;
    text-align:center;
    margin-top:30px;
}

.prediction h1{
    color:#234F2E;
    font-size:42px;
}

.prediction h2{
    color:#5B4636;
}


/* Sidebar */

section[data-testid="stSidebar"]{
    background:#234F2E;
}

section[data-testid="stSidebar"] *{
    color:white;
}


/* Button */

div.stButton > button{
    background:#234F2E;
    color:white;
    border:none;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
    padding:15px;
    width:100%;
    transition:0.3s;
}

div.stButton > button:hover{
    background:#355E3B;
    color:white;
}


/* Radio Buttons */

div[role="radiogroup"]{
    flex-direction:row !important;
}


/* Footer */

.footer{
    text-align:center;
    color:#666;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""

<div class="main-header">

<h1>🏡 AI Housing Price Prediction</h1>

<p>
Predict residential property prices using Machine Learning Regression Algorithms
</p>

</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🏠 Housing Predictor")

    st.markdown("---")

    st.markdown("### 🤖 Regression Models")

    st.success("✔ Linear Regression")

    st.success("✔ Random Forest")

    st.success("✔ XGBoost")

    st.markdown("---")

    st.markdown("### ⭐ Selected Model")

    st.info("XGBoost Regressor")

    st.markdown("---")

    st.markdown("### 📊 Dataset")

    st.write("Housing.csv")

    st.markdown("---")

    st.markdown("### 💡 About")

    st.write(
        """
        This application predicts the estimated
        price of a house based on its
        characteristics using Machine Learning.
        """
    )

# ==========================================
# PROPERTY DETAILS TITLE
# ==========================================

st.markdown(
    '<div class="section-title">🏘 Property Details</div>',
    unsafe_allow_html=True
)

st.write("Please enter the property details below.")

st.write("")

# ==========================================
# TWO COLUMN LAYOUT
# ==========================================

left_col, right_col = st.columns(2)
# ==========================================
# LEFT COLUMN
# ==========================================

with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    area = st.number_input(
        "📐 Area (sq.ft)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=100
    )

    bedrooms = st.number_input(
        "🛏 Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "🏢 Stories",
        min_value=1,
        max_value=5,
        value=2
    )

    parking = st.number_input(
        "🚗 Parking Spaces",
        min_value=0,
        max_value=5,
        value=1
    )

    furnishingstatus = st.selectbox(
        "🛋 Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# RIGHT COLUMN
# ==========================================

with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    mainroad = st.radio(
        "🛣 Main Road",
        ["yes", "no"],
        horizontal=True
    )

    guestroom = st.radio(
        "🛏 Guest Room",
        ["yes", "no"],
        horizontal=True
    )

    basement = st.radio(
        "🏠 Basement",
        ["yes", "no"],
        horizontal=True
    )

    hotwaterheating = st.radio(
        "🔥 Hot Water Heating",
        ["yes", "no"],
        horizontal=True
    )

    airconditioning = st.radio(
        "❄ Air Conditioning",
        ["yes", "no"],
        horizontal=True
    )

    prefarea = st.radio(
        "🌳 Preferred Area",
        ["yes", "no"],
        horizontal=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# ENCODE CATEGORICAL VALUES
# ==========================================

mainroad = encoders["mainroad"].transform([mainroad])[0]

guestroom = encoders["guestroom"].transform([guestroom])[0]

basement = encoders["basement"].transform([basement])[0]

hotwaterheating = encoders["hotwaterheating"].transform(
    [hotwaterheating]
)[0]

airconditioning = encoders["airconditioning"].transform(
    [airconditioning]
)[0]

prefarea = encoders["prefarea"].transform([prefarea])[0]
furnishing_display = furnishingstatus
furnishingstatus = encoders["furnishingstatus"].transform(
    [furnishingstatus]
)[0]

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({

    "area":[area],

    "bedrooms":[bedrooms],

    "bathrooms":[bathrooms],

    "stories":[stories],

    "mainroad":[mainroad],

    "guestroom":[guestroom],

    "basement":[basement],

    "hotwaterheating":[hotwaterheating],

    "airconditioning":[airconditioning],

    "parking":[parking],

    "prefarea":[prefarea],

    "furnishingstatus":[furnishingstatus]

})

st.write("")

predict = st.button("🏡 Predict House Price")
# ==========================================
# PREDICTION
# ==========================================

if predict:

    try:

        prediction = model.predict(input_data)[0]

        st.write("")
        st.write("")

        st.markdown("## 🏠 Estimated House Price")
        st.metric(
             label="Predicted Price",
             value=f"₹ {prediction:,.2f}"
             )
        st.success("✅ Prediction Generated Successfully")

        st.write("")

        # ==========================================
        # PROPERTY SUMMARY
        # ==========================================

        st.markdown("## 📋 Property Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📐 Area", f"{area} sq.ft")
            st.metric("🛏 Bedrooms", bedrooms)
            st.metric("🚿 Bathrooms", bathrooms)

        with col2:
            st.metric("🏢 Stories", stories)
            st.metric("🚗 Parking", parking)
            st.metric("🛋 Furnishing", furnishing_display.title())

        with col3:
            st.metric(
                "🛣 Main Road",
                "Yes" if mainroad == 1 else "No"
            )

            st.metric(
                "❄ Air Conditioning",
                "Yes" if airconditioning == 1 else "No"
            )

            st.metric(
                "🌳 Preferred Area",
                "Yes" if prefarea == 1 else "No"
            )

        st.write("")

        st.success(
            "✔ The prediction was generated using the trained XGBoost Regression model."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

# ==========================================
# FOOTER
# ==========================================

st.write("")
st.write("")
st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <h4>🏡 AI Housing Price Prediction System</h4>

    Developed using <b>Python</b> • <b>Streamlit</b> •
    <b>Scikit-Learn</b> • <b>XGBoost</b>

    <br><br>

    <span style="color:#5B4636;">
    Machine Learning Regression Project
    </span>

    </div>
    """,
    unsafe_allow_html=True
)