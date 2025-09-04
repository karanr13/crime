import streamlit as st
st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ------------------ ANIMATED BACKGROUND ------------------
# ------------------ HORROR BACKGROUND CSS ------------------
import base64
import streamlit as st

# Path to your horror background image
file_path = "horror.webp"   # use raw string (r"...") to avoid \ issues

# Read and encode the image
with open(file_path, "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# CSS with background
horror_bg = f"""
<style>
.stApp {{
    background: url("data:image/webp;base64,{encoded}") no-repeat center center fixed;
    background-size: cover;
    color: #FF0000;
    font-family: 'Creepster', cursive;
    animation: flicker 3s infinite;
}}

@keyframes flicker {{
    0%   {{ filter: brightness(0.9); }}
    20%  {{ filter: brightness(1.2); }}
    40%  {{ filter: brightness(0.7); }}
    60%  {{ filter: brightness(1.3); }}
    80%  {{ filter: brightness(0.8); }}
    100% {{ filter: brightness(1); }}
}}

h1, h2, h3, h4 {{
    color: #FF0000 !important;
    text-shadow: 0 0 10px #FF0000, 0 0 20px #8B0000, 0 0 30px #FF4500;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%   {{ text-shadow: 0 0 5px #FF0000; }}
    50%  {{ text-shadow: 0 0 25px #FF0000, 0 0 50px #8B0000; }}
    100% {{ text-shadow: 0 0 5px #FF0000; }}
}}
</style>
"""

# Inject CSS
st.markdown(horror_bg, unsafe_allow_html=True)





# ------------------ LOAD DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("CrimesOnWomenData.csv")
    df.columns = [col.strip() for col in df.columns]
    crime_columns = ["Rape", "K&A", "DD", "AoW", "AoM", "DV", "WT"]

    missing_cols = [col for col in crime_columns if col not in df.columns]
    if missing_cols:
        st.error(f"⚠️ Missing columns in CSV: {missing_cols}")
        return None
    
    df_melted = df.melt(
        id_vars=["State", "Year"],
        value_vars=crime_columns,
        var_name="Crime",
        value_name="Cases"
    )
    return df_melted

df = load_data()

# ------------------ MAIN APP -----------------
if df is not None:
    st.title("🚨 Crime Analytics Dashboard")
    st.caption("✨ Unveiling State-wise Crime Trends & ML Insights")
    st.image("crimepic.gif", caption="Decoding Crime Trends in India")

    # ------------------ SIDEBAR ------------------
    st.sidebar.title("🔎 Choose Module")
    module = st.sidebar.radio("Select an analysis type:", ["EDA", "ML Classification"])

    # ------------------ EDA ------------------
    if module == "EDA":
        st.subheader("🔍 Exploratory Data Analysis")
        st.write("### 📑 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        states = df["State"].unique()
        crimes = df["Crime"].unique()
        years = df["Year"].unique()

        col1, col2, col3 = st.columns([2, 2, 1])
        state = col1.selectbox("Select State", states)
        crime = col2.selectbox("Select Crime", crimes)
        year = col3.selectbox("Select Year", years)

        filtered = df[(df["State"] == state) & (df["Crime"] == crime)]

        st.write(f"### 📈 Trend of {crime} in {state}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(filtered["Year"], filtered["Cases"], marker="o", color="red")
        ax.set_ylabel("Cases")
        ax.set_xlabel("Year")
        ax.set_title(f"{crime} Cases in {state} Over Time", fontsize=14, color="gold")
        st.pyplot(fig)

        st.write(f"### 🏆 Top 10 States for {crime} in {year}")
        top_states = df[df["Year"] == year].groupby("State")["Cases"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_states)

        st.markdown("---")
        st.write("### 🔗 Correlation Between Crime Types")
        df_pivot = df.pivot_table(index=["State", "Year"], columns="Crime", values="Cases").reset_index()
        df_corr = df_pivot.drop(columns=["State", "Year"]).corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Correlation Matrix of Crime Types")
        st.pyplot(fig_corr)

        st.markdown("---")
        st.write("### 🍩 Distribution of Crime Types")
        dist_year = st.selectbox("Select Year for Distribution", years, key="dist_year")
        dist_state = st.selectbox("Select State for Distribution", states, key="dist_state")
        df_dist = df[(df["State"] == dist_state) & (df["Year"] == dist_year)]
        fig_dist = px.pie(df_dist, values="Cases", names="Crime", 
                          title=f"Crime Distribution in {dist_state} ({dist_year})", hole=0.4,
                          color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig_dist)

    # ------------------ ML CLASSIFICATION ------------------
    elif module == "ML Classification":
        st.subheader("🤖 Machine Learning Classification")

        df_ml = df.copy()
        df_ml["Avg_Cases_State"] = df.groupby("State")["Cases"].transform("mean")
        df_ml["Yearly_Change"] = df.groupby(["State", "Crime"])["Cases"].diff().fillna(0)

        X = df_ml[["Cases", "Avg_Cases_State", "Yearly_Change"]]
        y = df_ml["Crime"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_name = st.selectbox("Select ML Model", ["Random Forest", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"])
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "K-Nearest Neighbors (KNN)":
            n_neighbors = st.slider("Select number of neighbors (k)", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_name == "Support Vector Machine (SVM)":
            kernel_type = st.selectbox("Select Kernel", ["linear", "rbf", "poly"])
            model = SVC(kernel=kernel_type, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write(f"### Results for {model_name}")
        st.write("#### 📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("#### 🔲 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        if model_name == "Random Forest":
            st.write("#### 🌟 Feature Importance")
            importance = pd.DataFrame({"Feature": X.columns,"Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            st.bar_chart(importance.set_index("Feature"))

