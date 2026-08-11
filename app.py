import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


# -----------------------------
# Load model and dataset
# -----------------------------

model = joblib.load("kmeans_model.pkl")

df = pd.read_csv("customer_segments.csv")
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

silhouette = silhouette_score(
    X,
    df["Cluster"]
)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("👥 Customer Segmentation System")

st.write(
    "An ML-based customer segmentation system "
    "using K-Means clustering."
)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Customer Information")

income = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=10,
    max_value=150,
    value=50
)

spending = st.sidebar.slider(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)


# -----------------------------
# Prediction
# -----------------------------

input_data = pd.DataFrame(
    [[income, spending]],
    columns=[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)

cluster = model.predict(input_data)[0]


# -----------------------------
# Segment names
# -----------------------------

segment_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Young High-Spenders",
    3: "High-Income Low-Spenders",
    4: "Low-Value Customers"
}


recommendations = {
    "Average Customers":
        "Use general promotions and product recommendations.",

    "Premium Customers":
        "Offer premium products, loyalty rewards and exclusive deals.",

    "Young High-Spenders":
        "Target with trendy products, discounts and social-media campaigns.",

    "High-Income Low-Spenders":
        "Use personalized offers to increase their spending.",

    "Low-Value Customers":
        "Offer affordable products and basic promotional campaigns."
}


segment = segment_names[cluster]

recommendation = recommendations[segment]


# -----------------------------
# Prediction Section
# -----------------------------

st.header("🔍 Customer Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Annual Income",
        f"${income}k"
    )

with col2:
    st.metric(
        "Spending Score",
        spending
    )

with col3:
    st.metric(
        "Cluster",
        cluster
    )


st.subheader("Customer Segment")

st.success(segment)


st.subheader("Business Recommendation")

st.info(recommendation)


# -----------------------------
# Dataset Overview
# -----------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Number of Segments",
        df["Cluster"].nunique()
    )

with col3:
    st.metric(
        "Average Income",
        f"${df['Annual Income (k$)'].mean():.2f}k"
    )

with col4:
    st.metric(
        "Silhouette Score",
        f"{silhouette:.3f}"
    )

st.info(
    "A Silhouette Score of 0.554 indicates that the "
    "five customer segments have reasonably good separation."
)


# -----------------------------
# Customer Distribution
# -----------------------------

st.header("📈 Customer Segment Distribution")

segment_counts = df["Customer Segment"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x=segment_counts.index,
    y=segment_counts.values,
    ax=ax
)

ax.set_xlabel("Customer Segment")
ax.set_ylabel("Number of Customers")
ax.set_title("Customers in Each Segment")

plt.xticks(rotation=20)

st.pyplot(fig)


# -----------------------------
# Income vs Spending
# -----------------------------

st.header("💰 Income vs Spending")

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Customer Segment",
    data=df,
    s=100,
    ax=ax
)

ax.set_title("Customer Segmentation")

st.pyplot(fig)


# -----------------------------
# Segment Summary
# -----------------------------

st.header("📋 Segment Analysis")

summary = df.groupby("Customer Segment").agg(
    Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

st.dataframe(summary)

st.header("🎯 Customer Segment Profiles")

for segment in summary.index:

    st.subheader(segment)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            int(summary.loc[segment, "Customers"])
        )

    with col2:
        st.metric(
            "Average Age",
            f"{summary.loc[segment, 'Average_Age']:.1f}"
        )

    with col3:
        st.metric(
            "Average Income",
            f"${summary.loc[segment, 'Average_Income']:.1f}k"
        )

    with col4:
        st.metric(
            "Average Spending",
            f"{summary.loc[segment, 'Average_Spending']:.1f}"
        )

    st.info(recommendations[segment])

st.header("👫 Gender Analysis")

fig, ax = plt.subplots(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Customer Segment",
    hue="Genre",
    ax=ax
)

ax.set_title("Gender Distribution Across Customer Segments")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Number of Customers")

plt.xticks(rotation=20)

st.pyplot(fig)

st.header("🎂 Age Analysis")

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Customer Segment",
    y="Age",
    ax=ax
)

ax.set_title("Age Distribution Across Customer Segments")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Age")

plt.xticks(rotation=20)

st.pyplot(fig)

st.header("📥 Download Results")

csv_data = df.to_csv(index=False)

st.download_button(
    label="Download Customer Segmentation Data",
    data=csv_data,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)



# -----------------------------
# Customer Data
# -----------------------------

st.header("👥 Customer Dataset")

st.dataframe(df)