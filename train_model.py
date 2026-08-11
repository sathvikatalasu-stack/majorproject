import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


# Load dataset
df = pd.read_csv("Mall_Customers.csv")


# Select features
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]


# Train K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

silhouette = silhouette_score(X, df["Cluster"])

print(f"\nSilhouette Score: {silhouette:.3f}")


# Customer segment names
segment_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Young High-Spenders",
    3: "High-Income Low-Spenders",
    4: "Low-Value Customers"
}

df["Customer Segment"] = df["Cluster"].map(segment_names)


# Recommendations
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

df["Recommendation"] = df["Customer Segment"].map(
    recommendations
)


# Save trained model
joblib.dump(kmeans, "kmeans_model.pkl")


# Save segmented dataset
df.to_csv(
    "customer_segments.csv",
    index=False
)


print("Model training completed successfully!")
print("K-Means model saved as kmeans_model.pkl")
print("Segmented dataset saved as customer_segments.csv")

print("\nCustomer Segments:")
print(df["Customer Segment"].value_counts())