# majorproject
# Customer Segmentation and Business Recommendation System

## Project Overview

This project uses Machine Learning to segment customers based on their annual income and spending behavior.

K-Means Clustering is used to divide customers into different groups. The project also provides business recommendations for each customer segment.

An interactive Streamlit dashboard is developed to allow users to enter customer information and identify the corresponding customer segment.

## Objectives

- Analyze customer purchasing behavior
- Apply K-Means clustering
- Divide customers into meaningful segments
- Evaluate the clustering performance
- Provide business recommendations
- Develop an interactive web application

## Dataset

The project uses the Mall Customers dataset.

The dataset contains 200 customers with information such as:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.

In this project, customers are grouped using:

- Annual Income
- Spending Score

The model uses 5 clusters.

## Customer Segments

The project identifies five customer segments:

1. Average Customers
2. Premium Customers
3. Young High-Spenders
4. High-Income Low-Spenders
5. Low-Value Customers

## Model Evaluation

The clustering model achieved a:

**Silhouette Score: 0.554**

This indicates reasonably good separation between the customer clusters.

## Streamlit Application

The project includes an interactive Streamlit dashboard.

The dashboard provides:

- Customer prediction
- Customer segment identification
- Business recommendations
- Dataset overview
- Customer segment distribution
- Income vs spending visualization
- Gender analysis
- Age analysis
- Segment analysis
- Downloadable customer data

## Project Structure

```text
Customer_Segmentation/
│
├── app.py
├── train_model.py
├── customer_segmentation_analysis.ipynb
├── Mall_Customers.csv
├── customer_segments.csv
├── kmeans_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
