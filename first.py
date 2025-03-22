import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit UI setup
st.title("🔍 Smart Customer Segmentation App")
st.write("Upload a CSV file, and we will automatically analyze & segment your customers!")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.write(df.head())

    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 3:
        st.error("❌ Error: The dataset must contain at least 3 numerical columns (e.g., Age, Income, Spending Score).")
    else:
        # Compute correlation with the most relevant feature
        corr_matrix = df[numeric_cols].corr()
        target_col = numeric_cols[-1]  # Assuming the last numeric column is the target (e.g., spending score)
        
        # Select the two most correlated features
        best_features = corr_matrix[target_col].drop(target_col).abs().nlargest(2).index.tolist()
        feature_x, feature_y = best_features

        # Standardize the selected features
        df_selected = df[[feature_x, feature_y]].dropna()
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

        # Select number of clusters using Elbow Method
        distortions = []
        K_range = range(2, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df_scaled)
            distortions.append(kmeans.inertia_)

        # Optimal K selection
        optimal_k = distortions.index(min(distortions)) + 2  # Best K from elbow method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(df_scaled)

        # Display clustered data
        st.write("### 📊 Clustered Data")
        st.write(df.head())

        # Scatter plot
        st.write(f"### 🎨 Scatter Plot: {feature_x} vs {feature_y}")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df["Cluster"], palette="viridis", s=100, edgecolor="black")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"Customer Segmentation ({feature_x} vs {feature_y})")
        plt.legend(title="Cluster")
        st.pyplot(plt)

        # Cluster Summary
        cluster_summary = df.groupby("Cluster")[[feature_x, feature_y]].mean()
        st.write("### 📌 Cluster Summary")
        st.write(cluster_summary)

        # Provide easy-to-understand insights
        st.write("### 📢 Insights & Conclusion")
        for cluster in range(optimal_k):
            st.write(f"#### 🏷 Cluster {cluster}:")
            st.write(f"- *Average {feature_x}*: {cluster_summary.loc[cluster, feature_x]:.2f}")
            st.write(f"- *Average {feature_y}*: {cluster_summary.loc[cluster, feature_y]:.2f}")
            st.write(f"- *Cluster Size*: {len(df[df['Cluster'] == cluster])} customers")
            
            # Simple customer behavior interpretation
            if cluster_summary.loc[cluster, feature_x] > df[feature_x].mean() and cluster_summary.loc[cluster, feature_y] > df[feature_y].mean():
                st.write("🔹 *This group consists of high spenders with high income.*")
            elif cluster_summary.loc[cluster, feature_x] < df[feature_x].mean() and cluster_summary.loc[cluster, feature_y] > df[feature_y].mean():
                st.write("🔹 *This group spends a lot despite having lower income, possibly using credit.*")
            elif cluster_summary.loc[cluster, feature_x] > df[feature_x].mean() and cluster_summary.loc[cluster, feature_y] < df[feature_y].mean():
                st.write("🔹 *This group earns well but spends conservatively.*")
            else:
                st.write("🔹 *This group consists of low earners and low spenders.*")

        st.success("✅ Customer Segmentation Completed Successfully!")

        # 🔥 Download segmented results
        st.write("### 📥 Download Segmented Data")
        csv_file = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_file,
            file_name="customer_segmentation_results.csv",
            mime="text/csv"
        )