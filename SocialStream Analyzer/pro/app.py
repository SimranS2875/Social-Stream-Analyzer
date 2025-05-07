import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Social Media Sentiment Analysis", page_icon="📊", layout="wide")

# Dark theme styling
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Social Media Sentiment Analysis Dashboard")

# File uploader
st.sidebar.header("📥 Upload Your Datasets")
insta_file = st.sidebar.file_uploader("Upload Instagram Comments CSV", type="csv")
chat_file = st.sidebar.file_uploader("Upload Live Chat Sentiments CSV", type="csv")
qos_file = st.sidebar.file_uploader("Upload QoS Metrics CSV", type="csv")

if insta_file and chat_file and qos_file:
    # Load datasets
    instagram_comments = pd.read_csv(insta_file)
    live_chat_sentiments = pd.read_csv(chat_file)
    benchmark_qos_metrics = pd.read_csv(qos_file)

    # -----------------------------
    # Instagram QoS Dashboard Output
    # -----------------------------

    st.subheader("📊 Average Instagram Sentiment")
    average_sentiment = instagram_comments['sentiment_score'].mean()
    st.metric(label="Average Sentiment Score", value=round(average_sentiment, 3))

    st.subheader("📊 Satisfaction Distribution")
    satisfaction_distribution = instagram_comments['satisfaction_level'].value_counts()

    fig1, ax1 = plt.subplots()
    satisfaction_distribution.plot(kind='bar', color='limegreen', ax=ax1)
    ax1.set_title('Comment Satisfaction Levels')
    ax1.set_xlabel('Satisfaction Level')
    ax1.set_ylabel('Number of Comments')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig1)

    # -----------------------------
    # Live Chat Sentiment Distribution Output
    # -----------------------------

    st.subheader("💬 Live Chat Sentiment Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(data=live_chat_sentiments, x='sentiment', palette='Blues', ax=ax2)
    ax2.set_title('Live Chat Sentiment Distribution')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    # -----------------------------
    # Benchmark QoS Metrics Summary Output
    # -----------------------------

    st.subheader("📈 QoS Metrics Summary")
    st.dataframe(benchmark_qos_metrics.describe())

else:
    st.warning("👈 Please upload all three CSV files in the sidebar to proceed.")

