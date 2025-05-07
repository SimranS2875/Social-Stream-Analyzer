import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
instagram_comments = pd.read_csv('instagram_comments.csv')
live_chat_sentiments = pd.read_csv('live_chat_sentiments.csv')
benchmark_qos_metrics = pd.read_csv('benchmark_qos_metrics.csv')

# -----------------------------
# Instagram QoS Dashboard Output
# -----------------------------

# Calculate average sentiment
average_sentiment = instagram_comments['sentiment_score'].mean()
print("Average Sentiment:", round(average_sentiment, 3))

# Satisfaction Distribution count
satisfaction_distribution = instagram_comments['satisfaction_level'].value_counts()
print("\nSatisfaction Distribution:")
print(satisfaction_distribution)

# Plot Satisfaction Distribution
plt.figure(figsize=(8, 5))
satisfaction_distribution.plot(kind='bar', color='limegreen')
plt.title('Comment Satisfaction Levels')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Comments')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# Live Chat Sentiment Distribution Output
# -----------------------------

plt.figure(figsize=(7, 5))
sns.countplot(data=live_chat_sentiments, x='sentiment', palette='Blues')
plt.title('Live Chat Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# Benchmark QoS Metrics Summary Output
# -----------------------------

print("\nQoS Benchmark Summary:")
print(benchmark_qos_metrics.describe())
