from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base — all 16 real portfolio projects, self-contained chunks
documents = [
    "Madhuri built a Population Distribution Visualization project at Prodigy InfoTech: bar chart and histogram to visualize population distribution using World Bank data across 200+ countries. Tools: Python, Pandas, Matplotlib, Seaborn.",
    "Madhuri performed Titanic EDA and Data Cleaning at Prodigy InfoTech: complete exploratory data analysis on Titanic dataset uncovering survival patterns by gender, class, and age. Tools: Python, Pandas, Seaborn, EDA.",
    "Madhuri built a Decision Tree Classifier at Prodigy InfoTech: predicted customer purchase behavior using Bank Marketing dataset with Full and Pruned Decision Trees. Tools: Python, Scikit-learn, Decision Tree, ROC-AUC.",
    "Madhuri built a Twitter Sentiment Analysis project at Prodigy InfoTech: analyzed social media sentiment patterns using NLP and ML models to understand public opinion on brands. Tools: Python, NLTK, TF-IDF, SVM.",
    "Madhuri built a US Traffic Accident Analysis project at Prodigy InfoTech: identified accident patterns related to weather, road conditions, and time of day using 1M+ records. Tools: Python, Pandas, Matplotlib, Seaborn.",
    "Madhuri built a Netflix Data Cleaning project at Oasis Infobyte: cleaned and standardized Netflix dataset handling missing values, duplicates, outliers, and datetime features. Tools: Python, Pandas, Missingno, EDA.",
    "Madhuri built a Retail Sales EDA project at Oasis Infobyte: time series and customer analysis on retail sales data revealing seasonal trends and purchasing patterns. Tools: Python, Pandas, Matplotlib, Time Series.",
    "Madhuri built a Customer Segmentation project at Oasis Infobyte: segmented mall customers into 5 distinct groups using K-Means clustering on income and spending data. Tools: Python, K-Means, Scikit-learn, Silhouette.",
    "Madhuri built a Sentiment Analysis NLP project at Oasis Infobyte: built a sentiment classifier using TF-IDF, Naive Bayes and Linear SVC on Twitter dataset with a 12-plot dashboard. Tools: Python, NLTK, TF-IDF, WordCloud.",
    "Madhuri built a House Price Prediction project at Oasis Infobyte: built Linear, Ridge, and Lasso regression models to predict house prices with 5-fold cross-validation. Tools: Python, Scikit-learn, Ridge, Lasso.",
    "Madhuri built a Wine Quality Prediction project at Oasis Infobyte: predicted wine quality using Random Forest, SGD, and SVC classifiers on chemical characteristics. Tools: Python, Random Forest, SVC, SGD.",
    "Madhuri built a Credit Card Fraud Detection project at Oasis Infobyte: detected fraudulent transactions using SMOTE for imbalanced data with Logistic Regression and Random Forest. Tools: Python, SMOTE, Random Forest, ROC-AUC.",
    "Madhuri built a Google Play Store Analysis project at Oasis Infobyte: cleaned and analyzed 10K+ apps to uncover category trends, rating distributions, and user sentiment patterns. Tools: Python, Pandas, WordCloud, Seaborn.",
    "Madhuri built an Autocomplete and Autocorrect System at Oasis Infobyte: built NLP-based autocomplete using prefix matching and autocorrect using Jaccard similarity on 50K+ words. Tools: Python, NLTK, textdistance, NLP.",
    "Madhuri built a Crime Data Analysis System as a personal project: a data pipeline to clean 5000+ crime records, surfacing high-risk zones and seasonal patterns with 8+ visualizations. Tools: Python, Pandas, NumPy, Matplotlib.",
    "Madhuri built a Smart Event Invitation System as a personal project: a responsive multi-page web app with real-time form validation, dynamic UI updates, and zero-reload RSVP flow. Tools: HTML5, CSS3, JavaScript, Git."
]

doc_embeddings = model.encode(documents)


def retrieve(query, top_k=1, threshold=0.25):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    for i in top_indices:
        print(f"  [{similarities[i]:.3f}] {documents[i][:60]}...")

    results = [documents[i] for i in top_indices if similarities[i] >= threshold]
    return results

def generate_answer(query):
    context_chunks = retrieve(query, top_k=1)   # ← changed from top_k=2
    
    
    if not context_chunks:
        return "I don't have information about that in her portfolio."
    
    context = "\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below. Be concise and specific.

Context:
{context}

Question: {query}

Answer:"""
    
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    return response['message']['content']

# Test queries
print(generate_answer("What clustering project has she done?"))
print()
print(generate_answer("Has she worked with regression models?"))
print()
print(generate_answer("What frontend or web projects has she built?"))
