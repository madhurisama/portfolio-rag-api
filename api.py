import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

llm = genai.GenerativeModel('gemini-3.8-flash')

# ---- Auto-detect an embedding model this API key can use ----
def pick_embedding_model():
    names = [
        m.name for m in genai.list_models()
        if 'embedContent' in m.supported_generation_methods
    ]
    print("Available embedding models:", names)
    for preferred in ("text-embedding", "embedding"):
        for n in names:
            if preferred in n:
                return n
    return names[0]

EMBED_MODEL = pick_embedding_model()
print("Using embedding model:", EMBED_MODEL)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Knowledge base: full portfolio ----
documents = [
    # Projects
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
    "Madhuri built a Smart Event Invitation System as a personal project: a responsive multi-page web app with real-time form validation, dynamic UI updates, and zero-reload RSVP flow. Tools: HTML5, CSS3, JavaScript, Git.",

    # About
    "Madhuri is a B.Tech Data Science student at Siddhartha Institute of Engineering and Technology, Hyderabad, with a CGPA of 9.38. She has hands-on experience in Python, SQL, data analysis, and frontend web development. She is currently interning at both Oasis Infobyte and Prodigy InfoTech, and has completed 14+ real-world data projects covering EDA, machine learning, NLP, and data visualization.",

    # Skills
    "Madhuri's programming skills include Python, Pandas, NumPy, Matplotlib, Java, and C.",
    "Madhuri's web technology skills include HTML5, CSS3, JavaScript ES6+, and Responsive Design.",
    "Madhuri's data and analytics skills include SQL, EDA, Data Visualization, and MS Excel.",
    "Madhuri's machine learning skills include Scikit-learn, Decision Trees, Random Forest, and NLP.",
    "Madhuri's developer tools include Git & GitHub, VS Code, Jupyter, Google Colab, and Adobe Tools.",
    "Madhuri's other skills include DBMS, Data Structures and Algorithms (DSA), Problem Solving, and Analytical Thinking.",

    # Experience
    "Madhuri worked as a Data Science Intern at Prodigy InfoTech from May 2026 to June 2026, completing 5 data science projects covering population visualization, Titanic EDA, Decision Tree classification, Twitter sentiment analysis using NLP, and US traffic accident pattern analysis.",
    "Madhuri worked as a Data Analytics Intern at Oasis Infobyte from May 2026 to June 2026, completing all 9 projects across Level 1 and Level 2, including data cleaning, retail sales EDA, customer segmentation, sentiment analysis, house price prediction, wine quality prediction, fraud detection, Google Play Store analysis, and an NLP autocomplete system. She is eligible for a Letter of Recommendation (LOR).",
    "Madhuri completed a Data Analytics Job Simulation with Deloitte via Forage in March 2026, covering real business problem solving, data interpretation, and professional analytics workflows.",

    # Education
    "Madhuri is pursuing a B.Tech in Data Science at Siddhartha Institute of Engineering and Technology, Ibrahimpatnam, affiliated with JNTUH and accredited by NBA & NAAC, expected to graduate in 2028, with a CGPA of 9.38/10.",
    "Madhuri completed her Intermediate (Class XII) with MPC (Mathematics, Physics, Chemistry) at Sri Chaitanya Junior College under TSBIE in 2024, scoring 92.1% (921/1000).",
    "Madhuri completed her Secondary School Certificate (Class X) at St. Mark's High School under the SSC Board, Telangana, in 2022, with a GPA of 9.7.",

    # Certifications & achievements
    "Madhuri holds a certification: Foundation Course in Python, Data Analytics, DBMS, and DSA through Edunet Foundation / SAP Code Unnati, 2025-2026.",
    "Madhuri holds a certification: Design Fundamentals with AI by Adobe x UNICEF (YuWaah), December 2025, scoring 100%.",
    "Madhuri has an achievement: Internal Hackathon for Smart India Hackathon 2024 at SIET (MHRD/AICTE), September 2024.",
    "Madhuri has an achievement: 1st Prize in Poster Presentation at Eminence 2026, Sri Indu College of Engineering, February 2026.",
    "Madhuri has an achievement: App Expo participation at Tech Samprathi 2026, NNRG Institutions, February 2026.",

    # Languages
    "Madhuri is professionally proficient in English, a native speaker of Telugu, and conversational in Hindi.",

    # Contact
    "Madhuri is open to internship opportunities, project collaborations, and full-time roles in Data Analytics and Frontend Development. She can be reached by email at madhurisama89@gmail.com, on LinkedIn at linkedin.com/in/madhuri-sama-3518bb324, or on GitHub at github.com/madhurisama. She is based in Hyderabad, Telangana, India."
]

# ---- Embeddings (computed by Google's servers, so no heavy local model) ----
def embed_documents(texts):
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return np.array(result['embedding'])

def embed_query(text):
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return np.array(result['embedding'])

# Embed all documents once at startup
doc_embeddings = embed_documents(documents)
doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

def retrieve(query, top_k=3, margin=0.08):
    q = embed_query(query)
    q = q / np.linalg.norm(q)
    scores = doc_norms @ q
    top_indices = np.argsort(scores)[::-1][:top_k]
    best = scores[top_indices[0]]

    for i in top_indices:
        print(f"  [{scores[i]:.3f}] {documents[i][:60]}...")

    # Keep matches that are close to the best score (scale-independent)
    return [documents[i] for i in top_indices if scores[i] >= best - margin]

def generate_answer(query):
    try:
        context_chunks = retrieve(query)
        context = "\n".join(context_chunks)

        prompt = f"""You are Shiro, a friendly assistant on Madhuri's portfolio website. Answer the question using ONLY the context below. Be specific: mention exact names, dates, and details. If several relevant items exist, list all of them. If the context does not contain the answer, reply exactly: "I don't have information about that in Madhuri's portfolio."

Context:
{context}

Question: {query}

Answer:"""

        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong on my side. Please try again in a moment."

# ---- API endpoints ----
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    return {"answer": generate_answer(query.question)}

@app.get("/")
def root():
    return {"status": "RAG API is running"}
