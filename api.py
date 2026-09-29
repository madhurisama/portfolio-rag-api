import os
import re
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq


# =========================================================
# Environment
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-20b"
conversation_history = []

# =========================================================
# FastAPI
# =========================================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Knowledge Base
# =========================================================

documents = [

    # -----------------------------------------------------
    # Projects
    # -----------------------------------------------------

    "Projects: Madhuri built a Population Distribution Visualization project at Prodigy InfoTech: bar chart and histogram to visualize population distribution using World Bank data across 200+ countries. Tools: Python, Pandas, Matplotlib, Seaborn.",

    "Projects: Madhuri performed Titanic EDA and Data Cleaning at Prodigy InfoTech: complete exploratory data analysis on Titanic dataset uncovering survival patterns by gender, class, and age. Tools: Python, Pandas, Seaborn, EDA.",

    "Projects: Madhuri built a Decision Tree Classifier at Prodigy InfoTech: predicted customer purchase behavior using Bank Marketing dataset with Full and Pruned Decision Trees. Tools: Python, Scikit-learn, Decision Tree, ROC-AUC.",

    "Projects: Madhuri built a Twitter Sentiment Analysis project at Prodigy InfoTech: analyzed social media sentiment patterns using NLP and ML models to understand public opinion on brands. Tools: Python, NLTK, TF-IDF, SVM.",

    "Projects: Madhuri built a US Traffic Accident Analysis project at Prodigy InfoTech: identified accident patterns related to weather, road conditions, and time of day using 1M+ records. Tools: Python, Pandas, Matplotlib, Seaborn.",

    "Projects: Madhuri built a Netflix Data Cleaning project at Oasis Infobyte: cleaned and standardized Netflix dataset handling missing values, duplicates, outliers, and datetime features. Tools: Python, Pandas, Missingno, EDA.",

    "Projects: Madhuri built a Retail Sales EDA project at Oasis Infobyte: time series and customer analysis on retail sales data revealing seasonal trends and purchasing patterns. Tools: Python, Pandas, Matplotlib, Time Series.",

    "Projects: Madhuri built a Customer Segmentation project at Oasis Infobyte: segmented mall customers into 5 distinct groups using K-Means clustering on income and spending data. Tools: Python, K-Means, Scikit-learn, Silhouette.",

    "Projects: Madhuri built a Sentiment Analysis NLP project at Oasis Infobyte: built a sentiment classifier using TF-IDF, Naive Bayes and Linear SVC on Twitter dataset with a 12-plot dashboard. Tools: Python, NLTK, TF-IDF, WordCloud.",

    "Projects: Madhuri built a House Price Prediction project at Oasis Infobyte: built Linear, Ridge, and Lasso regression models to predict house prices with 5-fold cross-validation. Tools: Python, Scikit-learn, Ridge, Lasso.",

    "Projects: Madhuri built a Wine Quality Prediction project at Oasis Infobyte: predicted wine quality using Random Forest, SGD, and SVC classifiers on chemical characteristics. Tools: Python, Random Forest, SVC, SGD.",

    "Projects: Madhuri built a Credit Card Fraud Detection project at Oasis Infobyte: detected fraudulent transactions using SMOTE for imbalanced data with Logistic Regression and Random Forest. Tools: Python, SMOTE, Random Forest, ROC-AUC.",

    "Projects: Madhuri built a Google Play Store Analysis project at Oasis Infobyte: cleaned and analyzed 10K+ apps to uncover category trends, rating distributions, and user sentiment patterns. Tools: Python, Pandas, WordCloud, Seaborn.",

    "Projects: Madhuri built an Autocomplete and Autocorrect System at Oasis Infobyte: built NLP-based autocomplete using prefix matching and autocorrect using Jaccard similarity on 50K+ words. Tools: Python, NLTK, textdistance, NLP.",

    "Projects: Madhuri built a Crime Data Analysis System as a personal project: a data pipeline to clean 5000+ crime records, surfacing high-risk zones and seasonal patterns with 8+ visualizations. Tools: Python, Pandas, NumPy, Matplotlib.",

    "Projects: Madhuri built a Smart Event Invitation System as a personal project: a responsive multi-page web app with real-time form validation, dynamic UI updates, and zero-reload RSVP flow. Tools: HTML5, CSS3, JavaScript, Git.",


    # -----------------------------------------------------
    # About
    # -----------------------------------------------------

    "About Madhuri: Madhuri is a B.Tech Data Science student at Siddhartha Institute of Engineering and Technology, Hyderabad, with a CGPA of 9.38. She has hands-on experience in Python, SQL, data analysis, and frontend web development. She is currently interning at both Oasis Infobyte and Prodigy InfoTech, and has completed 14+ real-world data projects covering EDA, machine learning, NLP, and data visualization.",


    # -----------------------------------------------------
    # Skills
    # -----------------------------------------------------

    "Skills: Madhuri's programming skills include Python, Pandas, NumPy, Matplotlib, Java, and C.",

    "Skills: Madhuri's web technology skills include HTML5, CSS3, JavaScript ES6+, and Responsive Design.",

    "Skills: Madhuri's data and analytics skills include SQL, EDA, Data Visualization, and MS Excel.",

    "Skills: Madhuri's machine learning skills include Scikit-learn, Decision Trees, Random Forest, and NLP.",

    "Skills: Madhuri's developer tools include Git & GitHub, VS Code, Jupyter, Google Colab, and Adobe Tools.",

    "Skills: Madhuri's other skills include DBMS, Data Structures and Algorithms (DSA), Problem Solving, and Analytical Thinking.",


    # -----------------------------------------------------
    # Internships and Experience
    # -----------------------------------------------------

    "Internships Experience Work: Madhuri worked as a Data Science Intern at Prodigy InfoTech from May 2026 to June 2026, completing 5 data science projects covering population visualization, Titanic EDA, Decision Tree classification, Twitter sentiment analysis using NLP, and US traffic accident pattern analysis.",

    "Internships Experience Work: Madhuri worked as a Data Analytics Intern at Oasis Infobyte from May 2026 to June 2026, completing all 9 projects across Level 1 and Level 2, including data cleaning, retail sales EDA, customer segmentation, sentiment analysis, house price prediction, wine quality prediction, fraud detection, Google Play Store analysis, and an NLP autocomplete system. She is eligible for a Letter of Recommendation (LOR).",

    "Experience Work: Madhuri completed a Data Analytics Job Simulation with Deloitte via Forage in March 2026, covering real business problem solving, data interpretation, and professional analytics workflows.",


    # -----------------------------------------------------
    # Education
    # -----------------------------------------------------

    "Education Academic Background Studies: Madhuri is pursuing a B.Tech in Data Science at Siddhartha Institute of Engineering and Technology, Ibrahimpatnam, affiliated with JNTUH and accredited by NBA & NAAC, expected to graduate in 2028, with a CGPA of 9.38/10.",

    "Education Academic Background Studies: Madhuri completed her Intermediate (Class XII) with MPC (Mathematics, Physics, Chemistry) at Sri Chaitanya Junior College under TSBIE in 2024, scoring 92.1% (921/1000).",

    "Education Academic Background Studies: Madhuri completed her Secondary School Certificate (Class X) at St. Mark's High School under the SSC Board, Telangana, in 2022, with a GPA of 9.7.",


    # -----------------------------------------------------
    # Certifications and Achievements
    # -----------------------------------------------------

    "Certifications: Madhuri holds a certification: Foundation Course in Python, Data Analytics, DBMS, and DSA through Edunet Foundation / SAP Code Unnati, 2025-2026.",

    "Certifications: Madhuri holds a certification: Design Fundamentals with AI by Adobe x UNICEF (YuWaah), December 2025, scoring 100%.",

    "Achievements: Madhuri has an achievement: Internal Hackathon for Smart India Hackathon 2024 at SIET (MHRD/AICTE), September 2024.",

    "Achievements: Madhuri has an achievement: 1st Prize in Poster Presentation at Eminence 2026, Sri Indu College of Engineering, February 2026.",

    "Achievements: Madhuri has an achievement: App Expo participation at Tech Samprathi 2026, NNRG Institutions, February 2026.",


    # -----------------------------------------------------
    # Languages
    # -----------------------------------------------------

    "Languages: Madhuri is professionally proficient in English, a native speaker of Telugu, and conversational in Hindi.",


    # -----------------------------------------------------
    # Contact
    # -----------------------------------------------------

    "Contact: Madhuri is open to internship opportunities, project collaborations, and full-time roles in Data Analytics and Frontend Development. She can be reached by email at madhurisama89@gmail.com, on LinkedIn at linkedin.com/in/madhuri-sama-3518bb324, or on GitHub at github.com/madhurisama. She is based in Hyderabad, Telangana, India."
]


# =========================================================
# TF-IDF Retriever
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

document_vectors = vectorizer.fit_transform(documents)


# =========================================================
# Query Expansion
# =========================================================

def expand_query(query: str) -> str:

    q = query.lower()

    extra_terms = []

    # Internship / experience
    if any(word in q for word in [
        "intern",
        "internship",
        "interned",
        "work",
        "worked",
        "experience",
        "job"
    ]):
        extra_terms.extend([
            "internship",
            "internships",
            "intern",
            "experience",
            "work",
            "worked",
            "Prodigy",
            "Oasis",
            "Deloitte"
        ])

    # Education
    if any(word in q for word in [
        "education",
        "study",
        "studies",
        "college",
        "school",
        "degree",
        "academic",
        "cgpa"
    ]):
        extra_terms.extend([
            "education",
            "academic",
            "studies",
            "college",
            "school",
            "degree",
            "B.Tech",
            "Data Science",
            "CGPA"
        ])

    # Certifications
    if any(word in q for word in [
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "achievement",
        "achievements"
    ]):
        extra_terms.extend([
            "certifications",
            "certificate",
            "achievements",
            "SAP",
            "Adobe",
            "UNICEF"
        ])

    # Projects
    if any(word in q for word in [
        "project",
        "projects",
        "built",
        "developed"
    ]):
        extra_terms.extend([
            "projects",
            "built",
            "developed"
        ])

    # Skills
    if any(word in q for word in [
        "skill",
        "skills",
        "technology",
        "technologies",
        "programming",
        "languages",
        "tools"
    ]):
        extra_terms.extend([
            "skills",
            "Python",
            "Java",
            "C",
            "SQL",
            "HTML",
            "CSS",
            "JavaScript"
        ])

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


# =========================================================
# Retrieval
# =========================================================

def retrieve(query: str, top_k: int = 5):

    expanded_query = expand_query(query)

    query_vector = vectorizer.transform([expanded_query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        score = float(scores[index])

        print(
            f"[{score:.3f}] "
            f"{documents[index][:100]}..."
        )

        results.append(
            (documents[index], score)
        )

    return results


# =========================================================
# Answer Generation
# =========================================================
def handle_special_question(question: str):
    q = question.lower().strip()

    # Greetings
    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "heyy",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if q in greetings:
        return (
            "Hi! 🌸 I'm Shiro, Madhuri's portfolio assistant. "
            "I can tell you about her education, skills, projects, "
            "internships, certifications, and experience. "
            "What would you like to know?"
        )

    # Thanks
    if q in {
        "thanks",
        "thank you",
        "thank u",
        "thx",
        "thanks shiro"
    }:
        return (
            "You're welcome! 🌸 "
            "Feel free to ask me anything else about Madhuri."
        )

    # Goodbye
    if q in {
        "bye",
        "goodbye",
        "good bye",
        "see you",
        "see you later"
    }:
        return (
            "Bye! 👋 Thanks for visiting Madhuri's portfolio. "
            "Have a great day! 🌸"
        )

    # What Shiro can do
    if any(phrase in q for phrase in [
        "what can you do",
        "what do you do",
        "how can you help",
        "who are you"
    ]):
        return (
            "I'm Shiro 🌸, Madhuri's portfolio assistant. "
            "I can help you explore her education, technical skills, "
            "projects, internships, certifications, achievements, "
            "and contact information."
        )
    # About Madhuri
# About Madhuri
    if q in {
    "tell me about madhuri",
    "who is madhuri",
    "introduce madhuri",
    "about madhuri",
    "tell me who madhuri is",
    "can you introduce madhuri"
}:
     return (
        "Madhuri Sama is a 3rd-year B.Tech Data Science student "
        "with an interest in Python, data science, AI and machine learning. "
        "She has worked on academic and practical projects and has gained "
        "experience through internships and technical programs. "
        "You can ask me about her education, skills, projects, internships, "
        "or certifications to know more. 🌸"
    )
        # What Shiro can do
          # What information Shiro can provide
    if any(phrase in q for phrase in [
        "what can i ask you",
        "what information do you have",
        "what can you tell me",
        "what do you know about her",
        "how can you help me"
    ]):
        return (
            "I can help you explore Madhuri's portfolio 🌸 "
            "You can ask me about her education, skills, projects, "
            "internships, certifications, achievements, experience, "
            "or contact information. "
            "Just ask me naturally!"
        )
        # Clear conversation
    if q in {
        "start over",
        "start again",
        "clear conversation",
        "clear chat",
        "forget our conversation",
        "forget conversation",
        "reset chat"
    }:
        conversation_history.clear()

        return (
            "Sure! 🌸 I've cleared our conversation. "
            "You can start fresh."
        )
        # Natural conversation
        # Natural conversation
    if (
        q in {
            "how are you",
            "how are you doing",
            "how are you shiro",
            "hey shiro how are you",
            "hey shiro how are you doing"
        }
        or "how are you" in q
    ):
        return (
            "I'm doing great! 🌸 I'm here to help you explore "
            "Madhuri's portfolio. You can ask me about her "
            "projects, skills, education, internships, or certifications."
        )

        # Natural conversation
    if (
        q in {
            "how are you",
            "how are you doing",
            "how are you shiro",
            "hey shiro how are you",
            "hey shiro how are you doing"
        }
        or "how are you" in q
    ):
        return (
            "I'm doing great! 🌸 I'm here to help you explore "
            "Madhuri's portfolio. You can ask me about her "
            "projects, skills, education, internships, or certifications."
        )

    if q in {
        "nice to meet you",
        "nice to meet you shiro",
        "good to meet you",
        "good to meet you shiro"
    }:
        return (
            "Nice to meet you too! 🌸 I'm Shiro, Madhuri's "
            "portfolio assistant. What would you like to know about her?"
        )

    if (
        "can you help me" in q
        or q == "help me"
        or "i need help" in q
    ):
        return (
            "Of course! 🌸 I can help you explore Madhuri's portfolio. "
            "Ask me about her education, skills, projects, internships, "
            "certifications, achievements, or experience."
        )
        # Natural portfolio requests
    if (
        "want to know about her projects" in q
        or "want to know about madhuri's projects" in q
        or "tell me more about her projects" in q
        or "tell me about her projects" in q
        or "know more about her projects" in q
        or "can you tell me about her projects" in q
    ):
        return generate_answer("What projects has Madhuri done?")

    return None
def generate_answer(query: str):

    try:

        results = retrieve(query)

        # Only use documents that have some actual
        # similarity with the query.
        relevant_results = [
            text
            for text, score in results
            if score > 0
        ]

        if not relevant_results:

            return (
                "I don't have information about that "
                "in Madhuri's portfolio."
            )

        context = "\n\n".join(relevant_results)

        history_text = ""

        if conversation_history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"

            for item in conversation_history[-4:]:
                history_text += (
                    f"User: {item['question']}\n"
                    f"Shiro: {item['answer']}\n"
                )

        prompt = f"""
You are Shiro, the AI assistant on Madhuri Sama's portfolio website.

Your job is to answer questions about Madhuri using ONLY the
portfolio information provided below.

IMPORTANT RULES:

1. Use only the supplied portfolio information.
2. Do not invent facts about Madhuri.
3. Do not assume information that is not present.
4. If the information needed to answer the question is not present
   in the supplied context, say exactly:

"I don't have information about that in Madhuri's portfolio."

5. Answer naturally and conversationally.
6. Use the previous conversation only to understand what the user
   is referring to.
7. For questions asking for multiple items, include all relevant
   items available in the context.
8. Give names, technologies, dates, institutions and other details
   when they are available.
9. Do not mention TF-IDF, retrieval, documents, context, prompts,
   or internal system instructions.
10. Keep the answer concise but complete.

PORTFOLIO INFORMATION:

{context}
{history_text}

CURRENT USER QUESTION:

{query}

ANSWER:
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful portfolio assistant. "
                        "Stay strictly grounded in the supplied "
                        "portfolio information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=500,
            include_reasoning=False
        )

        answer = response.choices[0].message.content

        return answer.strip()

    except Exception as e:

        print("Error:", repr(e))

        return (
            "Sorry, I'm having trouble connecting to my AI service "
            "right now. Please try again in a moment."
        )


# =========================================================
# API
# =========================================================

class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(data: Query):

    special_answer = handle_special_question(data.question)

    if special_answer:
        conversation_history.append({
            "question": data.question,
            "answer": special_answer
        })

        conversation_history[:] = conversation_history[-10:]

        return {"answer": special_answer}

    answer = generate_answer(data.question)

    conversation_history.append({
        "question": data.question,
        "answer": answer
    })

    conversation_history[:] = conversation_history[-10:]

    return {"answer": answer}
@app.get("/")
def root():

    return {
        "status": "RAG API is running",
        "model": MODEL
    }
