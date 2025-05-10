import pandas as pd
import re
import json
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import google.generativeai as genai
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# --- Constants ---
GOOGLE_API_KEY = "AIzaSyA--Yhu0e2duoPSewMeMLCupFBR6yCaW44"  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)
N_SAMPLES = 10
CSV_OUTPUT = "comparison_results.csv"

# --- Load and clean data ---
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.replace('\n', ' ').strip().lower()

def truncate_text(text, max_length=500):
    return text[:max_length]

fake_df = pd.read_csv("data/Fake.csv")#ml_model/data
real_df = pd.read_csv("data/True.csv")
fake_df['label'] = 'fake'
real_df['label'] = 'real'
combined_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)
combined_df['cleaned_text'] = (combined_df['title'] + " " + combined_df['text']).apply(clean_text).apply(truncate_text)

X = combined_df['cleaned_text']
y = combined_df['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
test_df = pd.DataFrame({'cleaned_text': X_test, 'label': label_encoder.inverse_transform(y_test)})

# --- Train SVM model ---
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
svm_model = SVC(probability=True, kernel='linear')
svm_model.fit(X_train_vec, y_train)

# --- SVM Inference ---
def classify_with_svm(text: str) -> dict:
    cleaned = truncate_text(clean_text(text))
    vectorized = vectorizer.transform([cleaned])
    prediction = svm_model.predict(vectorized)[0]
    confidence = svm_model.predict_proba(vectorized)[0][prediction]
    label = label_encoder.inverse_transform([prediction])[0]
    return {
        "classification": label.upper(),
        "confidence": round(float(confidence), 3),
        "reasoning": "Predicted using TF-IDF + SVM."
    }

# --- ChromaDB Setup ---
chroma_client = Client(settings=Settings(persist_directory="/home/Ahamed_Shojib/fake_info/ml_model/svm_content/chroma_db", is_persistent=True))
embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
collection = chroma_client.get_collection(name="news_articles", embedding_function=embedding_func)

# --- Gemini Setup ---
model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
generation_config = {
    "temperature": 0.1,
    "response_mime_type": "application/json"
}

# --- Gemini Helper Functions ---
def retrieve_context(input_text: str, n_results: int = 5):
    results = collection.query(query_texts=[input_text], n_results=n_results)
    return [{"text": doc, "label": meta["label"].upper()} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

def build_prompt(input_text: str, retrieved_context: list):
    with open("few_shot_examples.json", "r") as f:
        few_shot_examples = json.load(f)
    examples = retrieved_context + few_shot_examples
    example_str = ""
    for i, ex in enumerate(examples[:5]):
        example_str += f"Example {i+1} ({ex['label']}): {ex['text'][:200]}...\n\n"
    return f"""
You are a fact-checking assistant. Classify the news article below as FAKE or REAL.

**Instructions**:
1. Analyze these examples:
{example_str}

2. Classify this article:
{input_text[:2000]}

Return **ONLY** a JSON object with:
- "classification" (FAKE/REAL)
- "confidence" (0.0-1.0)
- "reasoning" (1-2 sentences)
"""

def classify_news_article(input_text: str):
    retrieved = retrieve_context(input_text)
    prompt = build_prompt(input_text, retrieved)
    response = model.generate_content(prompt, generation_config=generation_config)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"classification": "ERROR", "confidence": 0.0, "reasoning": "Gemini failed to return JSON."}

# --- Run comparison ---
samples = test_df.sample(n=N_SAMPLES, random_state=42)
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "true_label", "gemini_label", "gemini_confidence",
        "svm_label", "svm_confidence", "agreement", "text_snippet"
    ])
    writer.writeheader()
    for idx, row in samples.iterrows():
        text = row["cleaned_text"]
        true_label = row["label"].upper()
        gemini = classify_news_article(text)
        svm = classify_with_svm(text)
        gemini_label = gemini.get("classification", "ERROR")
        gemini_conf = gemini.get("confidence", 0.0)
        svm_label = svm.get("classification", "ERROR")
        svm_conf = svm.get("confidence", 0.0)
        agreement = gemini_label == svm_label
        writer.writerow({
            "true_label": true_label,
            "gemini_label": gemini_label,
            "gemini_confidence": gemini_conf,
            "svm_label": svm_label,
            "svm_confidence": svm_conf,
            "agreement": agreement,
            "text_snippet": text[:200]
        })

print(f"\n✅ Results saved to {CSV_OUTPUT}")

# --- Visualize confusion matrices ---
df = pd.read_csv(CSV_OUTPUT)
valid = df[(df["gemini_label"] != "ERROR") & (df["svm_label"] != "ERROR")]
label_map = {"FAKE": 0, "REAL": 1}
y_true = valid["true_label"].map(label_map)
y_gemini = valid["gemini_label"].map(label_map)
y_svm = valid["svm_label"].map(label_map)

cm_gemini = confusion_matrix(y_true, y_gemini)
disp = ConfusionMatrixDisplay(cm_gemini, display_labels=["FAKE", "REAL"])
disp.plot()
plt.title("Gemini Classification")
plt.show()

cm_svm = confusion_matrix(y_true, y_svm)
disp = ConfusionMatrixDisplay(cm_svm, display_labels=["FAKE", "REAL"])
disp.plot()
plt.title("SVM Classification")
plt.show()
