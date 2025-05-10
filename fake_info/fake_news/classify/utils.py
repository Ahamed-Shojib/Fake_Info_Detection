import os
import json
import re
import google.generativeai as genai
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# Load environment API key securely
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini Model Configuration
model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 0.1,
    "response_mime_type": "application/json",
}

# ChromaDB Client Setup
chroma_client = Client(
    settings=Settings(persist_directory="/home/Ahamed_Shojib/fake_info/ml_model/content/chroma_db", is_persistent=True)
)
collection = chroma_client.get_or_create_collection(
    name="news_articles",
    embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
)

# Text processing functions
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace('\n', ' ').strip()
    return text.lower()

def truncate_text(text, max_length=500):
    return text[:max_length]

def retrieve_context(input_text: str, n_results: int = 5):
    results = collection.query(query_texts=[input_text], n_results=n_results)
    context = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append({"text": doc, "label": meta["label"].upper()})
    return context

def build_prompt(input_text: str, retrieved_context: list):
    example_path = os.path.join(os.path.dirname(__file__), "few_shot_examples.json")
    with open(example_path, "r") as f:
        few_shot_examples = json.load(f)
    all_examples = retrieved_context + few_shot_examples
    example_str = ""
    for i, ex in enumerate(all_examples[:5]):
        example_str += f"Example {i+1} ({ex['label']}): {ex['text'][:200]}...\n\n"
    prompt = f"""
You are a fact-checking assistant. Classify the news article below as FAKE or REAL.

**Instructions**:
1. Analyze these examples:
{example_str}

2. Classify this article:
{input_text[:2000]}

Return **ONLY** a JSON object with these keys:
- \"classification\" (FAKE/REAL)
- \"confidence\" (0.0-1.0)
- \"reasoning\" (1-2 sentences)
"""
    return prompt

def classify_news_article(input_text: str):
    retrieved_context = retrieve_context(input_text)
    prompt = build_prompt(input_text, retrieved_context)
    response = model.generate_content(prompt, generation_config=generation_config)
    try:
        result = json.loads(response.text)
        # Save the result model in a JSON file for record
        model_path = os.path.join(os.path.dirname(__file__), "../model_storage/model.json")
        with open(model_path, "w") as f:
            json.dump(result, f)
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}
