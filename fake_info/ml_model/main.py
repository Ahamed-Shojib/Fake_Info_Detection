import pandas as pd

# Load fake and real news datasets
fake_df = pd.read_csv("/home/Ahamed_Shojib/fake_info/ml_model/data/Fake.csv")
real_df = pd.read_csv("/home/Ahamed_Shojib/fake_info/ml_model/data/True.csv")

# Add labels
fake_df['label'] = 'fake'
real_df['label'] = 'real'

combined_df = pd.concat([fake_df, real_df], axis=0)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(combined_df)}")
print(combined_df.head())


# Data Preprocessing

import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    # Remove URLs, special characters, and extra spaces
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace('\n', ' ').strip()
    return text.lower()

# Clean title and text columns
combined_df['cleaned_text'] = combined_df['title'] + " " + combined_df['text']
combined_df['cleaned_text'] = combined_df['cleaned_text'].apply(clean_text)

def truncate_text(text, max_length=500):
    return text[:max_length]  # Truncate to first 500 characters

combined_df['cleaned_text'] = combined_df['cleaned_text'].apply(truncate_text)

# Check max document length
max_length = combined_df['cleaned_text'].apply(len).max()
print(f"Longest document: {max_length} characters")

# A split of 70/30
train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

few_shot_df = train_df.sample(frac=0.1, random_state=42)

print(f"Training data: {len(train_df)} articles")
print(f"Few-shot examples: {len(few_shot_df)} articles")


#Vector Database

import os
import pandas as pd
import google.generativeai as genai
from chromadb import Client, Settings
from chromadb.utils import embedding_functions

# --- Manually set your API Key ---
# Either paste directly or use os.environ (secure via Colab secrets UI)
GOOGLE_API_KEY =   # Replace this
#AIzaSyBZVVD4P2oU26lx2TPleNfEpi8bIEQqq2s
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB client with Colab path
chroma_client = Client(
    settings=Settings(persist_directory="/home/Ahamed_Shojib/fake_info/ml_model/content/chroma_db", is_persistent=True)
)

# Define embedding function
embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY
)

# Create a collection
collection = chroma_client.create_collection(
    name="news_articles",
    embedding_function=embedding_func
)

# Define the batch ingestion function
def add_to_chroma(df, batch_size=100):
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        if len(row['cleaned_text']) > 10:
            documents.append(row['cleaned_text'])
            metadatas.append({"label": row['label']})
            ids.append(f"id_{idx}")

    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print(f"Added {len(documents)} valid articles (skipped {len(df)-len(documents)} empty)")

# Example: Make sure train_df, val_df, test_df are already loaded
add_to_chroma(train_df, batch_size=10)

# Save validation and test sets
val_df.to_csv("ml_model/validate_data/validation_set.csv", index=False)
test_df.to_csv("ml_model/validate_data/test_set.csv", index=False)


# Save the few-shot examples to a JSON file

import json
# Save few-shot examples to a JSON file
few_shot_examples = []
for _, row in few_shot_df.iterrows():
    few_shot_examples.append({
        "text": row['cleaned_text'],
        "label": row['label']
    })

with open("few_shot_examples.json", "w") as f:
    json.dump(few_shot_examples, f)

print(f"Saved {len(few_shot_examples)} few-shot examples.")


#Article Retrieval from VictorDataset

query = "A news article about trump"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print("Retrieved articles:")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"Label: {meta['label']}\nText: {doc[:100]}...\n")



#Gemini Model Inference

import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)
# Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.0-flash-exp")

# Define generation config for JSON output
generation_config = {
    "temperature": 0.1,  # Low temperature for factual tasks
    "response_mime_type": "application/json",  # Force JSON output
}


#Retrieving Context from VictorDataset

def retrieve_context(input_text: str, n_results: int = 5):
    # Query ChromaDB for similar articles
    results = collection.query(
        query_texts=[input_text],
        n_results=n_results,
    )

    # Extract text and labels from results
    context = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append({
            "text": doc,
            "label": meta["label"].upper()  # Convert to uppercase (FAKE/REAL)
        })

    return context

#Building the Prompt

def build_prompt(input_text: str, retrieved_context: list):
    # Load pre-saved few-shot examples
    with open("few_shot_examples.json", "r") as f:
        few_shot_examples = json.load(f)

    # Combine retrieved articles + few-shot examples
    all_examples = retrieved_context + few_shot_examples

    # Build prompt
    prompt = """
You are a fact-checking assistant. Classify the news article below as FAKE or REAL.

**Instructions**:
1. Analyze these examples:
{examples}

2. Classify this article:
{input_text}

Return **ONLY** a JSON object with these keys:
- "classification" (FAKE/REAL)
- "confidence" (0.0-1.0)
- "reasoning" (1-2 sentences)

Example response:
{{
  "classification": "FAKE",
  "confidence": 0.95,
  "reasoning": "This article contains sensationalist claims without credible sources."
}}
"""

    # Format examples
    example_str = ""
    for i, ex in enumerate(all_examples[:5]):  # Use top 5 examples
        example_str += f"Example {i+1} ({ex['label']}): {ex['text'][:200]}...\n\n"

    return prompt.format(
        examples=example_str,
        input_text=input_text[:2000]  # Trim to fit context window
    )


#Classifying the News Article

def classify_news_article(input_text: str):
    # Retrieve context
    retrieved_context = retrieve_context(input_text)

    # Build prompt
    prompt = build_prompt(input_text, retrieved_context)

    # Generate response
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )

    # Parse JSON output
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}
    


#Test Article from Dataset

# Get a test article (real or fake)
test_article = few_shot_df.iloc[0]["cleaned_text"]

print(test_article)

# Run classification
result = classify_news_article(test_article)
print(json.dumps(result, indent=2))




# Example: Analyze a new text

def analyze_text(text: str) -> dict:
    cleaned = clean_text(text)
    truncated = truncate_text(cleaned)
    return classify_news_article(truncated)

result = analyze_text("Trump pass his Phd at 2019")
print(result)
