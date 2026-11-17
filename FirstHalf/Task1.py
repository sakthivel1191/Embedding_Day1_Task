from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()
sentences = [
    "I love learning Python",
    "Machine learning is interesting",
    "I want to become a data scientist",
    "Python is easy to learn",
    "Artificial intelligence is growing",
    "I enjoy coding every day",
    "Data science uses machine learning",
    "I am learning about embeddings"
]
model = SentenceTransformer('all-MiniLM-L6-v2')
start = time.time()
local_embeddings = model.encode(sentences)
local_time = time.time() - start
print("LOCAL EMBEDDINGS")
print("Shape:", local_embeddings.shape)
print("Time:", round(local_time, 2), "seconds")
print("First 5 values:", local_embeddings[0][:5])
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OpenRouter API key not found in .env file")
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)
start = time.time()
response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=sentences
)
api_time = time.time() - start
api_embeddings = [item.embedding for item in response.data]
print("\nOPENROUTER EMBEDDINGS")
print("Shape:", (len(api_embeddings), len(api_embeddings[0])))
print("Dimension:", len(api_embeddings[0]))
print("Time:", round(api_time, 2), "seconds")
print("First 5 values:", api_embeddings[0][:5])
sentences_100 = sentences * 12 + sentences[:4]
start = time.time()
local_100 = model.encode(sentences_100)
local_100_time = time.time() - start
start = time.time()
api_100 = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=sentences_100
)
api_100_time = time.time() - start
total_tokens = api_100.usage.total_tokens
print("\n100 SENTENCES COMPARISON")
print("Total sentences:", len(sentences_100))
print("Local time:", round(local_100_time, 2), "seconds")
print("OpenRouter API time:", round(api_100_time, 2), "seconds")
print("Total API tokens:", total_tokens)
print("\nEmbedding model selection:")
print("1. Local models are useful for offline processing.")
print("2. API models are useful for easy integration.")
print("3. Local models do not require API charges.")
print("4. API models are useful for managed scalable services.")
