import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"

documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

document_embeddings = model.encode(documents)

def cosine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query_embedding):
    scores = []
    for i, document in document_embeddings:
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents))
    scores.sort(reverse=True)
    return  scores[0] 




query = "How much vacation do I get?"
query_embedding = model.encode(query)

