# RAG-Based HR Policy Q&A System

A simple **Retrieval-Augmented Generation (RAG)** application that answers employee questions based only on a predefined set of HR policies.

The project uses **Sentence Transformers** to convert documents and queries into embeddings, **cosine similarity** to retrieve the most relevant policy, and **Groq's Llama 3.3 70B** model to generate the final answer.

## Features

* Semantic search using sentence embeddings
* Retrieves the most relevant HR policy using cosine similarity
* Uses Groq Llama 3.3 70B for answer generation
* Restricts the LLM to the retrieved context to reduce hallucination
* Simple and lightweight RAG pipeline
* Environment-variable based API key management

## How It Works

The application follows a basic RAG pipeline:

```text
HR Documents
     ↓
Sentence Transformer
     ↓
Document Embeddings
     ↓
User Query
     ↓
Query Embedding
     ↓
Cosine Similarity
     ↓
Most Relevant Document
     ↓
Retrieved Context
     ↓
Groq Llama 3.3 70B
     ↓
Final Answer
```

## Example

Given the following HR policy:

```text
Employees receive 24 days of paid leave per year.
```

User query:

```text
How much vacation do I get?
```

The system:

1. Converts the question into an embedding.
2. Compares it with the embeddings of all HR policies.
3. Finds the most relevant policy.
4. Sends the retrieved policy along with the question to the LLM.
5. Generates the answer:

```text
You get 24 days of paid leave per year.
```

## Technologies Used

* **Python**
* **Sentence Transformers**
* **NumPy**
* **Groq API**
* **Llama 3.3 70B Versatile**
* **python-dotenv**

## Project Structure

```text
RAG-HR-Policy/
│
├── main.py
├── .env
├── .gitignore
└── README.md
```


## Core Components

### 1. Document Embeddings

The project uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Each HR policy is converted into a numerical vector representation.

```python
document_embeddings = model.encode(documents)
```

This allows the system to compare the semantic meaning of the user's question with each policy.

### 2. Cosine Similarity

Cosine similarity is used to measure how similar the query embedding is to each document embedding.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

A higher similarity score means that the document is more relevant to the query.

### 3. Retrieval

The `retrieve()` function compares the query with every document and returns the document with the highest similarity score.

```python
def retrieve(query_embedding):
    scores = []

    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))

    scores.sort(reverse=True)

    return scores[0]
```

### 4. LLM Generation

The retrieved policy is passed to the Groq Llama model as context.

The system prompt instructs the model to:

* Answer in one line
* Use only the provided context
* Avoid hallucinating information

This makes the generation step grounded in the retrieved HR policy.

## Why RAG?

A normal LLM may not know your company's internal HR policies.

RAG solves this by providing the relevant company information to the LLM at query time.

Instead of:

```text
Question → LLM → Answer
```

this project uses:

```text
Question
   ↓
Retrieve relevant information
   ↓
Provide information to LLM
   ↓
Generate grounded answer
```

This approach is useful for applications such as:

* HR assistants
* Company policy assistants
* Internal knowledge bases
* Customer support systems
* Documentation assistants
* FAQ systems

## Limitations

* Documents are currently hardcoded in the Python file.
* Only the single highest-scoring document is retrieved.
* There is no vector database.
* No document chunking is implemented.
* The system does not currently use a similarity threshold.
* The application is currently command-line based.

## Learning Outcome

This project demonstrates the fundamentals of a **Retrieval-Augmented Generation pipeline**, including:

```text
Embeddings
   ↓
Semantic Search
   ↓
Cosine Similarity
   ↓
Context Retrieval
   ↓
LLM Generation
```

It provides a simple foundation for building more advanced RAG applications using vector databases and production-grade retrieval systems.

## License

This project is for educational and demonstration purposes.
