from sentence_transformers import SentenceTransformer
import numpy as np

# 🔥 LOAD ONCE (GLOBAL)
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("genai/knowledge.txt", "r") as f:
    documents = f.readlines()

doc_embeddings = model.encode(documents)


def retrieve_context(query, top_k=2):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:]
    return [documents[i] for i in top_indices]


def refine_decision(data, weather, health, rl_action):

    try:
        query = f"""
        soil moisture {data['soil_moisture']},
        temperature {weather['temperature']},
        humidity {weather['humidity']},
        plant health {health}
        """

        context = retrieve_context(query)

        adjustment = 0

        for text in context:
            if "Low soil moisture" in text:
                adjustment += 0.3
            if "High temperature" in text:
                adjustment += 0.1
            if "High humidity" in text:
                adjustment -= 0.1
            if "Poor plant health" in text:
                adjustment += 0.2

        final = rl_action + adjustment

        return max(0, min(final, 1)), context

    except Exception as e:
        print("RAG ERROR:", e)
        return rl_action, []