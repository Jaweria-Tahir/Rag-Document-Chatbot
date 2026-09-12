# to analyze the concepts of  cosine simialrity and embedding 
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "The soup needs more salt and a squeeze of lemon.",       # cooking
    "I always let the dough rest for an hour before baking.", # cooking
    "The brakes on my car started squeaking last week.",      # cars
    "He swapped the old engine for a turbocharged V6.",       # cars
    "It's been raining on and off all afternoon.",             # weather
]

vectors = []

#converting each sentence into an embedding
for s in sentences:
    resp = ollama.embeddings(model="nomic-embed-text", prompt=s)
    emb = resp["embedding"]
    vectors.append(emb)
    print(f"'{s[:40]}...' -> dim={len(emb)}, first 5 values: {emb[:5]}")

vectors = np.array(vectors)

#manual cosine anlysis 
def manual_cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nManual cosine similarity:")
print("cooking-cooking:", manual_cosine(vectors[0], vectors[1]))
print("cooking-car    :", manual_cosine(vectors[0], vectors[2]))
print("car-car        :", manual_cosine(vectors[2], vectors[3]))
print("cooking-weather:", manual_cosine(vectors[0], vectors[4]))

#cosine similarity anlysis using sklearn 
sim_matrix = cosine_similarity(vectors)
print("\nFull similarity matrix (sklearn):")
np.set_printoptions(precision=3, suppress=True)
print(sim_matrix)

#QUESTION: If I embed my question and embed  100 chunks of a document, how do I find which chunks are relevant?

#ANSWER: Embed the question the same way (same model), compute cosine similarity between the question's vector and each of the 100 chunk vectors, sort by score descending, take the top-k. That's the entire retrieval step in RAG