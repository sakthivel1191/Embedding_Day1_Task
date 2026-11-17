from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

model = SentenceTransformer('all-MiniLM-L6-v2')
print("BONUS: PRECISION AND RECALL")
sample = [
    "My internet is not working",
    "Internet connection is not available",
    "My salary is delayed",
    "My salary has not been credited",
    "My laptop is overheating",
    "My office chair is uncomfortable"
]
true_labels = [1, 1, 0]
embeddings = model.encode(sample)
similarity_scores = []
for i in range(0, len(sample), 2):
    score = cosine_similarity(
        [embeddings[i]],
        [embeddings[i + 1]]
    )[0][0]
    similarity_scores.append(score)
threshold = 0.70
predicted_labels = []
for score in similarity_scores:
    if score >= threshold:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)
precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)
recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)
print("Threshold:", threshold)
print("Actual Labels:", true_labels)
print("Predicted Labels:", predicted_labels)
print("Similarity Scores:", [round(x, 4) for x in similarity_scores])
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
