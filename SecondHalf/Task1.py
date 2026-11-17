import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_scratch(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
test_pairs = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0])
]
for a, b in test_pairs:
    result = cosine_similarity_scratch(a, b)
    sklearn_result = cosine_similarity([a], [b])[0][0]
    print(a, "vs", b)
    print("My Result:", round(result, 4))
    print("Sklearn Result:", round(sklearn_result, 4))
    print()
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
normalized_a = a / np.linalg.norm(a)
normalized_b = b / np.linalg.norm(b)
cosine_result = cosine_similarity_scratch(a, b)
dot_result = np.dot(normalized_a, normalized_b)
print("Cosine Similarity:", round(cosine_result, 4))
print("Normalized Dot Product:", round(dot_result, 4))
