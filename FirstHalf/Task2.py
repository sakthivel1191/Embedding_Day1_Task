from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What is the leave policy?",
    "How many casual leaves do I get?",
    "The weather in Chennai is hot",
    "Office timings are 9 to 6"
]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True)
similarity_matrix = util.cos_sim(embeddings, embeddings)
print("SIMILARITY MATRIX")
print(similarity_matrix.cpu().numpy().round(4))
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j].item()
        pairs.append((score, sentences[i], sentences[j]))
pairs.sort(key=lambda x: x[0], reverse=True)
print("\nTOP 3 MOST SIMILAR PAIRS")
for score, sentence1, sentence2 in pairs[:3]:
    print("\nSentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("Similarity:", round(score, 4))
print("\nLEAST SIMILAR PAIR")
score, sentence1, sentence2 = pairs[-1]
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity:", round(score, 4))
print("\nPAIR WITH DIFFERENT WORDS BUT SIMILAR MEANING")
print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Similarity:", round(similarity_matrix[0][1].item(), 4))
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
plt.figure(figsize=(12, 7))
plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    color='blue'
)
for i, sentence in enumerate(sentences):
    plt.annotate(
        sentence,
        (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
        fontsize=8
    )
plt.title("2D Visualization of Sentence Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.tight_layout()
plt.savefig("similarity_plot.png")
plt.show()
