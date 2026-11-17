from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sentences = [
    "எனக்கு தமிழ் மிகவும் பிடிக்கும்",
    "நான் தமிழ் மொழியை விரும்புகிறேன்",
    "The weather is very hot today"
]
embeddings = model.encode(sentences, convert_to_tensor=True)
similarity = util.cos_sim(embeddings, embeddings)
print("TAMIL EMBEDDINGS")
print("Shape:", embeddings.shape)
print("\nSIMILARITY MATRIX")
print(similarity.cpu().numpy().round(4))
print("\nTAMIL SENTENCE SIMILARITY")
print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Similarity:", round(similarity[0][1].item(), 4))
