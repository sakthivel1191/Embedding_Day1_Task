from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
print("EMBEDDING USE CASES")
print("\nSEMANTIC SEARCH")
faqs = [
    "How can I reset my password?",
    "How do I change my account password?",
    "How can I recover my login credentials?",
    "What is the company leave policy?",
    "How many casual leaves are allowed?",
    "How do I apply for sick leave?",
    "What are the office working hours?",
    "When does the office open?",
    "What is the lunch break timing?",
    "How can I apply for work from home?",
    "How do I update my personal details?",
    "How can I contact HR?",
    "What is the salary payment date?",
    "How do I download my salary slip?",
    "How can I apply for annual leave?"
]
queries = [
    "I forgot my password",
    "How many holidays can I take?",
    "When should I come to office?"
]
faq_embeddings = model.encode(faqs)
for query in queries:
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    print("\nQuery:", query)
    for index in top_indices:
        print(faqs[index], "->", round(scores[index], 4))
print("\nBusiness Use: Semantic search helps customers find relevant answers based on meaning.")
print("DUPLICATE DETECTION")
complaints = [
    "My internet connection is not working",
    "Internet is not connecting properly",
    "I cannot connect to the company wifi",
    "My salary has not been credited",
    "Salary payment is still pending",
    "I did not receive my monthly salary",
    "The application crashes when I open it",
    "App keeps crashing during startup",
    "The mobile application is not opening",
    "My laptop battery is draining quickly",
    "Laptop battery runs out very fast",
    "I am unable to charge my laptop"
]
complaint_embeddings = model.encode(complaints)
similarity_matrix = cosine_similarity(complaint_embeddings)
threshold = 0.80
print("Similarity Threshold:", threshold)
for i in range(len(complaints)):
    for j in range(i + 1, len(complaints)):
        score = similarity_matrix[i][j]
        if score >= threshold:
            print("\nPossible Duplicate")
            print("Complaint 1:", complaints[i])
            print("Complaint 2:", complaints[j])
            print("Similarity:", round(score, 4))
print("\nBusiness Use: Duplicate detection helps customer support teams identify repeated complaints.")
print("\nCLUSTERING USING K-MEANS")
headlines = [
    "India wins the cricket match",
    "Football team wins championship",
    "Cricket player scores a century",
    "Olympics sports competition begins",
    "Football world cup final results",
    "Indian cricket team announces squad",
    "Stock market reaches new high",
    "Company reports increased profits",
    "Bank announces new interest rates",
    "Technology company shares increase",
    "Investors expect market growth",
    "New business startup receives funding",
    "Artificial intelligence changes technology",
    "New smartphone launched",
    "Python programming language is popular",
    "Machine learning improves healthcare",
    "New computer processor announced",
    "AI tools help software developers",
    "Technology companies develop new products",
    "Cloud computing becomes more popular",
    "Government announces new education policy",
    "Students prepare for examinations",
    "University introduces new courses",
    "Schools reopen after holidays",
    "Teachers attend training programs",
    "Students receive scholarship opportunities",
    "New research improves medical treatment",
    "Hospital introduces modern equipment",
    "Doctors discover new treatment",
    "Health awareness campaign begins"
]
headline_embeddings = model.encode(headlines)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(headline_embeddings)
for i in range(3):
    print("\nCluster:", i)
    for j in range(len(headlines)):
        if clusters[j] == i:
            print("-", headlines[j])
print("\nCluster names should be assigned after checking the headlines in each cluster.")
print("\nBusiness Use: Clustering helps businesses group similar news and documents automatically.")
print("\nPRODUCT RECOMMENDATION")
products = [
    "Wireless Bluetooth headphones with noise cancellation",
    "Wireless earbuds with high quality sound",
    "Smartphone with powerful camera",
    "Laptop with 16GB RAM and SSD storage",
    "Gaming laptop with high performance graphics",
    "Smart watch with fitness tracking",
    "Fitness band with heart rate monitor",
    "Bluetooth speaker with deep bass",
    "Portable wireless music speaker",
    "Mechanical keyboard for gaming",
    "Wireless mouse for office use",
    "Tablet with large display",
    "Laptop with long battery life",
    "Noise cancelling wireless earphones",
    "Smartphone with long battery life"
]
new_product = "Wireless headphones with excellent sound quality"
product_embeddings = model.encode(products)
new_embedding = model.encode([new_product])
scores = cosine_similarity(new_embedding, product_embeddings)[0]
top_indices = np.argsort(scores)[::-1][:3]
print("Input Product:", new_product)
print("\nTop 3 Recommended Products:")
for index in top_indices:
    print(products[index], "->", round(scores[index], 4))
print("\nBusiness Use: Product recommendation helps online stores suggest similar products to customers.")
