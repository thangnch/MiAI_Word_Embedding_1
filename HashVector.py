from sklearn.feature_extraction.text import HashingVectorizer

sentences = [
    "Hôm nay chúng ta học Mì AI theo cách Mì ăn liền".lower(),
         "Chúng ta cứ học giỏi là có tiền liền".lower()
]

vectorizer = HashingVectorizer(norm=None, n_features=20)
sentence_vectors = vectorizer.fit_transform(sentences)

print(sentence_vectors.toarray())
print(vectorizer.n_features)