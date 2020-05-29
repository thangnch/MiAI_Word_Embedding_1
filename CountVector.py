from sklearn.feature_extraction.text import CountVectorizer

corpus = [
     "Hôm nay chúng ta học Mì AI theo cách mì ăn liền".lower(),
     "Chúng ta cứ học giỏi là có tiền liền".lower()
]

vectorizer = CountVectorizer()
sentences_vectors = vectorizer.fit_transform(corpus)

print(sentences_vectors.toarray())
print("Vocab = ", vectorizer.vocabulary_)
print("Vector của câu 2 =\n", sentences_vectors.toarray()[1])

print("Vector từ của từ Mì =\n",sentences_vectors.toarray()[:,10])

# Vocab =  {'hôm': 6, 'nay': 11, 'chúng': 1,
#           'ta': 12, 'học': 7, 'mì': 10, 'ai': 0,
#           'theo': 13, 'cách': 2, 'ăn': 15, 'liền': 8,
#           'cứ': 4, 'giỏi': 5, 'là': 9, 'có': 3, 'tiền': 14}

# [[1 1 1 0 0 0 1 1 1 0 2 1 1 1 0 1]
#  [0 1 0 1 1 1 0 1 1 1 0 0 1 0 1 0]]

