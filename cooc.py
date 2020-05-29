import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Định nghĩa Corpus
corpus = [ "Hôm nay chúng ta học Mì AI theo cách Mì ăn liền".lower(),
         "Chúng ta cứ học giỏi là có tiền liền".lower()
        ]

# Cách 1. Lấy vocab bằng Count Vectorize
vectorizer = CountVectorizer()
sentences_vectors = vectorizer.fit_transform(corpus)
print(sentences_vectors.toarray())
vocab  = list(vectorizer.vocabulary_)
print(vocab)

# Cách 2. Làm tương tự bài one-hot (dùng Set)

# Khởi tạo ma trận DXH A

A = np.zeros(   (len(vocab),len(vocab))    ,np.int32)

# Duyet tung cau, tung tu de tinh ma trận đồng xuất hiện

def get_beside_words(sentence, index, window_size):
    words = sentence.split()
    beside_words = []

    for w in range(1,window_size+1):
        if index-w>=0:
            beside_words.append(words[index-w])
        if index+w<len(words):
            beside_words.append(words[index+w])

    return beside_words

window_size = 1
# Duyệt từng câu trong corps
for sentence in corpus:
    # Duyệt từng từ trong câu
    for index, word in enumerate(sentence.split()):
        print("Xử lý từ ", word)
        # Lấy các từ xung quanh từ hiện tại
        beside_words = get_beside_words(sentence, index, window_size)
        print("Các từ xung quanh = ", beside_words)
        # Duyệt và tăng phần tử trong ma trận A
        for word2 in beside_words:
            A[vocab.index(word),vocab.index(word2)]+=1

print(A)