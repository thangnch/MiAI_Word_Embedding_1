#pip install numpy
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Corpus gồm 2 văn bản doc1, doc2
doc1 = "Hôm nay chúng ta học Mì AI theo cách mì ăn liền".lower()
doc2 = "Chúng ta cứ học giỏi là có tiền liền".lower()

# Tính vocab
doc1 =  doc1.split()
#print(doc1)
doc2 = doc2.split()
#print(doc2)
corpus = doc1 + doc2
#print(corpus)

vocab = list(set(corpus))
print("Vocab của Corpus=", vocab)

# Chuyen list vocab thành số
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(vocab)
print("Gán số cho từng từ trong vocab=\n", integer_encoded)
for i in range(len(vocab)):
    print("{} - {}".format(vocab[i], integer_encoded[i]))


onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("Bảng biểu diễn các từ=\n", onehot_encoded)

word = label_encoder.inverse_transform([np.argmax(onehot_encoded[5,:])])
print("Từ của vector từ số 5 =", word)
# ['hôm', 'nay', 'chúng', 'ta', 'học', 'mì', 'ai',
#  'theo', 'cách', 'mì', 'ăn', 'liền', 'chúng', 'ta',
#  'cứ', 'học', 'giỏi', 'là', 'có', 'tiền', 'liền']
