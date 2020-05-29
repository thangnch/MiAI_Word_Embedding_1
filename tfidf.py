from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#pip install pandas

#corpus gồm 2 câu
docs = [ "Hôm nay chúng ta học Mì AI theo cách Mì ăn liền".lower(),
         "Chúng ta cứ học giỏi là có tiền liền".lower()
        ]
# Tham so user_idf de tinh ca TF và IDF
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
tfidf_encoder = tfidf_vectorizer.fit_transform(docs)

# Lấy tf idf của câu đầu tiên
first_sentence_encoder = tfidf_encoder[0]

#Vector biểu diễn câu
print("Vector TF IDF của câu 0=\n", first_sentence_encoder.toarray())
print("Vocab = ", tfidf_vectorizer.get_feature_names())

# Visualize cho dễ nhìn
df = pd.DataFrame(first_sentence_encoder.T.todense(),
                  index=tfidf_vectorizer.get_feature_names(),
                  columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False, inplace=True)
print(df)



