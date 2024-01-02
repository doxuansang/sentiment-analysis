import io
import sys
import nltk
#nltk.download()

# input: câu phản hồi
input = 'thầy giảng hay, tận tình'

# list chứa sents từ train và dev
sents_list = []

# load dữ liệu
filename = "sents_train.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sents_list.append(line)

filename = "sents_dev.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sents_list.append(line)

# list chứa sentiments từ train và dev
sentiments_list = []

filename = "sentiments_train.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sentiments_list.append(line)

filename = "sentiments_dev.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sentiments_list.append(line)

# list chứa topics từ train và dev
topics_list = []

filename = "topics_train.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        topics_list.append(line)

filename = "topics_dev.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        topics_list.append(line)

# list chứa sents từ test
sents_list_test = []

filename = "sents_test.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sents_list_test.append(line)

# list chứa sentiments từ test
sentiments_list_test = []

filename = "sentiments_test.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        sentiments_list_test.append(line)

# list chứa topics từ test
topics_list_test = []

filename = "topics_test.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip(' .\n')
        topics_list_test.append(line)


# kiểm tra xem câu phản hồi đã có trong bộ dữ liệu Vietnamese Students’ Feedback Corpus (UIT-VSFC) chưa?
# nếu có, ta có luôn kết quả
# nếu không có, ta chạy thuật toán Naive Bayes

# vị trí của câu input, nếu nó xuất hiện trong sents_list
i_index = 0

if sents_list.count(input) > 0:
    i_index = sents_list.index(input)
    #print(i_index)
    filename = "result_sentiments.txt"
    with open(filename, mode='w') as f:
        f.write(sentiments_list[i_index])
    sys.exit()

"""
# chia nhóm dữ liệu thành 4 nhóm
text_list0 = [] # giảng viên
text_list1 = [] # chương trình đào tạo
text_list2 = [] # cơ sở vật chất
text_list3 = [] # khác

for i in range(len(text_list)):
    if topics_list[i] == '0':
        text_list0.append(text_list[i])
    elif topics_list[i] == '1':
        text_list1.append(text_list[i])
    elif topics_list[i] == '2':
        text_list2.append(text_list[i])
    else:
        text_list3.append(text_list[i])

print(text_list0)
print(text_list1)
print(text_list2)
print(text_list3)
"""
print("Naive Bayes")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from nltk.tag import pos_tag
import pandas as pd


# list chứa các stop words, những từ xuất hiện nhiều nhưng không có biểu đạt cảm xúc
# do stop words được tự tạo, thời gian hạn chế, tài nguyên thiếu thốn, ...
# vì thế nên tập stop words cón khá sơ sài và có thể có sai sót
stop_words_list = []

filename = "stop_words.txt"
with io.open(filename, 'r', encoding='utf-8', newline='\n') as f:
    for line in f.readlines():
        line = line.strip('\r\n')
        stop_words_list.append(line)

# xử lý dữ liệu đầu vào
def data_cleaning(text_list):
    stopwords_rem = False
    stopwords_vn = stop_words_list
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer()
    reconstructed_list = []
    for each_text in text_list:
        lemmatized_tokens = []
        tokens = tokenizer.tokenize(each_text.lower())
        pos_tags = pos_tag(tokens)
        for each_token, tag in pos_tags:
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_token = lemmatizer.lemmatize(each_token, pos)
            if stopwords_rem:  # False
                if lemmatized_token not in stopwords_vn:
                    lemmatized_tokens.append(lemmatized_token)
            else:
                lemmatized_tokens.append(lemmatized_token)
        reconstructed_list.append(' '.join(lemmatized_tokens))
    return reconstructed_list

estimators = [('cleaner', FunctionTransformer(data_cleaning)),
              ('vectorizer', TfidfVectorizer(max_features=100000, ngram_range=(1, 2)))]
preprocessing_pipeline = Pipeline(estimators)

X_train_transformed = preprocessing_pipeline.fit_transform(sents_list)

# Create a Naive Bayes model and fit training data
nb = MultinomialNB()
nb.fit(X_train_transformed, sentiments_list)
X_test_transformed = preprocessing_pipeline.transform(sents_list_test)

print("Độ chính xác:")
print(nb.score(X_test_transformed, sentiments_list_test))
print(nb.score(X_train_transformed, sentiments_list))

count_vect = CountVectorizer()
count_vect.fit(sents_list)

transformed = count_vect.transform([input])

counts = count_vect.transform(sents_list)


nb = MultinomialNB()
nb.fit(counts, sentiments_list)
sentiments = nb.predict(transformed)

filename = "result_sentiments.txt"
with open(filename, mode='w') as f:
    f.writelines(sentiments)
