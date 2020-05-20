import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
def make_label(df):
    df["sentiment"] = df["star"].apply(lambda x: 1 if x>3 else 0)
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding='gb2312') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list
stop_words_file = "stopwordsHIT.txt"
stopwords = get_custom_stopwords(stop_words_file)
nb = MultinomialNB()
df = pd.read_csv('data.csv', encoding='gb18030')
print(df.head())
print(df.shape)
make_label(df)
X = df[['comment']]
y = df.sentiment
X['cutted_comment'] = X.comment.apply(chinese_word_cut)
X.cutted_comment[:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = CountVectorizer()
term_matrix = pd.DataFrame(vect.fit_transform(X_train.cutted_comment).toarray(), columns=vect.get_feature_names())
vect = CountVectorizer(stop_words=frozenset(stopwords))
term_matrix = pd.DataFrame(vect.fit_transform(X_train.cutted_comment).toarray(), columns=vect.get_feature_names())
max_df = 0.8 # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
min_df = 3 # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
vect = CountVectorizer(max_df = max_df, 
                       min_df = min_df, 
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                       stop_words=frozenset(stopwords))
term_matrix = pd.DataFrame(vect.fit_transform(X_train.cutted_comment).toarray(), columns=vect.get_feature_names())
print(term_matrix.head())
pipe = make_pipeline(vect, nb)
pipe.steps
cross_val_score(pipe, X_train.cutted_comment, y_train, cv=5, scoring='accuracy').mean()
pipe.fit(X_train.cutted_comment, y_train)
pipe.predict(X_test.cutted_comment)
y_pred = pipe.predict(X_test.cutted_comment)
print('score:',metrics.accuracy_score(y_test, y_pred))

