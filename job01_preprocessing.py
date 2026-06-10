import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('datasets/reviews_2017_2022.csv')
df.info()

df_stopwords = pd.read_csv('datasets/stopwords.csv')
stopwords = df_stopwords['stopword'].tolist()
stopwords = stopwords + ['가다', '감독', '연출', '연기', '배우', '하다', '모르다', '보여주다', '주연', '많다', '좋다']

okt = Okt()
print(df.titles[0])
print(df.reviews[0])

# pos - 형태소 분리 및 품사 표현(명사, 동사, 부사만 표현됌)
# tokened_review = okt.pos(df.reviews[0])
# print(tokened_review)

cleaned_sentences = []
for review in df.reviews:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem = True)
    df_token = pd.DataFrame(tokened_review, columns = ['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]
    words = []
    for word in df_token['word']:
        if len(word) > 1:
            # 불용어 제거
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df.reviews = cleaned_sentences
df.dropna(inplace = True)
df.info()
df.to_csv('datasets/reviews_2017_2022_test.csv', index = False)