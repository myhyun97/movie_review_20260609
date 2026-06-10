import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NanumBarunGothic')

df = pd.read_csv('datasets/reviews_2017_2022.csv')
movie_index1 = 16
words1 = df.iloc[movie_index1, 1].split()
print(df.iloc[movie_index1, 0])
movie_index2 = 308
words2 = df.iloc[movie_index2, 1].split()
print(df.iloc[movie_index2, 0])

worddict1 = collections.Counter(words1)
worddict1 = dict(worddict1)
print(worddict1)
worddict2 = collections.Counter(words2)
worddict2 = dict(worddict2)
print(worddict2)

wordcloud1 = WordCloud(font_path=font_path, background_color='white').generate_from_frequencies(worddict1)
wordcloud2 = WordCloud(font_path=font_path, background_color='white').generate_from_frequencies(worddict2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(wordcloud1)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(wordcloud2)
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[0].set_title(df.iloc[movie_index1, 0])
axes[1].set_title(df.iloc[movie_index2, 0])

plt.tight_layout()
plt.show()