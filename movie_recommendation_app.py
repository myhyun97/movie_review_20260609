import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('movie_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('models/Tfidf_movie_review.mtx').tocsr()
        with open('models/tfidf.pkl', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('models/word2vec_movie_review.model')

        self.df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')
        self.titles = list(self.df_reviews.titles)
        self.titles.sort()
        for title in self.titles:
            self.cb_title.addItem(title)

        model = QStringListModel()
        model.setStringList((self.titles))
        completer = QCompleter()
        completer.setModel(model)
        self.le_keyword.setCompleter(completer)

        self.cb_title.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommend.clicked.connect(self.btn_keywords_clicked)

    def btn_keywords_clicked(self):
        keyword = self.le_keyword.text()
        if keyword in self.titles:
            recommendations = self.recommendation_by_title(keyword)
        else:
            recommendations = self.recommendation_by_keyword(keyword)
        self.lb_recommendation.setText(recommendations)

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[:11]

    def combobox_slot(self):
        title = self.cb_title.currentText()
        print(title)
        recommendations = self.recommendation_by_title(title)
        self.lb_recommendation.setText(recommendations)

    def recommendation_by_title(self, title):
        movieIdx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movieIdx], self.Tfidf_matrix)
        recommendations = self.getRecommendation(cosine_sim)
        recommendations = '\n'.join(recommendations[1:])
        return recommendations

    def recommendation_by_keyword(self, keyword):
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
        except:
            return '제가 모르는 단어에요 ㅠㅠ'
        sentence = [keyword] * 11
        count = 10
        for word, _ in sim_word:
            sentence = sentence + [word] * count
            count = count - 1
        # print(sentence)
        sentence = ' '.join(sentence)
        # print(sentence)
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendations = self.getRecommendation(cosine_sim)
        recommendations = '\n'.join(recommendations[:10])
        return recommendations

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())