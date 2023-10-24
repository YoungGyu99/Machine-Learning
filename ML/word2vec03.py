import os
from konlpy.tag import Okt
from gensim.models import word2vec
from gensim.models import fasttext
import csv

data = word2vec.PathLineSentences('naver_movie.nlp')
word2vec_model = word2vec.Word2Vec(data
                                   , vector_size=200, window=8,min_count=2,sg=1)
fasttext_model = fasttext.FastText(data, vector_size=200, window=8
                                   ,min_count=2, sg=1)
word2vec_model.save('movie_word2vec.model')
fasttext_model.save('movie_fasttext.model')
while True:
    text = input("검색 단어")
    # print('word2vec:', word2vec_model.wv.most_similar(positive=[text]))
    print('fasttext:', fasttext_model.wv.most_similar(positive=[text]))
