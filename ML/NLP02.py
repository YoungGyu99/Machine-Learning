import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout
from keras.preprocessing.text import text_to_word_sequence

docs = ['너무 재미있다'
        ,'최고 예요'
        ,'참 잘 만든 영화 대박'
        ,'추천하고 싶은 영화 입니다.'
        ,'한번 더 보고싶다'
        ,'별로예요'
        ,'생각보다 지루하다'
        ,'연기가 어색해요'
        ,'재미없어요'
        ,'최악의 영화']
y = np.array([1,1,1,1,1,0,0,0,0,0])
# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)
print(x)
# 임베딩 단어수
word_size = len(token.word_index) + 1

embedding_size = 200
pad_x = pad_sequences(x, 4)
embedding_matrix = np.zeros((word_size, embedding_size))
from gensim.models import FastText
fasttext_model = FastText.load('movie_fasttext.model')
wv = fasttext_model.wv
print(pad_x)
def fn_getvector(w, embe):
    if w in embe:
        return embe[w]
    else:
        return None

for word, idx in token.word_index.items():
    if word in wv:
        embedding_matrix[idx] = wv[word]
# 모델생성
model = Sequential()
model.add(Embedding(word_size, embedding_size, input_length=4, weights=[embedding_matrix], trainable=False))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(pad_x, y, epochs=20)
print('acc:', model.evaluate(pad_x, y)[1])

while True:
    text = input("평가 내용:")
    test_x = token.texts_to_sequences([text])
    test_x_pad = pad_sequences(test_x, 4)
    pred = model.predict(test_x_pad)
    print(pred)