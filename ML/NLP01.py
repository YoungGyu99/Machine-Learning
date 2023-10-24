import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
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
pad_x = pad_sequences(x, 4)
print(pad_x)

# 임베딩 단어수
word_size = len(token.word_index) + 1
# 모델생성
model = Sequential()
model.add(Embedding(word_size, 8, input_length=4))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, y, epochs=20)
print('acc:', model.evaluate(pad_x, y)[1])
while True:
    text = input("평가 내용:")
    token.fit_on_texts([text])
    test_x = token.texts_to_sequences([text])
    test_x = pad_sequences(test_x, 4)
    pred = model.predict(test_x)
    print(pred)