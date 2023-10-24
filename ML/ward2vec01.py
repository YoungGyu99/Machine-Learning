# pip install gensim
import requests
import re
from gensim.models import Word2Vec

res = requests.get('https://www.gutenberg.org/files/2591/2591-0.txt')
grimm = res.text[2801:530661]
grimm = re.sub(r'[^a-zA-Z\.]', ' ', grimm)
sentences = grimm.split('. ')
data = [s.split() for s in sentences]
embedding_model = Word2Vec(data, sg=1,  # 0: CBOW, 1: skip-gram
                            vector_size=100,   # 단어의 임베딩 차원
                            window=3,   # 앞뒤 단어를 얼마나 고려할지
                            min_count=3,  # 출현 최소 빈도
                            workers=4)  # 동시처리 작업 수
embedding_model.save("book.model")
print(embedding_model)

while True:
    input_text = input("비교 단어를 입력 (end=q): ")
    if input_text == 'q':
        break
    text1, text2 = input_text.split()
    print("가까운 단어:", embedding_model.wv.most_similar(positive=[text1, text2]))
    print("먼 단어:", embedding_model.wv.most_similar(negative=[text1, text2]))
