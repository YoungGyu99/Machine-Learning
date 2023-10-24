import os
from konlpy.tag import Okt
from gensim.models import  word2vec
from gensim.models import fasttext
import csv

def readText(path):
    f = open(path, 'r', encoding='utf-8')
    reader = csv.reader(f, delimiter="\t")
    data = list(reader)
    f.close()
    print(data)
    okt = Okt()
    result = []
    for line in data:
        d = okt.pos(line[1], norm=True, stem=True)
        r = []
        for word in d:
            if not word[1] in ["Josa", 'Eomi', 'Punctuation']:
                r.append(word[0])
        rl = " ".join(r).strip()
        result.append(rl)

    # 파일에 처리된 데이터를 저장
    with open('naver_movie.nlp', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(result))

    # 모든 데이터 처리 후에 "데이터 저장" 메시지 출력
    print("save data")


# 함수 호출
readText("ratings_train.txt")
