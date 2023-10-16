from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

reviews = [
    "이 제품 정말 좋아요. 다음에도 구매 의향 있습니다.",
    "배송이 너무 늦어서 별로였습니다.",
    "가격 대비 품질이 훌륭합니다.",
    "서비스가 아쉬웠어요. 다음 번엔 개선 바랍니다.",
    "색상이 사진과 조금 다르네요.",
    "설명대로의 제품이 도착해서 만족합니다.",
    "제품이 고장나서 AS를 받았습니다.",
    "디자인이 너무 예쁘고 성능도 좋습니다.",
    "사용 방법이 조금 어렵네요.",
    "다른 사람들에게도 추천하고 싶은 제품입니다."
]

target_keyword = input("검색어를 입력해 주세요(단어):")

okt = Okt()

def tokenizer(doc):
    return ["-".join(t) for t in okt.pos(doc)]  # 수정된 부분

vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=["./Punctuation", "이/Josa", "가/Josa", "을/Josa", "를/Josa", "는/Jose", "이/VCP"])
tfidf_matrix = vectorizer.fit_transform(reviews)
print(tfidf_matrix)
# 단어 tf-idf 벡터 생성
target_tfidf = vectorizer.transform([target_keyword])
# 코사인 유사도
# 두 벡터의 dot product (flatten() 1차원 배열로 만듬
cosine_sim = linear_kernel(target_tfidf, tfidf_matrix).flatten()
review_idx = cosine_sim.argsort()[:-4:-1]  # 상위3개

for idx in review_idx:
    print(reviews[idx], "-> sim:", cosine_sim[idx])


