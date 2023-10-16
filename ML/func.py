# python 익명함수

# 정의와 동시에 사용
result = (lambda x : x + 1)(3)
print(result)
# 객체에 담아 사용
func01 = lambda  x, y : x * y + 1
print(func01(4, 2))
# 제곱시키기
arr = [1, 2, 3, 4, 5]
# map은 각 데이터에 함수를 대응시켜 그 값을 반환하는 역할
res = list(map(lambda  x : x**2, arr))
print('res :', res)