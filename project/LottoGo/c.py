# -*- coding: utf-8 -*- 
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import date
import csv
f = open('data.csv', 'r')
rdr = csv.reader(f)

date1 = []
year = []
month = []
day = []

firstDate = date(2002,12,7)
firstganzi = [8,8,5]
firstzizi = [6,0,9]

manseTemp =[]
manse = []
ganzi = ["갑","을","병","정","무","기","경","신","임","계"]
ganziNumber = [3,8,7,2,5,10,9,4,1,6]
zizi = ["자","축","인","묘","진","사","오","미","신","유","술","해"]
ziziNumber = [1,10,3,8,5,2,7,10,9,4,5,6]

xDataTemp = []
xData = []
# xData = [[] for row in range(1000)]

numberTemp =[]
number = []
# number = [[] for row in range(1000)]
index=0;
# print "hello"
for line in rdr:
	date1.append(date(int(line[0].split('.')[0]),int(line[0].split('.')[1]),int(line[0].split('.')[2])))
	
	year.append(int(line[0].split('.')[0]) - 2002);
	month.append(int(line[0].split('.')[1]) + year[index]*12 - 12);
	gap = date1[index] - firstDate 
	day.append(gap.days)

	manseTemp = [(firstganzi[0] + year[index])%10,
				(firstzizi[0] + year[index])%12,
				(firstganzi[1] + month[index])%10,
				(firstzizi[1] + month[index])%12,
				(firstganzi[2] + day[index])%10,
				(firstzizi[2] + day[index])%12]
	manse.append(manseTemp)

	xDataTemp = [ganziNumber[(firstganzi[0] + year[index])%10],
				ziziNumber[(firstzizi[0] + year[index])%12],
				ganziNumber[(firstganzi[1] + month[index])%10],
				ziziNumber[(firstzizi[1] + month[index])%12],
				ganziNumber[(firstganzi[2] + day[index])%10],
				ziziNumber[(firstzizi[2] + day[index])%12]]
	xData.append(xDataTemp)
    
	numberTemp = []
	for k in range(45):
		numberTemp.insert(k,0)
		if(int(line[1])-1==k):
			numberTemp.insert(k,1)
	number.append(numberTemp);

    
	index = index+1

index = 0
f.close()   

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다.
W = tf.Variable(tf.random_uniform([6, 46], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
# 편향은 아웃풋의 갯수, 즉 최종 결과값의 분류 갯수인 3으로 설정합니다.
b = tf.Variable(tf.zeros([46]))

# 신경망에 가중치 W과 편향 b을 적용합니다
L = tf.add(tf.matmul(X, W), b)
# 가중치와 편향을 이용해 계산한 결과 값에
# 텐서플로우에서 기본적으로 제공하는 활성화 함수인 ReLU 함수를 적용합니다.
L = tf.nn.relu(L)

# 마지막으로 softmax 함수를 이용하여 출력값을 사용하기 쉽게 만듭니다
# softmax 함수는 다음처럼 결과값을 전체합이 1인 확률로 만들어주는 함수입니다.
# 예) [8.04, 2.76, -6.52] -> [0.53 0.24 0.23]
model = tf.nn.softmax(L)

# 신경망을 최적화하기 위한 비용 함수를 작성합니다.
# 각 개별 결과에 대한 합을 구한 뒤 평균을 내는 방식을 사용합니다.
# 전체 합이 아닌, 개별 결과를 구한 뒤 평균을 내는 방식을 사용하기 위해 axis 옵션을 사용합니다.
# axis 옵션이 없으면 -1.09 처럼 총합인 스칼라값으로 출력됩니다.
#        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
# 예) [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
#     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]
# 즉, 이것은 예측값과 실제값 사이의 확률 분포의 차이를 비용으로 계산한 것이며,
# 이것을 Cross-Entropy 라고 합니다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: xData, Y: number})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: xData, Y: number}))


#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: xData}))
print('실제값:', sess.run(target, feed_dict={Y: number}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: xData, Y: number}))


# plt.plot(cost, 'ro', label='Original data')
# plt.legend()
# plt.show()