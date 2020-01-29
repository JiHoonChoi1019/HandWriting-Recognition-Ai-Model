import tensorflow as tf
import os
import matplotlib.pyplot as plt

# MNIST 데이터 적재
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 경고 메시지 화면출력 금지

#파라미터
learning_rate = 0.001 #신경망 학습률
training_epochs = 100 # 학습 횟수 (epoch)
batch_size = 100 # 미니배치의 크기
display_step = 10 # 중간결과 출력 간격

#신경망 구조 관련 파라미터
n_hidden_1 = 15 # 은닉층의 노드 개수
n_input = 784 #입력층의 노드 개수 MNIST 데이터 (28x28)
n_classes = 10 # 출력층의 노드 수 MNIST 부류 개수(숫자 0~9)

#텐서 그래프 입력 변수
x = tf.placeholder("float", [None, n_input]) #입력 : 필기체 영상
y = tf.placeholder("float", [None, n_classes]) #출력 : 숫자

#학습모델 MLP정의
def multilayer_perceptron(x, weights, biases):
    #ReLu를 사용하는 은닉층
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #출력층 (활성화 함수 미사용)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

#학습할 파라미터: 가중치(weights), 편차항(biases)
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#신경망 모델 구성, 출력값 pred : 입력 x에 대한 신경망의 출력
pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope("cost"):
    #비용(오차) 정의 (신경망 출력 pred, 목표 출력 y): 교차 엔트로피 사용
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #학습 알고리즘 설정
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope("accuracy"):
    #모델 테스트 : out의 최대값 노드와 y 노드가 같으면 정답
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #correct_prediction 평균
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)

    
init = tf.global_variables_initializer() #변수 초기화 지정

#데이터 플로우 그래프 실행
with tf.Session() as sess:
    print("Start....")
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)  # for 0.8
    merged = tf.summary.merge_all()
    
    sess.run(init)
    summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    total_batch = int(mnist.train.num_examples / batch_size) #배치 개수

    for epoch in range(training_epochs): #정해진 횟수 만큼 학습
        avg_cost = 0.
        for i in range(total_batch): #미니 배치
            batch_x, batch_y = mnist.train.next_batch(batch_size) #적재
            #역전파 알고리즘 적용
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch #평균 손실(오류) 계산

            if epoch %display_step == 0: #현재 학습 상황 출력
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, i)
    
    print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("End...")
    


