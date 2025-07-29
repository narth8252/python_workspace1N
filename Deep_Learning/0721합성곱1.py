#250721 AM10시 딥러닝-CNN 쌤PPT400p (250717딥러닝종합_백현숙)
#하고나면 경고뜨는거 무시해라
# from sklearn import metrics
# from tensorflow.keras.datasets import fashion_mnist

# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# from keras import models, layers

# #CNN이더라도 반드시 스케일링필요, 차원은그대로
# img_height=28
# img_width=28
# network = models.Sequential(
#     [
#         layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
#         layers.Conv2D(32, (3,3), activation='relu'),
#     #            출력값32, 필터크기보통이값3,3
#         layers.Conv2D(32, (3,3), activation='relu'),
#         layers.MaxPooling((2,2layers.Rescaling), #서브샘플링,특성(feature)개수줄여서 과대적합방지

#         #CNN과완전연결망 연결하기위한 계층 4D → 2D로 변경필요
#         layers.Flatten(),
#         #완전연결망
#         layers.Dense(128, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax') #마지막출력층
#     ]
# )

# import tensorflow as tf
# network.compile(optimizer='adam',
#                 loss=tf.keras.SparseCategoricalCrossentropy(), #정수형라벨일때
#                 metrics = ['accuracy'])
# print(network.summary())

# network.fit(X_train, y_train, epochs=10, validation_split=0.2)
# print("훈련셋", network.evaluate(X_train, y_train))
# print("테스트셋", network.evaluate(X_test, y_test))


from sklearn import metrics
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models, layers
import tensorflow as tf

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 차원 추가 (채널)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

img_height=28
img_width=28
network = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

network.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print(network.summary())
network.fit(X_train, y_train, epochs=10, validation_split=0.2)
print("훈련셋", network.evaluate(X_train, y_train))
print("테스트셋", network.evaluate(X_test, y_test))

# 출력결과 Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  rescaling (Rescaling)       (None, 28, 28, 1)         0
#  conv2d (Conv2D)             (None, 26, 26, 32)        320
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248
#  max_pooling2d (MaxPooling2D  (None, 12, 12, 32)       0
#  )
#  flatten (Flatten)           (None, 4608)              0
#  dense (Dense)               (None, 128)               589952
#  dense_1 (Dense)             (None, 64)                8256
#  dense_1 (Dense)             (None, 64)                8256
#  dense_2 (Dense)             (None, 10)                650
# =================================================================
# Total params: 608,426
# Trainable params: 608,426
# Non-trainable params: 0
# _________________________________________________________________
# None
# Epoch 1/10
# Epoch 1/10
# 1500/1500 [==============================] - 38s 24ms/step - loss: 0.4251 - accuracy: 0.8451 - val_loss: 0.2981 - val_accuracy: 0.8920
# Epoch 2/10
# 1500/1500 [==============================] - 37s 24ms/step - loss: 0.2653 - accuracy: 0.9026 - val_loss: 0.2598 - val_accuracy: 0.9067
# ...
# Epoch 9/10
# 1500/1500 [==============================] - 43s 28ms/step - loss: 0.0571 - accuracy: 0.9784 - val_loss: 0.3181 - val_accuracy: 0.9233
# Epoch 10/10
# 1500/1500 [==============================] - 43s 28ms/step - loss: 0.0424 - accuracy: 0.9844 - val_loss: 0.3653 - val_accuracy: 0.9177
# 1875/1875 [==============================] - 19s 10ms/step - loss: 0.0957 - accuracy: 0.9756
# 훈련셋 [0.09573294222354889, 0.9756333231925964]
# 313/313 [==============================] - 5s 15ms/step - loss: 0.3861 - accuracy: 0.9153
# 테스트셋 [0.3861454725265503, 0.9153000116348267] #과대적합