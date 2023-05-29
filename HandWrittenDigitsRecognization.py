import tensorflow as tf
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000, 28, 28, 1))
X_train = X_train.astype('float32') / 255.0

X_test = X_test.reshape((10000, 28, 28, 1))
X_test = X_test.astype('float32') / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)


testLOSS, testACC = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy=', testACC)

