# # import libraries that we need
import os
import cv2  # loading and processing image
import numpy as np  # using numpy arrays
import matplotlib.pyplot as plt  # using for visualization (optional)
import tensorflow as tf  # important for machine learning

mnist = tf.keras.datasets.mnist  # handwritten digit database
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# limit the value 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # input layer
# model.add(tf.keras.layers.Dense(784, activation='relu')) # hidden layer
# model.add(tf.keras.layers.Dense(150, activation='relu'))# hidden layer
# model.add(tf.keras.layers.Dense(97, activation='relu'))# hidden layer
# model.add(tf.keras.layers.Dense(10, activation='softmax')) # output layer 0-9
# #
# model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])
# #
# # # # training
# model.fit(x_train, y_train, epochs=50)
#
# # save model
# model.save('handwriting.model')

model = tf.keras.models.load_model('handwriting.model')
#
# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f"/Users/yenphglam/Documents/MachineLearning/digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"/Users/yenphglam/Documents/MachineLearning/digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
