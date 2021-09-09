import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Problem solved using the links below as refernce
# https://www.tensorflow.org/guide/keras/train_and_evaluate

def feedForward(X,y,epoch, batch_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Normalize the images.
    train_images = X_train.astype('float32')
    test_images = X_test.astype('float32')
    train_images /= 255
    test_images /= 255

    train_labels = np_utils.to_categorical(Y_train, 10)
    test_labels = np_utils.to_categorical(Y_test, 10)

    # Build the model
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(784,)))
    # model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, verbose=0)

    # Evaluate the model
    results = model.evaluate(test_images, test_labels, verbose=0)
    print("test loss, test acc:", results)

    # Save the model to disk
    model.save_weights('model.h1')

    # Load the model from disk later using:
    # model.load_weights('model.h1')

    # Predict on the first 5 test images.
    predictions1 = model.predict(test_images[:10])

    # Print our model's predictions.
    print("prediction: ", np.argmax(predictions1, axis=1)) 

    # Check our predictions against the ground truths.
    print(test_labels[:10]) 
    return model 

def ConvolutionNN(X,y,epoch,batch_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    # Normalize and reshape the images.
    train_images = X_train.astype('float32')
    test_images = X_test.astype('float32')
    train_images /= 255
    test_images /= 255

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    train_labels = np_utils.to_categorical(Y_train, 10)
    test_labels = np_utils.to_categorical(Y_test, 10)

    # Define model architecture
    model = Sequential()
    
    model.add(Convolution2D(12, (3,3), activation='relu', input_shape=(28,28,1)))
    # model.add(Convolution2D(12, 3,3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Fit model on training data
    model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, verbose=0)

    # Evaluate the model
    results = model.evaluate(test_images, test_labels, verbose=0)
    print("test loss, test acc:", results)

    # Save the model to disk
    model.save_weights('model.h2')

    # Load the model from disk later using:
    # model.load_weights('model.h2')

    # Predict on the first 5 test images.
    predictions = model.predict(test_images[:10])

    # Print our model's predictions.
    print("prediction: ", np.argmax(predictions, axis=1)) 

    # Check our predictions against the ground truths.
    print(test_labels[:10]) 

    return model

if __name__ == "__main__":
    mnist = pd.read_csv("train.csv",header=0)
    mnist = mnist.to_numpy()
    X = mnist[0:1000,1:len(mnist[0])]
    y = mnist[0:1000,0]

    # m1 = feedForward(X,y, 1, 10)
    # layer1 = m1.layers[0].get_weights()
    # layer2 = m1.layers[1].get_weights()
    # print("len(layer1[0]): ",len(layer1[0]))
    # print("len(layer2[0]): ",len(layer2[0]))
    
    m2 = ConvolutionNN(X,y, 1, 10)
    layer1 = m2.layers[0].get_weights()
    print("len(layer1[0]): ",len(layer1[1]))
