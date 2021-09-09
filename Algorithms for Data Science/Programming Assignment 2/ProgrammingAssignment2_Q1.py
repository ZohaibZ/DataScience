import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.utils import np_utils, plot_model


# https://keras.io/guides/sequential_model/
# https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-cmodellassification/

def feedForward(X,y,epoch, batch_size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Normalize the images.
    train_images = X_train.astype('float32')
    test_images = X_test.astype('float32')
    train_images /= 255
    test_images /= 255

    train_labels = np_utils.to_categorical(Y_train, num_classes=10, dtype='uint8')
    test_labels = np_utils.to_categorical(Y_test, num_classes=10, dtype='uint8')

    # Build the model
    model = Sequential()
    model.add(Dense(28, activation='relu',input_shape=(784,)))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(10, activation='softmax', name="Layer"))

    # Compile the model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    model.layers
    plot_model(model, to_file='multilayer_graph.png')

    # extracting the features from the last layer of the network
    feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer(name="Layer").output)
    # Call feature extractor on test input.
    x = test_images
    print("x_shape: \n", test_images.shape)
    print("x : \n", test_images)
    features = feature_extractor.predict(x)
    print("features shape: \n", features.shape)
    print("features: \n", features)
    

    X = np.arange(10)
    plt.figure()
    index = 0
    while index<100: 
        plt.scatter(X , features[index])
        index+=1

    plt.show()

    # Train the model
    model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, verbose=0)

    # Evaluate the model
    results = model.evaluate(test_images, test_labels, verbose=0)
    print("test loss, test acc: \n", results)

    # Save the model to disk
    model.save_weights('model.h1')

    # Load the model from disk later using:
    # model.load_weights('model.h1')

    # Predict on the first 5 test images.
    preds = np.argmax(model.predict(test_images[0:10]), axis=1)
    print("Prediction Shape: \n", preds.shape)

    # Print our model's predictions.
    print("prediction: \n", preds) 

    # Check our predictions against the ground truths.
    print("test_labels shape: \n", test_labels[0:10].shape) 
    print("test_labels: \n", test_labels[0:10])
    print("test_labels argmax: \n", np.argmax(test_labels[0:10],axis=1))
    print("\nAccuracy on Test Data: ", accuracy_score(np.argmax(test_labels[0:10], axis=1), preds))
    print("\nNumber of correctly identified images: ", accuracy_score(np.argmax(test_labels[0:10], axis=1), preds, normalize=False),"\n")
    confusion_matrix(np.argmax(test_labels[0:10], axis=1), preds, labels=range(0,9))

    return model 


if __name__ == "__main__":
    mnist = pd.read_csv("train.csv",header=0)
    mnist = mnist.to_numpy()
    #1000 images
    X = mnist[0:1000,1:len(mnist[0])] 
    y = mnist[0:1000,0] #1000 images
    
    m = feedForward(X,y, 256, 28)
    # layers = m.layers[0].get_weights()
    # print("len(layer1[0]): ",len(layers[1]))
