### Let's run a simple image classifier, using the CIFAR 10 dataset of 10 image categories

import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10

image_row, image_height, image_depth = 32,32,3


#load a trained model or search in the web and use a trained model , upload it , and call it 
classifier = load_model('C:/Users/hp/Desktop/imkab/MY_DeepLearning_Projects/keras_cifar10_trained_model.h5')

#load the CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

color = True
scale = 8

def Drawing(name, res, input_image, scale, image_row, image_height):
    BLACK = [0,0,0]
    res = int(res)
    if res == 0:
        predicted = "AirPlane"
    if res == 1:
        predicted = "Automobile"
    if res == 2:
        predicted = "bird"
    if res == 3:
        predicted = "cat"
    if res == 4:
        predicted = "deer"
    if res == 5:
        predicted = "dog"
    if res == 6:
        predicted = "Frog"
    if res == 7:
        predicted = "Horse"
    if res == 8:
        predicted = "Ship"
    if res == 9:
        predicted = "Truck"
    expanded_image = cv2.copyMakeBorder(input_image,0,0,0, imageL.shape[0]*2, cv2.BORDER_CONSTANT, value = BLACK)

    if color == False:
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(predicted), (300,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (188, 200,19), 2)
    cv2.imshow(name, expanded_image)
    
    
    
for i in range (0,10):
    random = np.random.randint(0, len(x_test))
    image_input = x_test[random]
    imageL = cv2.resize(image_input, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    image_input = image_input.reshape(1, image_row, image_height, image_depth)
    
    #now let's test the prediction
    res = str(classifier.predict_classes(image_input, 1, verbose = 0)[0])
    Drawing("My Prediction says : ",res, imageL, scale, image_row, image_height)
    cv2.waitKey(0)
cv2.destroyAllWindows()    


