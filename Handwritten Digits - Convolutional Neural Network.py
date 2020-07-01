### Create, train, and test a neural network that will tell you what number an image is
# CNN is good for image recognition
# CNN feeds in convolutions or "chunks" of an image
# Reasoning behind this is that CNN will pick up on certain patterns
# b/c looks at chunks instead of one pixel at a time (ex chunk may contain line)
# using MNIST (handwritten images) dataset with corresponding y value (what digit is)

#=====Gathering Data=====#
import keras
mnist = keras.datasets.mnist # getting mnist dataset from Keras
#60,000 images for training model (x_train, y_train)
#10,000 images for testing model (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

### Show what data looks like
import matplotlib.pyplot as plt
# display first test data image
plt.imshow(x_train[0])
plt.show()
# print corresponding y value of image (what the digit is, in this case it is 5)
print(y_train[0])



#=====Formatting Data=====#
### Break dataset up into digestable chunks for the NN to digest easier, reshaping imgs
# this way, we can reduce time and error (also not waste too much computing power)
# change data from shape (n, width, height) --> (n, depth, width, height)
# depth is 1 b/c it is a 2D image, can work with 3D images later
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape is shape going into NN = size of image
input_shape = (28, 28, 1)

### Normalizing image (values btwn 0-225) --> (values btwn 0-1)
# .astype('float32') changes the type of the array to a float (32 bit)
x_train = x_train.astype('float32')
# normalize image by dividing all values by 255 so now all values btwn 0-1
x_train /= 255
# .astype('float32') changes the type of the array to a float (32 bit)
x_test = x_test.astype('float32')
# normalize image by dividing all values by 255 so now all values btwn 0-1
x_test /= 255

### Converting y values to 1s and 0s to make them categorical variables (aka one-hot encoding)
# Categorical means sorting the data into categories that have no order (ex red and blue flowers)
# Without this the NN could assume that higher numbers are more important
# and not classify them based on how they looks
print(y_train[0])
# .to_categorical(y_train, #) there are 10 possible classes aka candidates aka categories (digits 0-9)
y_train = keras.utils.to_categorical(y_train, 10)
# .to_categorical(y_train, #) there are 10 possible classes (digits 0-9)
y_test = keras.utils.to_categorical(y_test, 10)
print(y_train[0])
# Now, the y value look like [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] instead of 5
# picks the highest value in the array to output instead of an average of similar-ranking digits
# ex) if digits 4 and 7 were ~same output, would pick the higher value instead of outputting the average (5.5)



#=====Creating model=====#
model = keras.models.Sequential()

### Create input layer - input layer is a 2D convolutional layer with 32 neurons with chunks of size 3x3 pixels
# activation function is relu, and input_shape is the input shape we defined earlier when breaking up data
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
# Create another convolutional layer with 64 neurons with chunks of size 3x3 pixels
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))

### Create a max pooling layer
# takes section of image - okly keeps max pixel value in a chunk of image
# this down-sampling reduces the complexity of the image and allows for the NN
# decreasing image "resolution" also helps prevent against overfitting
# to make assumtions about patterns and features in the regions (decrease img size, keep important features)
# pool_size is like the kernel_size, the pool size is 2x2 pixels
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

### Create dropout layer
# b/c many neurons in first two layers (32 and 64), chances are some of those neurons are
# not going to be very helpful/impactful to the model and could hurt accuracy and processing power
# This layer will randomly drop out neurons and see which ones don't need and also prevents model
# from overfitting to the training dataset
# set the probability of neuron getting dropped out to 25%
model.add(keras.layers.Dropout(0.25))

### Flatten image --> convert matrix to 1D array
# flatten an image if want to use a Dense (a fully connected layer) after convolution layers
# need to do this b/c Dense takes in 1D array/tensor (we are "unstacking" a multidimensional tensor to 1D)
# ? makes all in-between values to extreme 1s or 0s (think like no gray values, just black and white)
model.add(keras.layers.Flatten())

### Create a Dense layer with 128 neurons(a dense layer is a regular layer of neurons in a NN
# e/a neuron receives inputs from all neurons in previous layer)
# use a fully connected layer (where all neurons are connected to all neurons in previous layer vs. "sparse interactions" (https://www.youtube.com/watch?v=m8pOnJxOcqY))
# in order to learn non-linear combination of features (kind of like how Sebastian League explained his "hope" of the NN learning the patterns of features to an img)
model.add(keras.layers.Dense(128, activation='relu'))

### Create another dropout layer
# this dropout layer is for the previous Dense layer
# chances are, many of the 128 neurons will not be too helpful and will take up computing power
# and will also make the model overfit, so we create a dropout layer to address this issue
# also, there is a higher probability here (50%) because there are many more neurons, thus more are likely to be unhelpful
model.add(keras.layers.Dropout(0.5))

### Create output layer
# 10 output neurons for the 10 digits (0-9)
# softmax activation fxn turns value of each output neuron into a probability
# in short, softmax makes sure all probabilities from output add up to 1 in the end
# (normalizes probability of an image belonging to e/a group into the
# output vector (2D array mentioned earlier ex. [0.1, 0.3, 0.7, 0, 0, 0, 0, 0, 0, 0]))
# The activation fxn does this by returning  np.exp(x) / np.sum(np.exp(x), axis=0)
# the equation (in the line above) shows the (individual / total sum) to give a %age
# We use e^x in this activation fxn b/c larger values have much larger outputs compared to smaller values
# and this we can use to train our model better with more "concrete" "advice" to the network
# (as opposed to simple normalization, you will not get as large as a difference between the final values)
model.add(keras.layers.Dense(10, activation='softmax'))



#=====Training model=====#
### Compile model - defining loss function, optimizer, and metrics (what you are monitoring/looking at)
# using categorical loss fxn b/c image classification is a categorical problem
# a metric value is calculated after each batch
# use adam or adadelta optimizer (optimizer=keras.optimizers.Adadelta())
# use optimizer='adam' for better results (~99% accuracy with adam, ~89% accuracy with adadelta)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

### Training model
# batch size = # of training datapoints used in one iteration
# epochs = # times go through entire training dataset
# validation data = data use to check the model's accuracy on outside data it was not trained on
model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test))

# displaying loss and accuracy of the model on the test set
# model.evaluate outputs the metric value specified in model.compile (in this case it is accuracy)
# (model.evaluate runs through the data - predicts the output for inputs and computes metrics from the y_pred and y_true values)
# this is different than model.predict b/c model.predict only outputs the predicted output from a given input (it doesn't output any metrics)
print(model.evaluate(x_test, y_test)) # prints [loss, accuracy out of 1]
input("") # wait for user input before close shell
