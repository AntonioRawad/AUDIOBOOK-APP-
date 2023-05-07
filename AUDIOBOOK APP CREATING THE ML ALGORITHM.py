#!/usr/bin/env python
# coding: utf-8

# FROM PREVOUSE nalysis of the case , i add here the intro "You are given data from an Audiobook app. Logically, it relates only to the audio versions of books. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company. we have Preprocess the data. Balance the dataset. Create 3 datasets: training, validation, and test. Save the newly created sets in a tensor friendly format (e.g. *.npz)
# "

# ## now we proceed with creating the Machine Learning Algorithm 

# ## import libreries 

# In[2]:


# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf


# In[3]:


# let's create a temporary variable npz, where we will store each of the three Audiobooks datasets
npz = np.load('Audiobooks_data_train.npz')

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that

train_inputs = npz['inputs'].astype(float)

# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)

train_targets = npz['targets'].astype(int)

# we load the validation data in the temporary variable

npz = np.load('Audiobooks_data_validation.npz')

# we can load the inputs and the targets in the same line

validation_inputs, validation_targets = npz['inputs'].astype(float), npz['targets'].astype(int)

# we load the test data in the temporary variable

npz = np.load('Audiobooks_data_test.npz')

# we create 2 variables that will contain the test inputs and the test targets

test_inputs, test_targets = npz['inputs'].astype(float), npz['targets'].astype(int)


# # we proceed with the ### Model

# 
# Outline, optimizers, loss, early stopping and training

# In[7]:


# Set the input and output sizes
input_size = 10
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 50
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 100


# fit the model
# note that this time the train, validation and test data are not iterable
#model.fit(train_inputs, # train inputs
          #train_targets, # train targets
          #batch_size=batch_size, # batch size
          #epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          #validation_data=(validation_inputs, validation_targets), # validation data
          #verbose = 2 # making sure we get enough information about the training process
         # )  


# # set an early stopping mechanism

# Early stopping is a technique used in machine learning to prevent overfitting and improve the generalization performance of a model. It works by monitoring the performance of the model on a validation set during the training process.
# 
# Specifically, early stopping stops the training process if the performance of the model on the validation set stops improving. The improvement is measured based on a chosen metric such as accuracy, loss, or F1 score.
# 
# By stopping the training process early, before the model starts to overfit to the training data, early stopping helps to prevent the model from becoming too complex and to improve its ability to generalize to new, unseen data

# ### we introduce a new method called CALLBACKS 

# Callbacks in machine learning are functions that can be called during the training process at certain points or intervals. These functions are called to perform some specific actions during the training process, such as saving the weights of a model at a certain point, adjusting the learning rate during training, or stopping the training early if some condition is met.
# 
# Callbacks are useful for monitoring the training process and for improving the performance of the model. They can help to prevent overfitting, reduce training time, and improve the accuracy of the model.
# 
# Some common callbacks used in deep learning include EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and TensorBoard. EarlyStopping is used to stop the training process if the validation loss does not improve after a certain number of epochs. ModelCheckpoint is used to save the weights of the model at certain intervals during training. ReduceLROnPlateau is used to reduce the learning rate if the validation loss does not improve after a certain number of epochs. TensorBoard is used to visualize the training process and to monitor the performance of the model.

# In[12]:


# set an early stopping mechanism
# let's set patience=5, to be a bit tolerant against random validation loss increases

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )  


# ## Test the model
# 
# As we discussed in the lectures, after training on the training data and validating on the validation data, we test the final prediction power of our model by running it on the test dataset that the algorithm has NEVER seen before.
# 
# It is very important to realize that fiddling with the hyperparameters overfits the validation dataset. 
# 
# The test is the absolute final instance. You should not test before you are completely done with adjusting your model.
# 
# If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.

# In[13]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


# In[14]:


print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# Using the initial model and hyperparameters given in this notebook, the final test accuracy should be roughly around 91%.
# 
# Note that each time the code is rerun, we get a different accuracy because each training is different. 
# 
# We have intentionally reached a suboptimal solution, so you can have space to build on it!

# ### FINAL NOTE 
# Based on the analysis and implementation of machine learning algorithms, we can conclude that predicting whether a customer will buy again from an Audiobook company is a challenging classification problem. Our model achieved a test accuracy of 79.91%, which is suboptimal compared to the expected accuracy of around 91%. This could be due to various reasons such as inadequate feature engineering, insufficient data, or suboptimal hyperparameters. However, we can still leverage the insights gained from our model to identify the most important metrics for a customer to come back again, which can lead to great savings by focusing efforts on customers that are likely to convert again. Further exploration of the data and refining the machine learning algorithms could potentially improve the accuracy of our model and provide more actionable insights for the Audiobook company.

# In[ ]:





# In[ ]:




