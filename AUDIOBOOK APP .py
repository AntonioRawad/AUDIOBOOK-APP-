#!/usr/bin/env python
# coding: utf-8

# ### Problem
# 
# You are given data from an Audiobook App. Logically, it relates to the audio versions of books ONLY. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.
# 
# The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertising to him/her. If we can focus our efforts SOLELY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.
# 
# You have a .csv summarizing the data. There are several variables: Customer ID, ), Book length overall (sum of the minute length of all purchases), Book length avg (average length in minutes of all purchases), Price paid_overall (sum of all purchases) ,Price Paid avg (average of all purchases), Review (a Boolean variable whether the customer left a review), Review out of 10 (if the customer left a review, his/her review out of 10, Total minutes listened, Completion (from 0 to 1), Support requests (number of support requests; everything from forgotten password to assistance for using the App), and Last visited minus purchase date (in days).
# 
# These are the inputs (excluding customer ID, as it is completely arbitrary. It's more like a name, than a number).
# 
# The targets are a Boolean variable (0 or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information. 
# 
# The task is simple: create a machine learning algorithm, which is able to predict if a customer will buy again. 
# 
# This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s. 
# 

# ## Preprocess the data. Balance the dataset. Create 3 datasets: training, validation, and test. Save the newly created sets in a tensor friendly format (e.g. *.npz)
# 
# Since we are dealing with real life data, we will need to preprocess it a bit. This is the relevant code, which is not that hard, but is crucial to creating a good model.
# 
# If you want to know how to do that, go through the code with comments. In any case, this should do the trick for most datasets organized in the way: many inputs, and then 1 cell containing the targets (supersized learning datasets). Keep in mind that a specific problem may require additional preprocessing.
# 
# Note that we have removed the header row, which contains the names of the categories. We simply want the data.

# In[1]:


import numpy as np

# We will use the sklearn preprocessing library, as it will be easier to standardize the data.
from sklearn import preprocessing


# ### Extract the data from the csv

# In[2]:


raw_csv_data = np.loadtxt(r'C:\Users\rawad\OneDrive\Desktop\aws Restart course\Udemy Data Science Course\exercise\Audiobooks_data.csv', delimiter=',')


# In[3]:


# The inputs are all columns in the csv, except for the first one [:,0]
# (which is just the arbitrary customer IDs that bear no useful information),
# and the last one [:,-1] (which is our targets)


# In[4]:


unscaled_inputs_all = raw_csv_data[:,1:-1]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:,-1]


# ### Balance the dataset

# In[5]:


# Count how many targets are 1 (meaning that the customer did convert)

num_one_targets = int(np.sum(targets_all))

# Set a counter for targets that are 0 (meaning that the customer did not convert)
zero_targets_counter = 0

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# Declare a variable that will do that:

indices_to_remove = []

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)


# ### we have a balanced data set 

# ### Standardize the inputs

# In[6]:


# That's the only place we use sklearn functionality. We will take advantage of its preprocessing capabilities
# It's a simple line of code, which standardizes the inputs, as we explained in one of the lectures.
# At the end of the business case, you can try to run the algorithm WITHOUT this line of code. 
# The result will be interesting.
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)


# ### a little trick is to Shuffle the data(inputs and targets )

# In[7]:


# When the data was collected it was actually arranged by date
# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.
# Since we will be batching, we want the data to be as randomly spread out as possible

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# ### Split the dataset into train, validation, and test

# In[8]:


# Count the total number of samples

samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.

train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.

test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)


# ### Save the three datasets in *.npz

# In[9]:


# Save the three datasets in *.npz.
# In the next lesson, you will see that it is extremely valuable to name them in such a coherent way!

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)


# # now we proceed with creating the Machine Learning Algorithm 

# ## import libreries 

# In[10]:


# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf


# In[11]:


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


# # we proceed with the Model 

# In[12]:


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


# ## set an early stopping mechanism

# ## Early stopping 
# is a technique used in machine learning to prevent overfitting and improve the generalization performance of a model. It works by monitoring the performance of the model on a validation set during the training process.
# 
# Specifically, early stopping stops the training process if the performance of the model on the validation set stops improving. The improvement is measured based on a chosen metric such as accuracy, loss, or F1 score.
# 
# By stopping the training process early, before the model starts to overfit to the training data, early stopping helps to prevent the model from becoming too complex and to improve its ability to generalize to new, unseen data
# 
# ## we introduce a new method called CALLBACKS
# Callbacks in machine learning are functions that can be called during the training process at certain points or intervals. These functions are called to perform some specific actions during the training process, such as saving the weights of a model at a certain point, adjusting the learning rate during training, or stopping the training early if some condition is met.
# 
# Callbacks are useful for monitoring the training process and for improving the performance of the model. They can help to prevent overfitting, reduce training time, and improve the accuracy of the model.
# 
# Some common callbacks used in deep learning include EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and TensorBoard. EarlyStopping is used to stop the training process if the validation loss does not improve after a certain number of epochs. ModelCheckpoint is used to save the weights of the model at certain intervals during training. ReduceLROnPlateau is used to reduce the learning rate if the validation loss does not improve after a certain number of epochs. TensorBoard is used to visualize the training process and to monitor the performance of the model.

# In[13]:


# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

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
# As we discussed in the lectures, after training on the training data and validating on the validation data, we test the final prediction power of our model by running it on the test dataset that the algorithm has NEVER seen before.
# 
# It is very important to realize that fiddling with the hyperparameters overfits the validation dataset.
# 
# The test is the absolute final instance. You should not test before you are completely done with adjusting your model.
# 
# If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.

# In[19]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


# In[18]:


print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# ## conclusion 
# Based on the provided code snippet, it appears that a neural network model has been trained on some data, and its performance has been evaluated on a test set. The model achieved a test accuracy of 80.58%, indicating that it is able to predict whether a customer will buy again with a reasonable degree of accuracy.
# 
# The training process involved setting an early stopping mechanism to prevent overfitting, and the model was trained for a maximum of 100 epochs. It is important to note that the results obtained from this particular model are specific to the data used for training and testing, and may not necessarily generalize well to other datasets or real-world scenarios. Further experimentation and testing would be necessary to determine the model's performance in other contexts.

# In[ ]:




