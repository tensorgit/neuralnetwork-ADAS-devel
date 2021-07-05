from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
#import joblib

## Import any additional libraries you need to define a model
import tensorflow as tf

# Provided model load function
#def model_fn(model_dir):


## The main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None, low_memory=False, skiprows=1, skipinitialspace=True)

    # Measured values are in the last column
    train_x = train_data.iloc[:,:15].values
    train_y = train_data.iloc[:,-1].values
    

    ## Define a model 
    model = tf.keras.models.Sequential()
    
    # Adding the input layer and the first hidden layer
    units = 25  # no. of hidden neurons 
    activation  = 'relu' # activation function 'rectifier'
    model.add(tf.keras.layers.Dense(units=units, activation=activation))
    
    # Adding the second hidden layer
    units = 25  # no. of hidden neurons
    activation  = 'relu' 
    model.add(tf.keras.layers.Dense(units=units, activation=activation))
    
    # Adding the third hidden layer
    units = 25  # no. of hidden neurons
    activation  = 'relu' 
    model.add(tf.keras.layers.Dense(units=units, activation=activation))
    
    # Adding the fourth hidden layer
    units = 25  # no. of hidden neurons
    activation  = 'relu' 
    model.add(tf.keras.layers.Dense(units=units, activation=activation))
    
    # Adding the output layer
    units = 1 # the no. of output neurons is just 1 here coz the output is binary 
    model.add(tf.keras.layers.Dense(units=units))
    
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    
    ## Train the model
    batch_size =  32
    epochs = 200
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=2)
    

    # Save the trained model
    #model.save("model_ann_1", save_format="h5")
    model.save(os.path.join(args.model_dir, 'model_ann'), 'my_model.h5')
    print("model saved")
    print(model.summary())
