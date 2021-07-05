# Capstone Project README
## Jasvir Dhillon. June 2021.
Title: Predicting autonomous vehicle straight line collision attributes with detected traffic object and recommending vehicle features to prevent collision
### Project overview

This machine learning project has successfully implemented the following steps
- 	Data collection (CarMaker simulations) and retrieving
- 	Data pre-processing (edgecase_data_processing.ipynb)
•	Cleaning and transforming
•	Data analysis and insights
•	PCA (PCA.ipynb)
- Training a neural network on input data (Training_prediction.ipynb)
•	Evaluating and refining the model
- Injecting new data and retraining the model (edgecase_newdata_processing.ipynb)
- Evaluating the final model (Visualizing_model_predictions.ipynb)
•	Accurately predict collision possibility
•	Recommend feature values to avoid collision  

All development work has been conducted with AWS. Data analysis has revealed interesting relationships between the various feature and output variables. The final refined neural network model is a much-improved version of the benchmark model with remarkably high accuracy, precision and recall (the metrics) proven on test and validation data

> /source_tfkeras/: includes the train.py training script for the neural network model used by the AWS SageMaker estimator

> /data/: contains training and test files along with dataframe files stored for later use on demand

> Visualizing_model_predictions.html: Contains investigative visualization plots with model predictions

#### Libraries used
- AWS SageMaker, S3, CloudWatch
- glob
- natsort
- StringIO
- pandas
- SKLearn
- pickle
- MXNet
- keras from TensorFlow
- seaborn
- matplotlib 
- plotly

### Dataset information
The original [dataset][link] for the Capstone project is available for download in a .zip format  

It includes 3 folders 
- case-LIDAR-1 (64,806 files) 
- case-LIDAR-2 (64,800 files)
- case-LIDAR-3 (64,800 files)

Total no. of files: 194,406

1. Each folder includes simulation results performed with the same ego vehicle LIDAR sensor setting (update frequency). Hence, a total of 3 different sensor settings
2. Each folder contains simulation result data for different test variations for the CarMaker software. 
Each test variation has two corresponding data files:
-       A '.dat' file which provides 3 channels: 'Time', 'Collision' and 'Distance to collision'. The required data has been logged only for the last few sections of the simulation where necessary, to minimize file storage space
-       A '.dat.info' file which contains the unique feature values for that particular variation

The updated [dataset][link2] is also available to download in a .zip format here

It includes 2 folders
- new_edge-case_dataset/post_case-LIDAR-2 (1,296 files)
- new_edge-case_dataset/post_case-LIDAR-3 (1,296 files)

[link]: <https://www.dropbox.com/s/j9jrkkaptkygsb5/edge-case_dataset.zip?dl=0>
[link2]: <https://www.dropbox.com/s/hz0kia92aiwawan/new_edge-case_dataset.zip?dl=0>
