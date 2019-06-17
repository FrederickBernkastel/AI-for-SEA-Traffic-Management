# Overview
## Description
This project tackles Grab's 2019 AI for SEA challenge - Traffic Management, and aims to forecast demand ahead by 5 time intervals given data from the past 14 days.


## Official Problem Statement
Economies in Southeast Asia are turning to AI to solve traffic congestion, which hinders mobility and economic growth. The first step in the push towards alleviating traffic congestion is to understand travel demand and travel patterns within the city.

Can we accurately forecast travel demand based on historical Grab bookings to predict areas and times with high travel demand?

# Usage
## Training
To train a new model, place the new demand data csv file in the /Data folder and change the path variable “DATA_PATH” in Trainer.ipynb” before running.

Pre-trained models on the provided competition data has been included, thus it is not necessary to retrain the model for prediction.

## Prediction
Before running “Predictor.ipynb”, ensure that the following model parameter files are present in the /Model folder 
* latt_to_idx_dic.pkl
* long_to_idx_dic.pkl
* zero_demand.npy
* all_meta_features.pkl
* xgb_model.model
* lgb_model.model

To predict on new data, place the input data csv file in the /Data folder and change the path variable “DATA_PATH” in “Predictor.ipynb” to target that file before running. The predicted demand data will be saved to the /Predict/output.csv, although the path can be changed as needed using the "OUTPUT_CSV_PATH" variable.

Predicted demand data for the next 5 time periods after the last period is in the same format as the provided train data. If demand is predicted to be 0 for a given time period & geolocation, that time period & geolocation entry will be omitted from the csv file.

To prevent cheating, it is assumed that any scripts for evaluating results will be provided by Grab organizers, thus no code for evaluation is provided.

## Dependencies
The following python libraries and their versions were used for this project
* Geohash==1.0
* numpy==1.15.4
* pandas==0.23.4
* xgboost==0.90
* lightgbm==2.2.3

# Documentation
## Feature Engineering
### _Lag Variables_
Features for forecasting a time period ‘T’ include demand for the last 7 time periods (T-1 to T-7), to provide context about the current trend.

Recent trends in the same time period for the last 4 days were also included, alongside the previous 3 time periods and the next 3 time periods, to provide information about recent trends.
Finally, trends in the same time period and same day for the previous week were included, alongside the previous 3 time periods and next 4 time periods.

Note that the full 14 days were not used as input features. This is done in order to provide the models with around 7 additional days of training data, which is approximately 1 million samples.

Further processing of lag variables through simple / exponential moving averages reduced the model’s performance on the validation dataset, hence they are not included.

### _Lag Aggregated Variables_
Demand for all locations was aggregated throughout the 96 time periods in a day, and each time period was separately normalized to 0-1 for each period. This represents the total normalized demand in a given time period across all locations, which gauges the demand for the entire region relative to the typical maximum in that time period.
Aggregated demand was then used as lag variables, which includes the last 4 time periods (T-1 to T-4), and the previous to next time periods 7 days ago (T-7D-1 to T-7D+1).

### _Geospatial Information_
Normalized latitude and longitude information were used as features. While this information is typically not used directly as inputs to models as distance from an origin point is not correlated with demand, the model employed uses decision trees. The inclusion of this feature allows a few decision tree nodes to separately model outlier locations, clustered by proximity if necessary, and boosted the overall performance of the model on the validation set.

![Aggregrated Demand](./Images/aggregrated_demand.png?raw=true "Aggregrated Demand")

As seen above, trends in demand are correlated with the aggregated demand in each location. Thus, the normalized values of aggregated demand - which is constant between time periods for each location - were also used as features. This allows the model to further cluster locations with similar trends tegoether

Further processing of geospatial information through 1D / 2D gaussian filters on lagged demand reduced the model’s performance on the validation dataset, hence they are not included.

### _Cyclical Information_
Cyclical information refers to the daily time period from 0:0 to 23:45 and indexed from 0 to 95, and the weekdays indexed from 0 to 6.

To preserve the cyclical relationship between the start and end of time cycles, the indexes are first normalized to 0 - 2π, then projected onto a circle of radius 1 using sin() and cos().
An example of projecting weekdays onto a circle is shown below, with Sunday representing index 0 in the example. (No assumptions of Sunday being index 0 were made in the model.)


![Cyclical Features](./Images/cyclical_features.png?raw=true "Cyclical Features")

This allows the decision trees to group close time periods together for better generalization.

## Models
A simple averaged ensemble of XGBoost and LightGBM was used for prediction. Both XGBoost and LightGBM were trained on the full dataset provided.

LightBGM differs slightly from XGBoost during training in that a larger learning rate was first used for a few rounds in order to speed up training. This is done because LightGBM was unable to allocate sufficient GPU space for the bins, and training was instead done on the CPU, resulting in a much longer training time than XGBoost - which is trained on the GPU.

### _XGBoost Params_

| Param Name | Value |
| --- | --- |
| Tree Method | gpu\_hist |
| Maximum Depth | 6 |
| Number of Estimators | 1000 |
| Maximum Number of Bins | 2048 |
| Base Score (Global bias) | 0.052 (≈Mean of Demand) |
| Learning Rate | 0.05 |
| Loss Function | Squared Error |

### _LightGBM Params_
| Param Name | Value |
| --- | --- |
| Boosting Type | lgbt |
| Num Leaves | 40 |
| Min Data in Leaf | 50 |
| Maximum Number of Bins | 2048 |
| Min Split Gain | 0.001 |
| Subsample for Bin | 5000 |
| Early Stopping Rounds | 20 |
| Learning Rate 1 | 0.1 |
| Learning Rate 1 Max Rounds | 1000 |
| Learning Rate 2 | 0.05 |
| Learning Rate 2 Max Rounds | 500 |
| Loss Function | RMSE |
| L2 Regularizer | 1 |

### _Validation_
To select model parameters and input features, the data was split into 80% for training and 20% for validation, with the most recent 20% used for validation. The final configuration of model parameters achieved 0.02306 RMSE on the validation set for predicting the next time period.
The final model provided is trained on the full data.

# Future Work
The graph below shows the predicted and actual values of a geolocation with high demand for the first 7 days in the training data.

![High Demand Prediction](./Images/high_demand_prediction.png?raw=true "High Demand Prediction")

As seen from the graph, the predicted demand tends to be lower than 1, not higher, when the actual demand plateaus at 1. This is caused by RMSE penalities towards a slightly higher demand value, whereas slighly lower demand values are rewarded occassionally when demand dips slightly from its peak.

It is likely that the model will perform better in these regions if the loss function was tweaked to either remove penalties from slightly exceeding 0 and 1, or if values which are slightly above 1 or below 0 are clipped to 1 and 0 respectively before applying RMSE.
