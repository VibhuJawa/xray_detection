# Classify and localize Thoracic diseases using Deep Learning

We developed a system for finding and localizing thoracic diseases in X-rays using deep learining and computer vision techniques. Then we attempted to highlight the progression of a disease over the multiple visits of a patient

## Project Report
[Project Report](CV_Final_Report.pdf)


## Data Set Details
This code analyses data set [Data Set Analysis](Data_Analysis_of_Chest_Xray.ipynb)

## Main Model
This is the training model. [Model Training Code](Main_Model.ipynb)

### Model Training Steps:
* Load Data
* Img and CSV data loaders
* Data split
* Preprocess
* Augmentation trials 
* Train model
* Display results
* Localization enhancements

## Heat Map Generation
This code generate heat maps. [HeatMapGenerator.py](HeatMapGenerator.py)

## 	Comparing Heat Maps
This code compares heat maps and normal images. [CompareImages_Consolidated.py](CompareImages_Consolidated.py)


