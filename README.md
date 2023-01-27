# DIABETES_PREICTION USING MACHINE LEARNING



## OVERVIEW

This project is about predicting rocks against Mines by the SONAR technology with the help of Machine Learning. SONAR is an abbreviated form of Sound Navigation and Ranging. It uses sound waves to detect objects underwater. Machine learning-based tactics, and deep learning-based approaches have applications in detecting sonar signals and hence targets.The three stages of Machine Learning are taking some data as input, extracting features, and predicting new patterns. The most common ML 
algorithms in this field are Logistic Regression, support vector machine, principal component analysis, k-nearest neighbors (KNN), etc.

## OBJECTIVE 

The main aim is to predict the rock or mine in the underwater(sea , oceans) using SONAR that uses sound propagation (usually underwater, as in submarine navigation) to navigate, measure distances (ranging), communicate with or detect objects on or under the surface of the water , which will help the sea divers , submarines to know whether the object is mine or rock . I am using machine learning algorithms to predict these by using the dataset.

## LIBRARIES USED

A Python library is a collection of related modules. It contains bundles of code that can be used repeatedly in different programs. It makes Python Programming simpler and convenient for the programmer. As we don’t need to write the same code again and again for different programs. Python libraries play a very vital role in fields of Machine Learning, Data Science, Data Visualization, etc.Python libraries that are used in the project are:
• Pandas
• gradio
• Numpy
• Matplotlib

## MODULES DESCRIPTION


### Dataset Collection
This module includes data collection and understanding the data to study the patterns and trends which helps inprediction and evaluating theresults.Dataset description is given belowThis Diabetes dataset contains 800 records and 10 attributes.
Table 1. Dataset Information
Attributes Type
Number of Pregnancies N
Glucose Level N
 Blood Pressure N
Skin Thickness(mm) N
Insulin N
BMI N
Age N

### Data Pre-processing
This phase of model handles inconsistent data in order to get more accurate and precise results. This dataset containsmissing values. So we imputed missing values for few selected attributes like Glucose level, Blood Pressure, SkinThickness, BMI and Age because these attributes cannot have values zero. Then we scale the dataset to normalizeall values.

### Clustering
In this phase, we have implemented K-means clustering on the dataset to classify each patient into either a diabeticor non-diabetic class. Before performing K-means clustering, highly correlated attributes were found which were,Glucose and Age. K-means clustering was performed on these two attributes. After implementation of this
clustering we got class labels (0 or 1) for each of our record.

### Model Building
This is most important phase which includes model building for prediction of diabetes. In this we have implementedmachine learning algorithms for diabetes prediction. These algorithms include Support Vector Classifier,Logistic Regression, K-Nearest Neighbour, Gaussian Naïve Bayes,Bagging algorithm, Gradient Boost Classifier.In this,I have just implemented the MLP Classifier.

### Evaluation
This is the final step of prediction model. Here, we evaluate the prediction results using various evaluation metricslike classification accuracy, confusion matrix and f1-score.Classification Accuracy- It is the ratio of number of correct predictions to the total number of input samples. 
