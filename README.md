# Liver-Patient-Prediction

Problem Statement:
	The given dataset is related to Indian patients who have been tested for a liver disease. Based on chemical compounds (bilrubin, albumin, protiens, alkaline phosphatase) present in human body and tests like SGOT, SGPT the outcome mentioned is whether person is a patient i.e, whether he needs to be diagnosed further or not. Perform data cleansing, and required transformations and build a predictive model which will be able to predict most of the cases accurately.

Data Description:
VARIABLE DESCRIPTIONS	
Age:	Age 
Gender:	Gender (Male and Female)
Total_Bilirubin:	Total Bilirubin present in body
Direct_Bilirubin:	Direct Bilirubin present in body
Alkaline_Phosphotase:	Alkaline Phosphotase  present in body
Alamine_Aminotransferase:	Alamine Aminotransferase  present in body
Aspartate_Aminotransferase:	Aspartate Aminotransferase  present in body
Total_Protiens:	 Total Protiens present in body
Albumin:	Albumin  present in body
Albumin_and_Globulin_Ratio:	Albumin and Globulin Ratio present in body
Class:	 No (person is not patient) Yes (person is a patient)

Steps for Data Pre-processing:
1.	Data Exploration:
i.	We will do this problem in Python language.
ii.	First of all we will import all the necessary libraries and load the data and then we will check head, info, statistical information using describe, shape i.e dimensions to explore the data and decide what to do with each variable.

2.	Handling Missing Values:
i.	Now we will Check for Na’s in the data and then we will impute the Na’s with suitable values.
ii.	Here we have Na’s in Gender, Total_Protiens and Albumin_and_Globulin_Ratio. Now we will impute missing values in Gender with mode and we will impute missing value in Total_Protiens and Albumin_and_Globulin_Ratio with mean.
iii.	After imputing values for Na’s we will again check that all the Na’s are removed or not.

3.	Data Visualizations:
i.	Now we will explore data using visualizations. First of all we will plot pair-plots for the whole data.
ii.	Now we will plot the number of Class i.e number of patients.
iii.	After that we will plot Class with respect to Gender.
iv.	Now we will plot number of persons with respect to Age.

4.	Data Tuning:
i.	We will convert Gender to indicator variable by generating dummies.
ii.	Now we concatenate dummies in the main data.
iii.	 We will recode Class variable as (Yes : 1) and (No : 0) to ease the processing.
iv.	We will check the datatype of each variable and  change the datatype if necessary. Here it is not necessary to change the datatypes.
v.	Now we will remove the original Gender variable because we have created dummies.
vi.	After all these we will create copies of data to apply different machine learning algorithms.

5.	Applying Machine Learning Algorithms:
i.	We will Split the data into training and testing and then we will train the model on training data and then predict the values of Class on testing data. 
ii.	Also we will Scale the data for fast processing.
iii.	Now we will apply Logistic Regression, Gaussian Naïve Bayes, Random Forest, Support Vector Machine and K-Nearest Neighborhood algorithms on the data.
iv.	After applying algorithms we will print the performance parameters. 

Steps to perform algorithms:
i.	Split the data into training and testing and then import the necessary libraries for different algorithms.
ii.	Now we will show the Heatmap of the data.
iii.	Then we will Scale the data to ease calculations
iv.	Then create the model and fit the training data to the model.
v.	After fitting the model print the performance parameters like train score, test score, accuracy, Confusion Matrix.
vi.	After applying Logistic regression to the data it is observed that precision of 0 i.e (not patient) is more then the precision of 1 i.e (is patient). 
vii.	So we will check the probability i.e counts of 1 and 0. Here we observed that the Class 0 has much more counts then 1.
viii.	So now we will Up-sample the data to equal the probabilities i.e counts if 0 and 1.
ix.	After that we will again apply Logistic Regression and the compare the results.
x.	Now we will apply the remaining algorithms on Up-Sampled data.

Performance Parameters:
1.	Accuracy: This is the simplest scoring measure. It calculates the proportion of correctly classified instances. 
Accuracy = (TP + TN) / (TP+TN+FP+FN) 

2.	Sensitivity (also called Recall or True Positive Rate): Sensitivity is the proportion of actual positives which are correctly identified as positives by the classifier. 
Sensitivity = TP / (TP +FN) 

3.	Specificity (also called True Negative Rate): Specificity relates to the classifier’s ability to identify negative results. Consider the example of medical test used to identify a certain disease. The specificity of the test is the proportion of patients that do not to have the disease and will successfully test negative for it. In other words: 
Specificity: TN / (TN+FP) 

4.	Precision: This is a measure of retrieved instances that are relevant. In other words: 
Precision: TP/(TP+FP)
where ;
	True Positive (TP): Observation is positive, and is predicted to be positive.
	False Negative (FN): Observation is positive, but is predicted negative.
	True Negative (TN): Observation is negative, and is predicted to be negative.
	False Positive (FP): Observation is negative, but is predicted positive.

5.	F1-Score: It is difficult to compare two models with different Precision and Recall. So to make them comparable, we use F-Score. It is the Harmonic Mean of Precision and Recall. As compared to Arithmetic Mean, Harmonic Mean punishes the extreme values more. F-score should be high.
F1-Score : (2*Recall*Precision) / (Recall + Precision)

Conclusion:
	From the results we can conclude that Random Forest Algorithm predict most of the cases accurately with Accuracy of  approx (83.2%) and F1-Score of (84%) for Class 1 and (82%) for Class 0.
