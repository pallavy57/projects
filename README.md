This is a repository containing multiple projects and conceptual explanations

FAQ Hyperparameters
1. There is a list of HPs for any selected algorithm(s)
2.There are two generic approaches to search effectively in the HP space are GridSearch CV and RandomSearch CV.
3.Causes of Data Leakage
    Data Pre-processing
    The major root cause is doing all EDA processes before splitting the dataset into test and train
    Doing straightforward normalizing or rescaling on a given dataset
    Performing Min/Max values of a feature
    Handling missing values without reserving the test and train
    Removing outliers and Anomaly on a given dataset
    Applying standard scaler, scaling, assert normal distribution on the full dataset
4.Steps to Perform Hyperparameter Tuning
    Select the right type of model.
    Review the list of parameters of the model and build the HP space
        The learning rate for training a neural network.
        The C and sigma hyperparameters for support vector machines.
        The k in k-nearest neighbors. 
        Max_Depth,Min_Samples_Split,Min_Samples_Leaf,Max_Features for Decision Tree
    Finding the methods for searching the hyperparameter space
    Applying the cross-validation scheme approach
    Assess the model score to evaluate the model 
    
5. Hyperparameters
Logistic Regression Classifier : C=1/λ where lambda is the regularization parameter
KNN (k-Nearest Neighbors) Classifier: used for regression and classification problems
                                       KNeighborsClassifier(n_neighbors=5, p=2, metric=’minkowski’)
                                        – n_neighbors is the number of neighbors
                                        – p is Minkowski (the power parameter)
                                        If p = 1 Equivalent to manhattan_distance,
                                        p = 2. For Euclidean_distance
When a data point is provided to the algorithm, with a given value of K, it searches for the K nearest neighbors to that data point
the KNN algorithm next determines the majority of neighbors belong to which class. For example, if the majority of neighbors belong to class ‘Green’, then the given data point is also classified as class ‘Green’.
In Sklearn we can use GridSearchCV to find the best value of K from the range of values.
create a KNN classifier instance and then prepare a range of values of hyperparameter K from 1 to 31 that will be used by GridSearchCV to find the best value of K.
we set our cross-validation batch sizes cv = 10 and set scoring metrics as accuracy as our preference                                        
Support Vector Machine Classifier : SVC(kernel=’linear’, C=1.0, random_state=0)
                                    – kernel specifies the kernel type to be used in the chosen algorithm,
                                    kernel = ‘linear’, for Linear Classification
                                    kernel = ‘rbf’ for Non-Linear Classification.
                                    C is the penalty parameter (error)
                                    random_state is a pseudo-random number generator        
Decision Tree Classifier: measure the quality of a split, max_depth is the maximum depth of       the                       tree, and random_state is the seed used by the random number generator.                          
                           DecisionTreeClassifier(criterion=’entropy’, max_depth=3, random_state=0)     

6.Influencing on Models
Linear Model
What degree of polynomial features should use?
Decision Tree
What is the maximum allowed depth?
What is the minimum number of samples required at a leaf node in the decision tree?
Random forest
How many trees we should include?
Neural Network
How many neurons we should keep in a layer?
How many layers, should keep in a layer?
Gradient Descent
What learning rate should we?

7.Hyperparameter Optimization Techniques
Manual Search
Random Search
Grid Search
Halving
Grid Search
Randomized Search
Automated Hyperparameter tuning
Bayesian Optimization
Genetic Algorithms
Artificial Neural Networks Tuning
HyperOpt-Sklearn
Bayes Search

implement Hyperparameters optimization techniques, we have to have the Cross-Validation techniques as well in the flow because we may not miss out on the best combinations that work on tests and training.

8.CONFUSION MATRIX

TP: True Positive: The values which were actually positive and were predicted positive.

FP: False Positive: The values which were actually negative but falsely predicted as positive. Also known as Type I Error.

FN: False Negative: The values which were actually positive but falsely predicted as negative. Also known as Type II Error.

TN: True Negative: The values which were actually negative and were predicted negative.


True positive upper left: 540 records of the stock market crash were predicted correctly by the model.
False-positive upper right: 150 records of not a stock market crash were wrongly predicted as a market crash.
False-negative lower left: 110 records of a market crash were wrongly predicted as not a market crash.
True Negative lower right: 200 records of not a market crash were predicted correctly by the model.

focus on reducing the value of FN and increasing the value of Recall
focus on reducing the value of FP and increasing the value of Precision when the mail is falsely predicted as spam
In some cases of imbalanced data problems, both Precision and Recall are important so we consider the F1 score as an evaluation metric

https://www.kaggle.com/code/satishgunjal/tutorial-k-fold-cross-validation


The first way is to re-write False Negative and False Positive. False Positive is a Type I error because False Positive = False True and that only has one F. False Negative is a Type II error because False Negative = False False so thus there are two F’s making it a Type II. 
Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
Misclassification (all incorrect / all) = FP + FN / TP + TN + FP + FN
Precision (true positives / predicted positives) = TP / TP + FP
Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
Specificity (true negatives / all actual negatives) =TN / TN + FP


Should I choose Random Forest regressor or classifier
https://www.mygreatlearning.com/blog/random-forest-algorithm/#:~:text=A%20random%20forest%20classifier%20works,cannot%20be%20defined%20by%20classes.
analyticsvidhya.com/blog/2021/06/understanding-random-forest/