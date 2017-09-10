
# coding: utf-8

# In[4]:

from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# In[2]:

# this function is a utility to display face images from the dataset

def display_faces(images, label, num2display):
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(num2display):
        p = fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        
        p.text(0, 14, str(label[i]))
        p.text(0, 60, str(i))

from sklearn.datasets import fetch_lfw_people
faces = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# In[6]:

# store features in variable X and the label in variable y
X, y = faces.data, faces.target
np.shape(X)


# ---

# In[5]:
print(faces.DESCR)

print("Total number of data points: " + str(len(faces.data)))

print ("Number of unique target names: " + str(len(faces.target_names)))

# print out the number of classes, and corresponding name of each class
class_ids=np.unique(y)

print('Number of classes: ' + str(len(class_ids)) + "\n")
print('Class IDs and Names:\n')
for i in range(len(class_ids)):
    # insert your code here
    print(str(class_ids[i]) + ': ' + str(faces.target_names[i]))

# print out 10 photos of Geroge W Bush and 10 photos of Tony Blair

# George W Bush

display_faces(faces.images[faces.target==3],faces.target[faces.target==3],10)

# Tony Blair
display_faces(faces.images[faces.target==6],faces.target[faces.target==6],10)



**Write your code to use PCA for dimensionality reduction with *30 components* to transform variable `X` to variable `pca_X`**

# In[28]:

pca = decomposition.PCA(n_components=30)
pca_X = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
np.shape(pca_X)

** Use the reduced dimension features `pca_X` obtained in previous steps to write your code to show the performance (recall, precision, accuracy, F-score) using *Logistic Regression* as the classifier and *a single train-split* with 30% of data will be used for testing and the rest for training.**

# In[60]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Create training and testing sets

X_train, X_test, y_train, y_test = train_test_split(pca_X, y, test_size = 0.3, random_state=23)

# Verify correct split

print("% of Test samples: " + str(round(((len(X_test)/(len(X_train)+len(X_test))*100)),0)) + "%")

# Create Logistic Regression
log_reg = LogisticRegression(penalty='l1', random_state=123)

# Train model
train_model1 = log_reg.fit(X_train,y_train)

# Check results on test set
results1 = train_model1.decision_function(X_test)

#Assign most likely label to each result
results1_labels = [np.argmax(results1[i]) for i in range(len(results1))]

# print confusion matrix & Classification report to check accuracy per class
print("\nConfusion Matrix:\n")
print(metrics.confusion_matrix(results1_labels, y_test))
target_names=[faces.target_names[i] for i in range(len(np.unique(y)))]
print("\nClassification Report:\n")
print(metrics.classification_report(results1_labels, y_test, target_names=target_names))

# Print scores
acc1 = accuracy_score(results1_labels, y_test)
rec1 = recall_score(results1_labels, y_test, average="macro")
f11 = f1_score(results1_labels, y_test, average="macro")
prec1 = precision_score(results1_labels, y_test, average="macro")
print("Single Train-Split Results (Logistic Regression): \n")
print("Accuracy\t" + str(np.round(acc1,4)))
print("F1-Score\t" + str(np.round(f11,4)))
print("Recall\t\t" + str(np.round(rec1,4)))
print("Precision\t" + str(np.round(prec1,4)))

# <span style="color:red">**(c)**</span>** Use the reduced dimension features pca_X obtained in step (a) to write your code to show the performance (recall, precision, accuracy, F-score) *using Repeated Random Train-Split* (10 runs, each with 70/30 split for traning and testing). 

# In[61]:

from sklearn import model_selection

# Define CV method and measures

shuffle = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=61)
results_accuracy = model_selection.cross_val_score(log_reg, pca_X, y, cv=shuffle, scoring="accuracy")
results_f1 = model_selection.cross_val_score(log_reg, pca_X, y, cv=shuffle, scoring="f1_macro")
results_recall = model_selection.cross_val_score(log_reg, pca_X, y, cv=shuffle, scoring="recall_macro")
results_precision = model_selection.cross_val_score(log_reg, pca_X, y, cv=shuffle, scoring="precision_macro")

# Print results

print("Stratified Repeated Random Train-Split Results\n")
print("\t\tMean\t\tSt. Dev")
print("Accuracy\t" + str(results_accuracy.mean().round(4)) +"\t\t" + str(results_accuracy.std().round(4)))
print("F1-Score\t" + str(results_f1.mean().round(4)) +"\t\t" + str(results_f1.std().round(4)))
print("Recall\t\t" + str(results_recall.mean().round(4)) +"\t\t" + str(results_recall.std().round(4)))
print("Precision\t" + str(results_precision.mean().round(4)) +"\t\t" + str(results_precision.std().round(4)))
        
    
  
# *Notes*:  Stratified Shuffle/Split and Stratified KFold cross-validation has been used to ensure fair representation of different classes across splits.  Macro scoring has been used for F-score, recall and precision in order to measure the average classification ability of each class equally. 

# <span style="color:red">**(d)**</span>** Use the reduced dimension features pca_X obtained in step (a) to write your code to show the performance (recall, precision, accuracy, F-score) using *K-fold cross-validation with k=10 folds* **.
# <div style="text-align: right"> <span style="color:red">**[10 points]**</span> </div>

# In[62]:

# YOU ARE REQUIRED TO INSERT YOUR CODES IN THIS CELL

# Define CV method and measures

kfold = model_selection.StratifiedKFold(n_splits=10, random_state=98)

kfold_accuracy = model_selection.cross_val_score(log_reg, pca_X, y, cv=kfold, scoring="accuracy")
kfold_f1 = model_selection.cross_val_score(log_reg, pca_X, y, cv=kfold, scoring="f1_macro")
kfold_recall = model_selection.cross_val_score(log_reg, pca_X, y, cv=kfold, scoring="recall_macro")
kfold_precision = model_selection.cross_val_score(log_reg, pca_X, y, cv=kfold, scoring="precision_macro")

# Print results

print("Stratified K-Fold Cross Validation Results\n")
print("\t\tMean\t\tSt. Dev")
print("Accuracy\t" + str(kfold_accuracy.mean().round(4)) +"\t\t" + str(kfold_accuracy.std().round(4)))
print("F1-Score\t" + str(kfold_f1.mean().round(4)) +"\t\t" + str(kfold_f1.std().round(4)))
print("Recall\t\t" + str(kfold_recall.mean().round(4)) +"\t\t" + str(kfold_recall.std().round(4)))
print("Precision\t" + str(kfold_precision.mean().round(4)) +"\t\t" + str(kfold_precision.std().round(4)))


# <span style="color:red">**(e)**</span>** Preparing a table to summarize the performances obtained from step (c) and and (d) against recall, precision, accuracy and F-score. From this table, which approach, Repeated Random Train-Split or K-fold cross-validation, would you recommend to use?**
# <div style="text-align: right"> <span style="color:red">**[10 points]**</span> </div>

# |    |Stratified Train-Split||Stratified KFold|| 
# |---|---|---|---|---|
# |Measure|Mean  |Std Dev|Mean  | Std Dev|
# |Accuracy|0.7282|0.0170|0.7446|0.0281|
# |Recall|0.6126|0.0233|0.6354|0.0396|
# |Precision|0.6738|0.0292|0.7014|0.0475|
# |F-Score|0.6344|0.0231|0.6530|0.0368|
# 
# This table shows that the Stratified KFold approach produced slightly higher mean scores across all four measures, although it also showed greater variability in the form of higher standard deviations.  Overall the Kfold approach is the one I would recommend for a dataset of this small size, as it accounts for all samples on each iteration and, in this case, provided the better results.

# <span style="color:red">**Question 2.2.**</span> ** Once you have build some inition about the problem and the effectiveness of some modelling choices, if the initial results aren't too bad, you should start to gain some confidences in your approach. You are recommended to look at your results in the previous question again to convince yourselves of your results. Some questions you should starting asking are: Is your average prediction accuracy over 50%? over 70%? What else can you try to make it better? Is Logistic Regression a good choice? Did the reduced dimension help to improve the performance?**
# 

# In[3]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def ml_algorithm_screening_face(X,y,model, model_name, scoring_metrics, pca_dim, n_runs):
    estimators = []
    seed = 10
    if (pca_dim > 0):
        estimators.append(('pca', decomposition.PCA(n_components=pca_dim)))
    
    estimators.append((model_name,model))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=n_runs, random_state=seed)
    try:
        results = cross_val_score(pipeline, X, y, cv=kfold, scoring=scoring_metrics, verbose=1, n_jobs=-1)
    except ValueError:
        print("Opps! something went wrong!")
      
    return results

Write your code to call this function with following specification and print out the mean and standard deviation of the *accuracy* obtained.**
#     - model: Logistic Regression (using l1 regulaization)
#     - PCA dimnesion = 30
#     - the number of runs=10

# In[65]:

lreg = LogisticRegression(penalty='l1')
pipe_results = ml_algorithm_screening_face(X,y,lreg,'Logistic Regression','accuracy',pca_dim=30, n_runs=10)
print("Logistic Regression Results:\n\nAccuracy Mean: " +str(pipe_results.mean().round(4)) + "\nAccuracy Std. Dev.: " +str(pipe_results.std().round(4)))


# <span style="color:red">**(b)**</span>** Face recognition is a typical-dimensional data problem encountered in modern machine learning problems. This explains why one might routinely use PCA to reduce its dimesion.**
# 
# **Write your code to search for right dimension from a list of *dim = {10, 20,..,150}* using the same setting in question 2.2(a).**
# - Print out the results for each dimension.
# - Use box-plot to visualize the mean and standard deviation of the accuracy for each dimension on the same figure, and 
# - Report the dimension for PCA that gives the best result in term of accuracy.
# 
# <div style="text-align: right"> <span style="color:red">**[20 points]**<span> </div>

# In[7]:

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

# Run function for multiple values of pca_dim, store results
pca_results = {}
pca_stats = {}
for i in range(10,151,10):
    pca_run = ml_algorithm_screening_face(X,y,LogisticRegression(penalty='l1'),'Logistic Regression','accuracy',pca_dim=i, n_runs=10)
    pca_results[i] = (pca_run)
    pca_stats[i] = (pca_run.mean(),pca_run.std())

# In[8]:

# Print accuracy results for each set of dimensions (10 - 150)

for i in pca_results:
    print("Results for %s Dimensions: " %i, pca_results[i].round(3))


# In[9]:

#Print mean and standard deviation for each dimension
print("Dimensions\tMean\tStd. Dev")
for i in pca_stats:
    print ("\t{}\t".format(i) + str(pca_stats[i][0].round(4)) + "\t" + str(pca_stats[i][1].round(4)))


# In[24]:

# Plot Box Plot and ErrorBar of PCA Dimensions

import matplotlib.lines as mlines

# Create arrays of values to plot
pca_data = list(pca_results.values())
pca_index = list(pca_results.keys())
pca_data = np.array(pca_data)
y_std = np.asarray(pca_index)
y_std = (y_std/10)+0.25
pca_mean = pca_data.mean(1)
pca_std = pca_mean.std()
error = np.array(pca_data.std(1))

# Create Plots
fig, ax1 = plt.subplots()
ax1.boxplot(pca_data.T, widths=0.2)
ax1.set_xticklabels(pca_index)
ax1.set_xlabel('Number of Dimensions\n(Note ErrorBar is shown slightly off-axis for visual clarity)')
ax1.set_title('Box-plot and ErrorBar of Accuracy by PCA Dimensions')
ax1.set_ylabel('Accuracy')
ax1.grid(b=True, which='major', axis='x', color='gray', linestyle='--', alpha=0.4)
plt.errorbar(y_std, pca_mean, yerr = error, color='r')
black_line = mlines.Line2D([],[],color='black', label='Box Plot')
red_line = mlines.Line2D([],[],color='red', label='ErrorBar: Mean and Std. Dev.')
ax1.legend(handles = [black_line, red_line], loc=4)
plt.show()


# The number of PCA dimensions that gives the best result in terms of accuracy is 90 (mean = 0.8113, std. dev = 0.0264)

# ---
# ### Part 3: Model Screening and Comparison
# <div style="text-align: right"><span style="color:red">**[Total mark for this part: 70 points]**.<span></div>

#Fixing the PCA dimension to the value you founded in Part 2, write your code to compare the performance among these two linear models. Which model performs better? and write down any observation you might have.** 
# 
# * [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# * [Linear Discriminant Analysis (LDA)](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
# 
# <div style="text-align: right"> <span style="color:red">**[10 points]**</span> </div>

# In[100]:

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Test algorithms with default settings

metrics = ['accuracy','recall_macro','f1_macro','precision_macro']
logreg = LogisticRegression(penalty='l1')
lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
models = [(logreg,'Logistic Regression'),(lda, 'LDA')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))
   


# In[35]:

# Test accuracy with varied parameters incl solvers, multi_class, tolerance levels etc

metrics = ['accuracy']
logreg = LogisticRegression(penalty='l2', solver="lbfgs", multi_class="multinomial", warm_start="true")
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage="auto", tol=0.0001)
models = [(logreg,'Logistic Regression'),(lda, 'LDA')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# The **LDA model** performed better than the logistic regression across all four metrics, averaged across 10 iterations.  The LDA also displayed slightly less variance (measured by standard deviation) across all measures except for accuracy.  The best results for both algorithms were achieved with Scikit's default settings in terms of solver types (SVD for LDA, liblinear for Logistic Regression), and changing parameters such as class weightings and penalty types did not improve the results.


# In[37]:


# Write your code to  compare the performance among non-linear models.
#     - Support Vector Machines (SVM)
#     - Neural Networks

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Test algorithms with default settings

metrics = ['accuracy','recall_macro','f1_macro','precision_macro']
svm = SVC(kernel="rbf")
nn = MLPClassifier(solver='adam')
models = [(svm,'SVM'),(nn, 'Neural Network')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[52]:


# Test algorithms with different parameters


metrics = ['accuracy']
svm = SVC(kernel="linear")
nn = MLPClassifier(hidden_layer_sizes=(500,),solver='sgd')
models = [(svm,'SVM'),(nn, 'Neural Network')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[ ]:

# Test LinearSVC

from sklearn.svm import LinearSVC
metrics = ['accuracy']
lsvm = LinearSVC()
models = [(lsvm,'LSVM')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[53]:

# Get all metrics for most accurate versions found

metrics = ['accuracy','recall_macro','f1_macro','precision_macro']
svm = SVC(kernel="linear")
nn = MLPClassifier(hidden_layer_sizes=(500,),solver='sgd')
models = [(svm,'SVM'),(nn, 'Neural Network')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# The **Neural Network** outperformed the SVM on all criteria except for Recall, where the SVM performed slightly better, averaged across 10 iterations.  The NN did, however, have higher standard deviation across most criteria.  Changing the kernel type for SVM made a significant difference, with the original RBF kernel returning very poor results, while the linear kernel provided much better results.  The results of the NN were also significantly improved by increasing the number of hidden layers to 500 and changing the solver type to Stoachastic Gradient Descent (SGD).


# YOU ARE REQUIRED TO INSERT YOUR CODES IN THIS CELL

# Write your code to compare the performance among non-parametric and probabilistic models.
#     - Random Forest Classifier
#     - K-NN Classifer
#     - GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Test with default parameters for all algorithms

metrics = ['accuracy','recall_macro','f1_macro','precision_macro']
rfc = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors = 6)
gnb = GaussianNB()
models = [(rfc,'Random Forest'),(knn, 'KNN'),(gnb,'Naive Bayes')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))



# In[74]:

# Check accuracy with different parameters

metrics = ['accuracy']
rfc2 = RandomForestClassifier(n_estimators=100, max_features=0.5)
knn2 = KNeighborsClassifier(n_neighbors = 6, weights = "distance")
models = [(rfc2,'Random Forest'),(knn2, 'KNN')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# In[75]:

# Calculate full set of metrics based on parameters used in best accuracy score

metrics = ['accuracy','recall_macro','f1_macro','precision_macro']
rfc = RandomForestClassifier(n_estimators=100, max_features=0.5)
knn = KNeighborsClassifier(n_neighbors = 6, weights = "distance")
gnb = GaussianNB()
models = [(rfc,'Random Forest'),(knn, 'KNN'),(gnb,'Naive Bayes')]
for model, name in models:
    print('%s :' %name,)
    for metric in metrics:
        model_results = ml_algorithm_screening_face(X,y,model, name, metric, 90, 10)
        mean = model_results.mean()
        std = model_results.std()
        print(metric + "\t Mean:" + str(mean) + "\t Std Dev: " + str(std))


# The **Gaussian Naive Bayes** easily outperformed the other two probabilistic algorithms, having both significantly higher means and lower standard deviations across all four criteria, averaged over 10 iterations.  The performance of the Random Forest was signifiantly improved by increased the number of trees to 100 (although improvement stopped at over 100 trees) and by limited the number of features to 50%.  Several different K values were tested for the K-NN algorith, with 6 being found to produce the optimal results, however the improvements were minimal.
# 

# <span style="color:red">**Question 3.1**</span>.** Write a summary and report the performance of chosen models in terms of different metrics: precision, recall, accuracy, F-measure.**

#     
# |   |Recall   | Precision  |  Accuracy |F-measure   |
# |---|---|---|---|---|
# |  Logistic Regression |0.7551   | 0.7727  | 0.8090  | 0.7482  |
# |  LDA | **0.7974**   | **0.8113**  | **0.8392**  | **0.7950**|
# |  SVM | 0.7442  | 0.7480  | 0.7880  | 0.7320  |
# |  Neural Networks | 0.7421  | 0.7733  | 0.8176  | 0.7564  |
# |  Random Forest |0.4569   |0.6574   | 0.6484  | 0.4962  |
# |  K-NN | 0.4489  | 0.5904  | 0.6033  | 0.4736  |
# |  GaussianNB | 0.6852  | 0.7741  | 0.7718  | 0.7118  | |
# 
