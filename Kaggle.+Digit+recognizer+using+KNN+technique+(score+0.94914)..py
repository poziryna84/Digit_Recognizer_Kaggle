
# coding: utf-8

# # Digit recognizer using KNN technique

# ## Loading packages and data.

# In[82]:

# Import Pandas & KNeighborsClassifier from sklearn.neighbors
import pandas as pd
import numpy  as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')


# In[83]:

df_train = pd.read_csv("C:/Users/pozir/OneDrive/Documentos/XDigit_Recognizer/train.csv")
df_test = pd.read_csv("C:/Users/pozir/OneDrive/Documentos/XDigit_Recognizer/test.csv")
#list(df_test.columns.values)


# ## Viewing the data

# In[ ]:

i=25
img=df_train.drop('label', axis=1).iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(df_train.drop('label', axis=1).iloc[i,0])


# In[37]:

print(df_train)


# In[39]:

df_train.head()


# In[40]:

print(df_train.shape)
print(df_test.shape)


# We´ve got 42 thousand examples in the train set - it´ll take way to long to train a model, we´d better reduce it to let´s say 8 thousand example for train and 2 thousand for test/hold out set respectively.

# ## Creating arrays for the features and the response variable and dividing them into train and hold out sets.
# 

# In[84]:

X=df_train.iloc[5000:15000,1:].values # ".values" make them into arrays
y=df_train.iloc[5000:15000,:1].values


# In[85]:

# Split df_train into training and test(hold out) set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)


# In[86]:

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Over- and underfitting problems.
# 

# Let´s fit a knn model with the default number of neighbors which is 5 to the training dataset and then check its accuracy on the same data.

# In[87]:

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train.ravel())


# In[88]:

knn.score(X_train, y_train.ravel()) 

0.9655 accuracy! not bad, right?
# In[89]:

knn.score(X_test, y_test.ravel())

Though on the unseen data our model performes significantly worse predicting only 94% of the data correctly. 
# Let´s compute and plot the training and testing accuracy scores for different neighbor values. By observing how the accuracy scores differ for the training and testing sets with different values of k, we will check for overfitting and underfitting.

# In[90]:

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 25)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train.ravel())
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train.ravel())

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test.ravel())

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# We can see how the accuracy differs while training and testing. Normally a small number of neighbors can lead to an overfit model (using 1 neighbor it predicts 100% correctly on the seen data and drops down to 95% when predicting on the unseen data), while a large number of neigbors can lead to an underfit model. To better estimate how well our model will do with different numbers of neighbors we are going to use Tuning Grid Search and 5-fold cross-validation.
# 

# ## Tuning Grid Search

# In[91]:

# Setup the hyperparameter grid
neighbors = np.arange(1,9)
param_grid = {'n_neighbors': neighbors}


# In[92]:

# Instantiate a knn classifier: knn
knn = KNeighborsClassifier()
# Instantiate the GridSearchCV object: knn_cv
knn_cv = GridSearchCV(knn,param_grid, cv= 5)


# In[93]:

# Fit it to the data
knn_cv.fit(X_train, y_train.ravel())


# In[94]:

# Print the tuned parameters and score
print("Tuned KNN Parameters: {}".format(knn_cv.best_params_)) 
print("Best score is {}".format(knn_cv.best_score_))

It says that the best result can be reached using 1 neighbor and the average accuracy is  0.9435 let´s see how it does on the unseen data.
# In[96]:

knn_cv.score(X_test, y_test.ravel())

0.951 on the training set - our model overfits a lot less now:), let´s predict on the train data.
# ## Predicting and saving the results

# In[97]:

predict = knn_cv.predict(df_test)


# In[140]:

df=pd.DataFrame(predict)
df.index+=1


# In[141]:

final_submission=pd.DataFrame({"ImageId": list(range(1,len(predict)+1)),
                         "Label": predict})


# In[146]:

final_submission.head()


# In[150]:

final_submission.to_csv("submission.csv", index=False, header=True)

