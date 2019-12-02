import pandas as pd
import numpy as np
import os
import statistics
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


#Importing the dataset
Mediclaim= pd.read_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\ExcelR_Project\\Insurance Dataset .csv", low_memory=False)
print(Mediclaim.head(4))


print(Mediclaim.columns) #Prints the column names of the dataset

#Removing unnecessary columns from the dataset
Mediclaim.drop(['Hospital Id','Certificate_num','Area_Service','Hospital County','year_discharge','Abortion','ethnicity','Weight_baby'],axis=1,inplace= True)
print(Mediclaim.columns)

# # Exploratory Data Analysis
Mediclaim.loc[Mediclaim["Result"]=="Genuine"]
Mediclaim.loc[Mediclaim["Result"]=="Fraudulent"]

Mediclaim.corr()

Mediclaim.drop(['ccs_diagnosis_code','ccs_procedure_code'], axis=1, inplace=True)
Mediclaim.isnull().sum(axis=0)
sns.heatmap(Mediclaim.isnull(),yticklabels=False,cbar=False,cmap='viridis')
Mediclaim.drop(['payment_typology_2','payment_typology_3','zip_code_3_digits'], axis=1, inplace=True)


Mediclaim.Description_illness.mode() #Moderate
Mediclaim["Mortality risk"].mode() #Minor
Mediclaim['Description_illness'].fillna("Moderate",inplace=True)
Mediclaim['Mortality risk'].fillna("Minor",inplace=True)

print(Mediclaim.isnull().sum())
Mediclaim.describe() 
Mediclaim.info()

outliers=[]

def detect_outliers(data):    
    threshold=3
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


print(len(detect_outliers(Mediclaim["Tot_charg"]))) #12140
len(detect_outliers(Mediclaim["Tot_cost"])) #21994
len(detect_outliers(Mediclaim['ratio_of_total_costs_to_total_charges'])) #22551
sns.set_style('whitegrid')
sns.countplot(x='Result',data=Mediclaim)

#plt.savefig('UniResult.png', dpi=500)
# ### From the above plot,it is clearly visible that the proportion of the claims that are genuine are more than the fraudulent claims. ###
# # Univariate Analysis of Tot_charg
# # First Moment Business Decision/ Measures of Central Tendency
Mediclaim.Tot_charg.mean() #3052
Mediclaim.Tot_charg.median() #14589.88
Mediclaim.Tot_charg.mode() #3052

# # Second Moment Business Decision/ Measures of Dispersion
Mediclaim.Tot_charg.max() #6196973.5
Mediclaim.Tot_charg.min() #0.31
Range= Mediclaim.Tot_charg.max()-Mediclaim.Tot_charg.min()
print(Range) #6196973.19
np.var(Mediclaim["Tot_charg"]) #3011183744.69
np.std(Mediclaim["Tot_charg"]) #54874.25

# # Third Moment Business Decision
# #As a general thumbrule, if the skewness is <-1 or >1, then the distribution is highly skewed
# #If the skewness is between -1 to -0.5 or 0.5 to 1, then the distribution is moderately skewed
# #If the skewness is between -0.5 and 0.5, then the distribution is approximately symmetric

Mediclaim["Tot_charg"].skew() #18.1 (Highly Skewed)


# # Fourth Moment Business Decision
# #Normal distribution for kurtosis is 3
# #Kurtosis value greater than 3, leptokurtic
# #Kurtosis value between 1 and 3, platykurtic

Mediclaim["Tot_charg"].kurt() #775.17 (Leptokurtic)
plt.hist(Mediclaim["Tot_charg"])

# # Boxplot of Tot_charg
plt.boxplot(Mediclaim["Tot_charg"], vert=0)

# # Univariate Analysis of Tot_cost
# # First Moment Business Decision/ Measures of Central Tendency

Mediclaim.Tot_cost.mean() #10463.99
Mediclaim.Tot_cost.median() #5928.48
Mediclaim.Tot_cost.mode() #1208.47

# # Second Moment Business Decision/ Measures of Dispersion
Mediclaim.Tot_cost.max() #2562477.66
Mediclaim.Tot_cost.min() #0.1
Range= Mediclaim.Tot_cost.max()-Mediclaim.Tot_cost.min()
print(Range) #2562477.56
np.var(Mediclaim["Tot_cost"]) #497452596.13
np.std(Mediclaim["Tot_cost"]) #22303.65

# # Third Moment Business Decision
# #As a general thumbrule, if the skewness is <-1 or >1, then the distribution is highly skewed
# #If the skewness is between -1 to -0.5 or 0.5 to 1, then the distribution is moderately skewed
# #If the skewness is between -0.5 and 0.5, then the distribution is approximately symmetric

Mediclaim["Tot_cost"].skew() #27.59 (Highly Skewed)
# # Fourth Moment Business Decision
# #Normal distribution for kurtosis is 3
# #Kurtosis value greater than 3, leptokurtic
# #Kurtosis value between 1 and 3, platykurtic

Mediclaim["Tot_cost"].kurt() #1585.23 (Leptokurtic)
# # Graphical Visualization
# # Histogram of Tot_cost
plt.hist(Mediclaim["Tot_cost"])

# # Boxplot of Tot_cost
plt.boxplot(Mediclaim["Tot_cost"], vert=0)

# # Univariate Analysis of ratio_of_total_costs_to_total_charges
# # First Moment Business Decision/ Measures of Central Tendency
Mediclaim.ratio_of_total_costs_to_total_charges.mean() #0.46
Mediclaim.ratio_of_total_costs_to_total_charges.median() #0.41
Mediclaim.ratio_of_total_costs_to_total_charges.mode() #0.83

# # Second Moment Business Decision/ Measures of Dispersion

Mediclaim.ratio_of_total_costs_to_total_charges.max() #157.56
Mediclaim.ratio_of_total_costs_to_total_charges.min() #0.033
Range= Mediclaim.ratio_of_total_costs_to_total_charges.max()-Mediclaim.ratio_of_total_costs_to_total_charges.min()
print(Range) #157.53
np.var(Mediclaim["ratio_of_total_costs_to_total_charges"]) #0.645
np.std(Mediclaim["ratio_of_total_costs_to_total_charges"]) #0.802


# # Third Moment Business Decision
# #As a general thumbrule, if the skewness is <-1 or >1, then the distribution is highly skewed
# #If the skewness is between -1 to -0.5 or 0.5 to 1, then the distribution is moderately skewed
# #If the skewness is between -0.5 and 0.5, then the distribution is approximately symmetric
Mediclaim["ratio_of_total_costs_to_total_charges"].skew() #95.88 (Highly Skewed)


# # Fourth Moment Business Decision
# #Normal distribution for kurtosis is 3
# #Kurtosis value greater than 3, leptokurtic
# #Kurtosis value between 1 and 3, platykurtic

Mediclaim["ratio_of_total_costs_to_total_charges"].kurt() #11789.04 (Leptokurtic)

# # Graphical Visualization
# # Histogram of ratio_of_total_costs_to_total_charges
plt.hist(Mediclaim["ratio_of_total_costs_to_total_charges"])
# # Boxplot of ratio_of_total_costs_to_total_charges
plt.boxplot(Mediclaim["ratio_of_total_costs_to_total_charges"], vert=0)


# # Visualization of categorical variables
def plot_bar(Mediclaim, cols):
    for col in cols:
        fig= plt.figure(figsize=(8,8))
        ax= fig.gca() #Defining axis
        count= Mediclaim[col].value_counts()
        count.plot.bar(ax=ax, color='red')
        ax.set_title('Bar plot of '+ col)
        ax.set_xlabel(col)
        ax.set_ylabel('Counts')
        plt.show()
  
plot_cols=['Admission_type','Emergency dept_yes/No','Age','Gender']
plot_bar(Mediclaim,plot_cols)
sns.set_style('whitegrid')
sns.countplot(x='Admission_type',data=Mediclaim)
plt.savefig('UniAdmissionType.png', dpi=500)
sns.set_style('whitegrid')
sns.countplot(x='Emergency dept_yes/No',data=Mediclaim)
plt.savefig('UniEmerDept.png', dpi=500)

sns.set_style('whitegrid')
sns.countplot(x='Age',data=Mediclaim)
plt.savefig('UniAge.png', dpi=500)

sns.set_style('whitegrid')
sns.countplot(x='Gender',data=Mediclaim)
plt.savefig('UniGender.png', dpi=500)


# # Bivariate Analysis

sns.set_style('whitegrid')
sns.countplot(x='Result', hue='Gender',data=Mediclaim)


# ### From the above plot, it can be inferred that the proportion of both genuine as well as fraudulent claims made by the female population is higher than male population ###
plt.figure(figsize=(8, 5))
sns.set_style('whitegrid')
sns.countplot(x='Result',hue='Payment_typology_1',data=Mediclaim)
sns.set_style('whitegrid')
sns.countplot(x='Result',hue='Age',data=Mediclaim)


# ### Inference that can be drawn is that people aged 70 or Older are the ones making the highest claims that are genuine and at the same time making most fraudulent claims as well when compared to other age groups ###
from sklearn.preprocessing import LabelEncoder
lb_make= LabelEncoder()
Mediclaim['Hospital Name']= lb_make.fit_transform(Mediclaim['Hospital Name'])
Mediclaim['Age']=lb_make.fit_transform(Mediclaim['Age'])
Mediclaim['Gender']=lb_make.fit_transform(Mediclaim['Gender'])
Mediclaim['Cultural_group']=lb_make.fit_transform(Mediclaim['Cultural_group'])
Mediclaim['Admission_type']=lb_make.fit_transform(Mediclaim['Admission_type'])
Mediclaim['Home or self care,']=lb_make.fit_transform(Mediclaim['Home or self care,'])
Mediclaim['ccs_diagnosis_description']=lb_make.fit_transform(Mediclaim['ccs_diagnosis_description'])
Mediclaim['ccs_procedure_description']=lb_make.fit_transform(Mediclaim['ccs_procedure_description'])
Mediclaim['apr_drg_description']=lb_make.fit_transform(Mediclaim['apr_drg_description'])
Mediclaim['apr_mdc_description']=lb_make.fit_transform(Mediclaim['apr_mdc_description'])
Mediclaim['Description_illness']=lb_make.fit_transform(Mediclaim['Description_illness'])
Mediclaim['Mortality risk']=lb_make.fit_transform(Mediclaim['Mortality risk'])
Mediclaim['Surg_Description']=lb_make.fit_transform(Mediclaim['Surg_Description'])
Mediclaim['Payment_typology_1']=lb_make.fit_transform(Mediclaim['Payment_typology_1'])
Mediclaim['Days_spend_hsptl']=lb_make.fit_transform(Mediclaim['Days_spend_hsptl'])
Mediclaim['Emergency dept_yes/No']=lb_make.fit_transform(Mediclaim['Emergency dept_yes/No'])
Mediclaim['Result']=lb_make.fit_transform(Mediclaim['Result'])

# # Decision Tree
Mediclaim.shape

Mediclaim['Result'].unique()
print(Mediclaim.Result.head())
colnames= list(Mediclaim.columns)
colnames
predictors= colnames[:20]
predictors
target=colnames[20]
target

#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Mediclaim[predictors],Mediclaim[target])
X_train.shape,X_test.shape

from sklearn.tree import DecisionTreeClassifier
#Calling the function with 'entropy' as the criterion
model= DecisionTreeClassifier(criterion='entropy')
#Fitting the training data into the model
model.fit(X_train,y_train)
#Predicting the performance of the model on the training set
pred_train= model.predict(X_train)
#Measuring training accuracy by using CrossTabulation
pd.crosstab(y_train,pred_train)
#Measuring training accuracy
np.mean(pred_train == y_train)
#Predicting the performance of the model on the test set
pred_test= model.predict(X_test)
#Measuring testing accuracy by using CrossTabulation
pd.crosstab(y_test ,pred_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred_test))
confusion_matrix(y_test, pred_test)

# ### TP- 17204
# ### TN- 144898
# ### FP- 48418
# ### FN- 51624

# ### Sensitivity= TP/(TP+FN)= 0.25
# ### Specificity= TN/(TN+FP)= 0.75

#Measuring testing accuracy
np.mean(pred_test== y_test)
# # SMOTE Analysis on Imbalanced Dataset
from sklearn.model_selection import train_test_split
SX_train, SX_test, Sy_train, Sy_test= train_test_split(Mediclaim[predictors], Mediclaim[target],test_size=0.25, random_state=27)

from imblearn.over_sampling import SMOTE
sm= SMOTE(random_state=27, ratio=1.0)
SX_train, Sy_train= sm.fit_sample(SX_train, Sy_train)

#Calling the function with 'entropy' as criterion
S_model= DecisionTreeClassifier(criterion='entropy')

S_model.fit(SX_train, Sy_train)
S_test_pred= S_model.predict(SX_test)
np.mean(Sy_test== S_test_pred)
print(classification_report(Sy_test, S_test_pred))
confusion_matrix(Sy_test, S_test_pred)


# ### TP- 17429
# ### TN- 144378
# ### FP- 48187
# ### FN- 52150

# ### Sensitivity= TP/(TP+FN)= 0.25
# ### Specificity= TN/(TN+FP)= 0.749
help(DecisionTreeClassifier)
#Calling the function with gini as criterion
model_2= DecisionTreeClassifier(criterion='gini')

#Fitting the training data into the model
model_2.fit(SX_train, Sy_train)

#Predicting the model performance by using the test data
G_pred_test=model_2.predict(SX_test)
print(classification_report(Sy_test, G_pred_test))
confusion_matrix(Sy_test, G_pred_test)

# ### TP- 17709
# ### TN- 143134
# ### FP- 47907
# ### FN- 53394

# ### Sensitivity= TP/(TP+FN)= 0.249
# ### Specificity= TN/(TN+FP)= 0.749
#Measuring accuracy
np.mean(Sy_test== G_pred_test)

# # Neural Networks
# Splitting the data into training and testing
from sklearn.model_selection import train_test_split
NX_train, NX_test, Ny_train, Ny_test= train_test_split(Mediclaim[predictors], Mediclaim[target])

#Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

NX_train= scaler.fit(NX_train).transform(NX_train)
NX_test= scaler.fit(NX_test).transform(NX_test)

#Importing necessary libraries
from sklearn.neural_network import MLPClassifier
#Calling the function
mlp= MLPClassifier(hidden_layer_sizes=(30,30))
#Fitting the training data into the model
mlp.fit(NX_train, Ny_train)

#Predicting the model performance by using the test data
nn_test_pred= mlp.predict(NX_test)
print(classification_report(Ny_test, nn_test_pred))
confusion_matrix(Ny_test, nn_test_pred)

# ### TP- 0
# ### TN- 196796
# ### FP- 65348
# ### FN- 0

# ### Sensitivity= TP/(TP+FN)= 0
# ### Specificity= TN/(TN+FP)= 0.75
#Measuring accuracy
np.mean(Ny_test == nn_test_pred)

#building the naive_bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Naive Bias
# Splitting data into train and test
Xtrain,Xtest,ytrain,ytest = train_test_split(Mediclaim[predictors],Mediclaim[target],test_size=0.3, random_state=0)

ignb = GaussianNB()
imnb = MultinomialNB()

# Building and predicting at the same time 
pred_gnb = ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)

# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
'''
array([[   734,  77853],
       [  2488, 233498]], dtype=int64)
'''
np.mean(pred_gnb==ytest.values.flatten()) #0.74
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(ytest,pred_gnb))
'''
     precision    recall  f1-score   support

           0       0.23      0.01      0.02     78587
           1       0.75      0.99      0.85    235986

   micro avg       0.74      0.74      0.74    314573
   macro avg       0.49      0.50      0.44    314573
weighted avg       0.62      0.74      0.64    314573
'''


# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_mnb) # GaussianNB model
'''
array([[ 20760,  57827],
       [ 62037, 173949]], dtype=int64)
'''

pd.crosstab(ytest.values.flatten(),pred_mnb) # confusion matrix using 
np.mean(pred_mnb==ytest.values.flatten()) # 0.61

confusion_matrix(ytest,pred_mnb) # Multinomal model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(ytest,pred_mnb))
'''
print(classification_report(ytest,pred_mnb))
              precision    recall  f1-score   support

           0       0.25      0.26      0.26     78587
           1       0.75      0.74      0.74    235986

   micro avg       0.62      0.62      0.62    314573
   macro avg       0.50      0.50      0.50    314573
weighted avg       0.63      0.62      0.62    314573
'''
)
