import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\ExcelR_Project\\Insurance Dataset .csv")
df.columns

#checking for Null columns and data types
df.isnull().sum()
df.dtypes

#removing the unnnesacry columns
df.drop(["Area_Service"],inplace=True,axis=1)
df.drop(["Description_illness"],inplace=True,axis=1)
df.drop(["payment_typology_2"],inplace=True,axis=1)
df.drop(["payment_typology_3"],inplace=True,axis=1)
df.drop(["Abortion"],inplace=True,axis=1)
df.drop(["ccs_procedure_code"],inplace=True,axis=1)
df.drop(["ccs_diagnosis_code"],inplace=True,axis=1)
df.drop(["ethnicity"],inplace=True,axis=1)
df.drop(["Certificate_num"],inplace=True,axis=1)
df.drop(["Hospital Id"],inplace=True,axis=1)
df.drop(['zip_code_3_digits'],inplace=True,axis=1)
df.drop(['Hospital County'],inplace=True,axis=1)
df.drop(['Hospital Name'],inplace=True,axis=1)
df.drop(['year_discharge'],inplace=True,axis=1)
df.drop(['Weight_baby'],inplace=True,axis=1)

#EDA on each columns

#Mortality risk
df['Mortality risk'].value_counts()
df['Mortality risk'] = df['Mortality risk'].fillna(df['Mortality risk'].value_counts().index[0])

#checking the collinearity
df1 = df.corr() # none of the columns are collinear

#renaming columns to fit the model
cols = ['Age', 'Gender', 'Cultural_group',
'Days_spend_hsptl', 'Admission_type', 'Home_or_self_care',
 'ccs_diagnosis_description',
 'ccs_procedure_description',
'apr_drg_description', 'apr_mdc_description', 'Code_illness',
 'Mortality_risk', 'Surg_Description',
'Payment_typology_1', 
 'Emergency_dept_yes_No', 'Tot_charg', 'Tot_cost',
'ratio_of_total_costs_to_total_charges', 'Result']

df.columns = cols

str_columns = ["Age","Gender", "Cultural_group","Days_spend_hsptl","Admission_type","Home_or_self_care","ccs_diagnosis_description","apr_drg_description","ccs_procedure_description","apr_mdc_description","Mortality_risk","Surg_Description","Payment_typology_1","Emergency_dept_yes_No"]                         

str_cols = pd.DataFrame()
#convert the Object columns from numeric
for i in str_columns:
    df[i] = df[i].astype('category')

#converting results in code for building the model
df["Result_Output"] = np.zeros(df.shape[0])
df['Result'].value_counts()
df.loc[df.Result == "Genuine","Result_Output"] = 1
df.loc[df.Result == "Fraudulent","Result_Output"] = 0
df['Result'] = df["Result_Output"]

#building the logistic regression model
import statsmodels.formula.api as sm
logit_model = sm.logit('Result~Age+Gender+Cultural_group+Days_spend_hsptl+Admission_type+Home_or_self_care+ccs_diagnosis_description+ccs_procedure_description+apr_drg_description+apr_mdc_description+Code_illness+Mortality_risk+Surg_Description+Payment_typology_1+Emergency_dept_yes_No+Tot_charg+Tot_cost+ratio_of_total_costs_to_total_charges',data = df).fit()

logit_model.summary()

y_pred = logit_model.predict(df)

y_pred_val = y_pred

df["y_pred"]=y_pred
plt.hist(y_pred)
df.loc[y_pred>=0.5,"y_pred"] = 1
df.loc[y_pred<0.5,"y_pred"] = 0

from sklearn.metrics import classification_report
classification_report(df.Result,df.y_pred)

#confusion matrix
confusion_matrix = pd.crosstab(df['Result'],df.y_pred)
confusion_matrix

from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(df['Result'],y_pred_val)
plt.hist(y_pred_val)


plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr)
