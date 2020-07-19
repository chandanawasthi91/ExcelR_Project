import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


file_name=r'C:\Users\cawasthi\Desktop\Ankit\AMZN_stocks_20200711_210015.csv'
af=pd.read_csv(file_name)
sns.pairplot(af)
ml1 = smf.ols("af['asset']~af['volume']+af['close']",data=af).fit() 
ml1.summary()
af.corr()


X = af.iloc[:, 4:6].values
Y = af.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

import numpy as np
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

p = y_pred
q = y_test
pd.DataFrame(p,q)


#forcasting
af.columns

forcast = af[['date','close']]
forcast.columns
forcast.close.plot()

forcast.dtypes

forcast['dt_format'] = pd.to_datetime(forcast['date'])
forcast["month_year"] = forcast.dt_format.dt.strftime("%b-%Y")

dd = forcast.groupby('month_year').mean()
col1  = list(dd.index)
dd['Date'] = col1
dd['dt_format'] = pd.to_datetime(dd['Date'])

forcast_model = pd.DataFrame()
forcast_model = dd[['dt_format','close']]
forcast_model = forcast_model.sort_values(by=['dt_format'])
forcast_model.reset_index(inplace = True)
forcast_model.drop('dt_format',axis=1,inplace=True)
forcast_model.columns
forcast_model_bkp = forcast_model
plt.plot(forcast_model['month_year'],forcast_model['close']);plt.title("stock market");plt.xlabel("time");plt.ylabel("closing price")

forcast_model['months']= 0
#Model based
for i in range(279):
    p = forcast_model["month_year"][i]
    forcast_model['months'][i]= p[0:3]


forcast_model.drop('month_year',axis=1,inplace=True) 
month_dummies = pd.DataFrame(pd.get_dummies(forcast_model['months']))
forcast_model = pd.concat([forcast_model,month_dummies],axis = 1)    


forcast_model["t"] = np.arange(1,280)
forcast_model["t_squared"] = forcast_model["t"]*forcast_model["t"]
forcast_model["log_Rider"] = np.log(forcast_model["close"])
forcast_model.columns

Train = forcast_model.head(209)
Test = forcast_model.tail(70)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('close~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Rider~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('log_Rider~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('close~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Rider~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Rider~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#!pip install Cython
#!pip install fbprophet 
