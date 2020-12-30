---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
```

```python
import seffaflik
seffaflik.Kimlik(istemci_taniticisi="abcdefg")
from seffaflik.elektrik import tuketim
```

```python
# input reading
consumption_data = tuketim.gerceklesen(baslangic_tarihi='2016-01-01', bitis_tarihi='2020-12-01')
consumption_data.columns = ["Date", "Hour", "Consumption"]
consumption_data.head()
```

```python
# calculation of lag48 and lag168
consumption_data["Lag_48"] = consumption_data["Consumption"].shift(48)
consumption_data["Lag_168"] = consumption_data["Consumption"].shift(168)
consumption_data.tail()
```

```python
check = consumption_data.groupby("Date").apply(lambda x: False if len(x["Hour"].unique()) != 24 else True)
check[check == False]
```

```python
ind = check[check == False].index[0]
consumption_data[consumption_data["Date"]==ind]
```

- There is an error in the 2067<sup>th</sup> row. To prevent misinterpretation, I will drop this week from the data.

```python
# dropping the week
start = int(consumption_data[consumption_data["Date"]=="2016-03-21"].index[0])
consumption_data.drop(consumption_data.index[range(start,2088)],inplace=True)
consumption_data.reset_index(drop=True,inplace=True)
```

## Task a 
MAPE calculation for naive forecasts with Lag48 and Lag168

```python
# classification to test and train data
test_data = consumption_data[consumption_data["Date"] >= "2020-11-01"].reset_index(drop=True).copy().dropna()
train_data = consumption_data[consumption_data["Date"] < "2020-11-01"].copy().dropna().reset_index(drop=True)
test_data.head()
```

```python
# function for calculation of MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    _abs = np.abs((y_true - y_pred) / y_true)
    return np.mean(_abs) * 100, _abs
```

```python
# text formatting
from IPython.display import Markdown, display
def printmd(string,color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))
```

```python
# naive forecasts
nf_48,_abs48 = mean_absolute_percentage_error(test_data["Consumption"], test_data["Lag_48"]) 
nf_168,_abs168 = mean_absolute_percentage_error(test_data["Consumption"], test_data["Lag_168"]) 

mapes48 = []
mapes168 = []
for i in range(24):
    mapes48.append(mean_absolute_percentage_error(test_data[test_data["Hour"]==i]["Consumption"], test_data[test_data["Hour"]==i]["Lag_48"])[0])
    mapes168.append(mean_absolute_percentage_error(test_data[test_data["Hour"]==i]["Consumption"], test_data[test_data["Hour"]==i]["Lag_168"])[0])

printmd("**MAPE for Lag_48:** {}".format(round(nf_48,5)))
printmd("**MAPE for Lag_168:** {}".format(round(nf_168,5)))
```

```python
# plotting test data and naive forecasts
fig, axes = plt.subplots(2,1,sharex=True,sharey=True)

axes[0].set_title("Test Data and Naive Forecast with Lag_168",fontweight="bold")
axes[0].plot(np.arange(len(test_data)), test_data["Consumption"], color="tab:orange", label="test data")
axes[0].plot(np.arange(len(test_data)), test_data["Lag_168"], color='g', label="naive forecast") 
axes[0].set_xlabel('Time')
axes[0].set_ylabel('MWh')
axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

axes[1].set_title("Test Data and Naive Forecast with Lag_48",fontweight="bold")
axes[1].plot(np.arange(len(test_data)), test_data["Consumption"], color="tab:orange", label="test data")
axes[1].plot(np.arange(len(test_data)), test_data["Lag_48"], color="g", label="naive forecast")  
axes[1].set_xlabel('Time')
axes[1].set_ylabel('MWh')
plt.xticks([])
axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.subplots_adjust(hspace=0.2, top=1.5, bottom=0, left=0, right=2)
plt.show()
```

| Lag |MAPE |
| :-- | :-: | 
| Lag_48 | 8.06031 | 
| Lag_168 | 3.44919 | 



- As it can be seen in the above graphs and MAPE results, naive forecast with **Lag 168 performs better** than **Lag 48**.
This indicates that weekly relation in consumption is more important than two-day relation.


## Task b:

```python
from sklearn.linear_model import LinearRegression
```

```python
# training linear regression model
model = LinearRegression().fit(train_data[["Lag_48","Lag_168"]].values, train_data["Consumption"].values)
```

```python
# summary for linear regression model
from regressors import stats
print("\n==================== SUMMARY =======================")
stats.summary(model, train_data[["Lag_48","Lag_168"]], train_data["Consumption"], ["Lag_48","Lag_168"])
```

```python
# predicting consumption values for test data
y_test = test_data["Consumption"].values
y_pred = model.predict(test_data[["Lag_48","Lag_168"]].values )
```

```python
printmd("**----- Linear Regression Model -----**")
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
linear_mape, _abslinear_mape = mean_absolute_percentage_error(df['Actual'], df['Predicted'])
linear_forecast = _abslinear_mape.tolist()
printmd("**MAPE:**  {}".format(round(linear_mape,5)))
```

```python
#linear_mape, linear_forecast = mean_absolute_percentage_error(df['Actual'], df['Predicted'])
#linear_forecast = linear_forecast.tolist()
```

```python
import seaborn as sns
```

```python
# plotting actual and predicted consumption
fig, axes = plt.subplots(1,1)
axes.set_title("Time Series Plot of Actual and Predicted Consumption",fontweight="bold")
axes.plot(np.arange(len(y_test)), y_test,color="firebrick", label="actual")
axes.plot(np.arange(len(y_test)), y_pred,color="teal", label="predicted") 
axes.set_xlabel('Time')
axes.set_ylabel('MWh')
plt.xticks([])
axes.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.subplots_adjust(hspace=0.1, top=1, bottom=0, left=0, right=2)
```

```python
fig = plt.figure(figsize=(8,6))
sns.regplot(x=y_pred,y=y_test, marker='+')
plt.legend(labels=["Predicted","Actual"])
plt.title("Actual and Predicted Consumption", size=18)
plt.xlabel("Predicted", size=12)
plt.ylabel("Actual", size=12);
plt.show()
```

| Method | MAPE |
| :-- | :-: | 
| Lag_48 | 8.06031 | 
| Lag_168 | 3.44919 | 
| Linear Regression | 4.22959 |


- After two naive forecasts, in this step linear regression is used for predicting consumption values. MAPE value of linear regression is better than naive forecast with Lag 48 but it is worse than naive forecast with Lag 168. Thus, we can say that linear regression performs better than Lag 48 but worse than naive approach with Lag168.


## Task c:

```python
mapes = []

# function for applying linear regression each hour seperately
def hourly_linear_regression(hour):
    hdata_train = train_data[train_data["Hour"]==hour]
    hdata_test = test_data[test_data["Hour"]==hour]
    model = LinearRegression().fit(hdata_train[["Lag_48","Lag_168"]].values, hdata_train["Consumption"].values)
    y_test = hdata_test["Consumption"].values
    y_pred = model.predict(hdata_test[["Lag_48","Lag_168"]].values )
    
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    mean_mape, mape_abs =  mean_absolute_percentage_error(df['Actual'], df['Predicted'])
    
    printmd("**Linear Regression Model for Hour {}**".format(str(hour)),color="firebrick")
    printmd("**Intercept**: {}".format(model.intercept_))
    printmd("**Coefficients**: {}".format(model.coef_))
    printmd("**MAPE:**  {}".format(round(mean_mape,5)))
    visualize(y_test,y_pred)
    return mean_mape
    
def visualize(y_test,y_pred):
    fig = plt.figure()
    plt.title("Time Series Plot of Actual and Predicted Consumption",fontweight="bold")
    plt.plot(np.arange(len(y_test)), y_test,color="firebrick", label="actual")
    plt.plot(np.arange(len(y_test)), y_pred,color="teal", label="predicted") 
    plt.xlabel('Time')
    plt.ylabel('MWh')
    plt.xticks([])
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.subplots_adjust(hspace=0.1, top=1, bottom=0, right=1.5)
```

```python
mapes.append(hourly_linear_regression(0))
```

```python
mapes.append(hourly_linear_regression(1))
```

```python
mapes.append(hourly_linear_regression(2))
```

```python
mapes.append(hourly_linear_regression(3))
```

```python
mapes.append(hourly_linear_regression(4))
```

```python
mapes.append(hourly_linear_regression(5))
```

```python
mapes.append(hourly_linear_regression(6))
```

```python
mapes.append(hourly_linear_regression(7))
```

```python
mapes.append(hourly_linear_regression(8))
```

```python
mapes.append(hourly_linear_regression(9))
```

```python
mapes.append(hourly_linear_regression(10))
```

```python
mapes.append(hourly_linear_regression(11))
```

```python
mapes.append(hourly_linear_regression(12))
```

```python
mapes.append(hourly_linear_regression(13))
```

```python
mapes.append(hourly_linear_regression(14))
```

```python
mapes.append(hourly_linear_regression(15))
```

```python
mapes.append(hourly_linear_regression(16))
```

```python
mapes.append(hourly_linear_regression(17))
```

```python
mapes.append(hourly_linear_regression(18))
```

```python
mapes.append(hourly_linear_regression(19))
```

```python
mapes.append(hourly_linear_regression(20))
```

```python
mapes.append(hourly_linear_regression(21))
```

```python
mapes.append(hourly_linear_regression(22))
```

```python
mapes.append(hourly_linear_regression(23))
```

```python
# all MAPE results
printmd("**MAPE values:**")
for i in range(0,24):
    print("Hour {}: {}".format(str(i),round(mapes[i],5)))
```

| Method | MAPE |
| :-- | :-: | 
| Lag_48 | 8.06031 | 
| Lag_168 | 3.44919 | 
| Linear Regression | 4.22959 |

<!-- #region -->
- Using linear regression for each hour seperately improves MAPE values for hours between (18.00 - 7.00) compared to previous part. 


- Compared to naive forecast with Lag 168, this approach has a better performance for hours between (19.00 - 6.00), and this time period is called *off-peak times* in electricity market, electiricity suppliers tend to charge less price during these hours. But for other periods, this approach does not perform better than part(b) and naive forecast with Lag168.
<!-- #endregion -->

## Task d:

```python
consumption_data.dropna(inplace=True)
consumption_data.head()
```

```python
# transforming data to wide format
wide_data = consumption_data.pivot_table(index="Date", columns="Hour", values=["Lag_48","Lag_168","Consumption"])
wide_data.columns = ["Lag_day"+str(i)+"_"+"hour"+str(j) for i in (2,7) for j in range(24)] + ["Consumption_"+str(i) for i in range(24)]
wide_data.reset_index(inplace=True)
wide_data.head()
```

```python
wide_train_lasso = wide_data[wide_data["Date"]<"2020-11-01"].dropna()
wide_test_lasso = wide_data[wide_data["Date"]>="2020-11-01"].dropna()
wide_train_lasso.head()
```

```python
from sklearn.linear_model import LassoCV
```

```python
def lasso_regression (train_data, test_data, hour):
    Feature_List  = train_data.columns.to_list()
    feature = [s for s in Feature_List if "Lag" in s]
    # separating out the features
    x_train = train_data.loc[:, feature].values
    
    # separating out the target
    y_train = train_data.loc[:,['Consumption_'+ str(hour)]].values
    y_train = np.ravel(y_train)
    alpha = np.logspace(-10, 1, 400)
    lasso = LassoCV(cv=10, normalize=True, tol=0.001, selection='cyclic', random_state=123, alphas=alpha,max_iter=10000)
    lasso.fit(x_train, y_train)
    
    # separating out the features
    x_test = test_data.loc[:, feature].values
    
    # separating out the target
    y_test = test_data.loc[:,['Consumption_'+ str(hour)]].values 
    y_pred = lasso.predict(x_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    mean_mape, mape_abs =  mean_absolute_percentage_error(df['Actual'], df['Predicted'])
    return mean_mape
```

```python
lasso_forecast = []
printmd("**MAPE values:**")
for hour in range(24):
    output = lasso_regression(wide_train_lasso, wide_test_lasso, hour)
    lasso_forecast.append(output)
    print('Hour'+' '+str(hour)+':' + str(output))
```

```python
printmd("**MAPE difference to Part(c):**")
for hour in range(24):
    print('Hour'+' '+str(hour)+':' + str(lasso_forecast[hour]-mapes[hour]))
```

- Compared to part(c), MAPE values are decreased for most of the hours but the improvement is very small, all of them are less than 1. Thus, I cannot say that penalized regression model performs much better than linear regression model.


## Task f:

```python
mapes_df=pd.DataFrame()
mapes_df["Lag_48"]= mapes48
mapes_df["Lag_168"]=mapes168
mapes_df["Linear_Reg"]=mapes
mapes_df["Lasso_Reg"]=lasso_forecast
mapes_df
```

```python
sns.set(style="whitegrid")
plt.figure(figsize=(16,8))
sns.boxplot(data=mapes_df)
plt.show()
```

Considering the results, we can say that:
- The naive approach has a better performance with Lag168.
- Comparing linear regression and lasso regression; linear regression has a better performance but when it is compared with naive forecast with Lag 168, it performs worse.
- Naive forecast with Lag 168 seems as a better option for forecasting consumption compared to others.
- I was expecting to see that Lasso Regression performs better than others but it does not. This may be happened because of inefficient parameters or convergence parameters. 
