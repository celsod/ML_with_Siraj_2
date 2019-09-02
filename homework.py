import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math
import numpy as np
import pandas_datareader.data as web
import datetime
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor


start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2019, 9, 2)

df = web.DataReader("TPRE", "yahoo", start=start, end=end)
close_px = df['Adj Close'] # setting close_px to adjusted close from data
moving_avg = close_px.rolling(window=100).mean()  # moving average over a 100 period window

# feature engineering section
df_reg = df.loc[:, ['Adj Close', 'Volume']]  # create a new dataframe called df_reg

df_reg.fillna(value=-99999, inplace=True)  # fill in the NA values caused by the calculations
forecast_out = int(math.ceil(0.01*len(df_reg)))
# splitting off 1% of the data for forecasting purposes

forecast_col = 'Adj Close'
df_reg['label'] = df_reg[forecast_col].shift(-forecast_out)
X = np.array(df_reg.drop(['label'], 1))  # dropping the label column
# this portion is to separate the label that we want to predict

X = preprocessing.scale(X)
# need to preprocess the data to have the same distribution for regression purposes

X_late = X[-forecast_out:]  # creating a new dataframe called X_late
# this dataframe has the latter half of the data in it
X = X[:-forecast_out]  # updating the X dataframe
# this dataframe has the beginning half of the data in it

y = np.array(df_reg['label'])  # creating the label column AKA the targets
y = y[:-forecast_out]  # filling out the dataframe with the information

mlp_regress = MLPRegressor()
svr = SVR()
sgdr = SGDRegressor()

mlp_regress.fit(X, y)
svr.fit(X, y)
sgdr.fit(X, y)

print(f"The MLP Regressor score is: {mlp_regress.score(X, y)}")
print(f"The Support Vector Regressor score is: {svr.score(X, y)}")
print(f"The SGD Regressor score is: {sgdr.score(X, y)}")

forecast_mlpr = mlp_regress.predict(X_late)
forecast_svr = svr.predict(X_late)
forecast_sgdr = sgdr.predict(X_late)

df_reg['Forecast MLPR'] = np.nan  # creating an empty forecast dataframe column
# df_reg['Forecast SVR'] = np.nan
# df_reg['Forecast SGDR'] = np.nan

'''
I decided to plot only one forecast because whenever I tried to plot multiple forecasts the plot
would not show all three or even all two forecasts.  I am not sure why it is doing that.
'''

last_date = df_reg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_mlpr:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df_reg.loc[next_date] = [np.nan for _ in range(len(df_reg.columns)-1)]+[i]


mpl.rc('figure', figsize=(8, 7))
style.use('ggplot')
close_px.plot(label='TPRE')
moving_avg.plot(label='Moving Avg')
df_reg['Forecast MLPR'].tail(500).plot()
# df_reg['Forecast SVR'].tail(500).plot()
# df_reg['Forecast SGDR'].tail(500).plot()
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.autoscale(tight=True)
plt.show()