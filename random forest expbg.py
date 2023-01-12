from unittest.mock import _patch_dict

print("Glory to God")

import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
df = pd.read_csv("expt_bg_magpie - expt_bg_magpie (1).csv")
df2 = pd.read_csv('expt_bg_feature_importance.csv')
y = df['Experimental band gap']
excluded = ["Experimental band gap","chemicalFormula","composition"]
#excluded2 = ["Computed band gap"]
#X = df2.drop(excluded2,axis=1)
X = df2
print(X.shape)
#model
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

#rf = RandomForestRegressor(n_estimators=50, random_state=1)

#rf.fit(X, y)
#print('training R2 = ' + str(round(rf.score(X, y), 3)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
z=y_test

rf_reg = GradientBoostingRegressor()
rf_reg.fit(X_train, y_train)

pf = pd.read_csv("for gga 60.csv")
check_nan = pf.isnull().values.any()
print(check_nan)
pf = pf.dropna()
pf.to_csv("Selected for gga.csv")
pf = pf.drop(["formula"],axis=1)
print(pf.shape)


predict = rf_reg.predict(pf)
print(predict)
pf = pd.DataFrame(predict)
pf.to_csv("final hh predict.csv")
#GBtrain= rf_reg()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_reg, X, y, cv=10,scoring="r2")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# get fit statistics
print('training R2 = ' + str(round(rf_reg.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train))))
print('test R2 = ' + str(round(rf_reg.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test))))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import  mean_absolute_percentage_error

mae = mean_absolute_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
maep = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
mae = mean_absolute_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
maep = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
print("mae1=",mae)
print("maep1=",maep)





"""

feature_names = [f"feature {i}" for i in range(X.shape[1])]
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
result = permutation_importance(
    rf_reg, X_test, y_test, n_repeats=10, random_state=20, n_jobs=2)
forest_importances = pd.Series(result.importances_mean, index=feature_names)
#df3 = pd.DataFrame(forest_importances)
#df3 = pd.concat([df],axis=0)
#print(df3)
#indexNames = df3[ df3['forest_importances'] < 0 ].index
#df3.drop(indexNames , inplace=True)
#df3.to_csv("final features.csv")
y_Pred = rf_reg.predict(X_test)
#z1=y_Pred


feature_importance = rf_reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    rf_reg, X_test, y_test, n_repeats=10, random_state=40, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set(font_scale=1.3)
s = sns.regplot(
    x='Computed band gap',
    y='Experimental band gap',
    data=df)
s.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
s.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
s.set(title="Experimental bandgap vs PBE bandgap")
plt.show()
sns.set_style('whitegrid')
sns.set(font_scale=1.3)

from gapminder import gapminder
s = sns.scatterplot(
    x='Computed band gap',
    y='Experimental band gap',
    data=df,size='MagpieData mean CovalentRadius', legend="brief",hue='MagpieData mode MeltingT',sizes=(20, 200),alpha=0.8,palette="flare")
s.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
s.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
s.legend(fontsize=7)
s.set(title="Feature Comparision ")
sns.set(font_scale=1.3)


plt.show()
s = sns.scatterplot(
    x='MagpieData mode MeltingT',
    y='Experimental band gap',
    data=df,size='MagpieData mean CovalentRadius', legend='brief',hue='Computed band gap',alpha=0.8,hue_norm=(0,1),sizes=(20, 200),palette="flare")

s.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
s.set(title="Feature Comparision ")
s.legend(fontsize=7)
sns.set(font_scale=1.3)
plt.show()
s = sns.scatterplot(
    x='MagpieData mean Electronegativity',
    y='Experimental band gap',
    data=df,size='MagpieData mean CovalentRadius',hue='Computed band gap', legend='brief',alpha=0.8,hue_norm=(0,1),sizes=(20, 200),palette="flare")
s.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
s.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
s.set(title="Feature Comparision ")
sns.set(font_scale=1.3)
s.legend(fontsize=7)
plt.show()


sns.set(font_scale=.6)
plt.figure()
fig, ax = plt.subplots(figsize=(100,100))
sns.heatmap(
    df2.corr(),cmap="Reds" ,vmin= -1, vmax=1, square=True,
           linewidth=0.3, cbar_kws={"shrink": .8} );
plt.show()

#df5 = pd.DataFrame(y_Pred,columns=y_Pred)
#print(df5)
fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Gradient Boosting Regression',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 5, 'test R2 = 0.923', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 7, 'test RMSE = 0.627', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred, deg=1)
xseq = np.linspace(0, 10, num=100)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(.2))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()

from sklearn.ensemble import RandomForestRegressor

rf_reg2 = RandomForestRegressor()


rf_reg2.fit(X_train, y_train)
y_Pred2 = rf_reg2.predict(X_test)
mae2 = mean_absolute_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))
maep2 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))
print("mae2",mae2)
print("maep2",maep2)


# get fit statistics
print('training R2 = ' + str(round(rf_reg2.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg2.predict(X_train))))
print('test R2 = ' + str(round(rf_reg2.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Random Forest',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 5, 'test R2 = 0.888', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 7, 'test RMSE = 0.754', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred2, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred2, deg=1)
xseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()

from sklearn.ensemble import ExtraTreesRegressor
rf_reg3 = ExtraTreesRegressor()


rf_reg3.fit(X_train, y_train)
y_Pred3 = rf_reg3.predict(X_test)
mae3 = mean_absolute_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))
maep3 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))
print("mae3",mae3)
print("maep3",maep3)


# get fit statistics
print('training R2 = ' + str(round(rf_reg3.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg3.predict(X_train))))
print('test R2 = ' + str(round(rf_reg3.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for ExtraTrees Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 5, 'test R2 = 0.903', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 7, 'test RMSE = 0.701', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred3, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred3, deg=1)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()

from sklearn.linear_model import LinearRegression

rf_reg4 = LinearRegression()


rf_reg4.fit(X_train, y_train)
y_Pred4 = rf_reg4.predict(X_test)

mae4 = mean_absolute_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))
maep4 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))
print("mae4",mae4)
print("maep4",maep4)

# get fit statistics
print('training R2 = ' + str(round(rf_reg4.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg4.predict(X_train))))
print('test R2 = ' + str(round(rf_reg4.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Linear Regression',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.scatter(z,y_Pred4, s=60, alpha=0.7, edgecolors="k")
ax.text(1, 5, 'test R2 = 0.18', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 7, 'test RMSE = 2.039', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
b, a = np.polyfit(z, y_Pred4, deg=1)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()

from sklearn.svm import SVR
rf_reg5 = SVR(kernel='linear',degree=1,C=1,gamma='scale')
rf_reg5.fit(X_train, y_train)
y_Pred5 = rf_reg5.predict(X_test)
mae5 = mean_absolute_error(y_true=y_test, y_pred=rf_reg5.predict(X_test))
maep5= mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg5.predict(X_test))
print("mae5",mae5)
print("mae5",maep5)


# get fit statistics
print('training R2 = ' + str(round(rf_reg5.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg5.predict(X_train))))
print('test R2 = ' + str(round(rf_reg5.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg5.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Support Vector Regression',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 5, 'test R2 = 0.683', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 9, 'test RMSE = 1.269', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred5, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred5, deg=1)
xseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()




from sklearn.ensemble import AdaBoostRegressor

rf_reg6 = AdaBoostRegressor()
rf_reg6.fit(X_train, y_train)
y_Pred6 = rf_reg6.predict(X_test)
mae6 = mean_absolute_error(y_true=y_test, y_pred=rf_reg6.predict(X_test))
maep6 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg6.predict(X_test))
print("mae6",mae6)
print("maep6",maep6)


# get fit statistics
print('training R2 = ' + str(round(rf_reg6.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg6.predict(X_train))))
print('test R2 = ' + str(round(rf_reg6.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg6.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Ada Boost Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 4, 'test R2 = 0.829', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 5, 'test RMSE = 0.931', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred6, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z,y_Pred6,deg=1)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
plt.show()

from sklearn.ensemble import BaggingRegressor

rf_reg7 = BaggingRegressor()

rf_reg7.fit(X_train, y_train)
y_Pred7 = rf_reg7.predict(X_test)
mae7 = mean_absolute_error(y_true=y_test, y_pred=rf_reg7.predict(X_test))
maep7 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg7.predict(X_test))
print("mae7",mae7)
print("maep7",maep7)


# get fit statistics
print('training R2 = ' + str(round(rf_reg7.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg7.predict(X_train))))
print('test R2 = ' + str(round(rf_reg7.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg7.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Bagging Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 4, 'test R2 = 0.866', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 5, 'test RMSE = 0.826', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred7, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred7, deg=1)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()



from sklearn.ensemble import HistGradientBoostingRegressor

rf_reg9 = HistGradientBoostingRegressor()
rf_reg9.fit(X_train, y_train)
y_Pred9 = rf_reg9.predict(X_test)

mae9 = mean_absolute_error(y_true=y_test, y_pred=rf_reg9.predict(X_test))
maep9 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg9.predict(X_test))
print("mae9",mae9)
print("maep9",maep9)

# get fit statistics
print('training R2 = ' + str(round(rf_reg9.score(X_train, y_train), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg9.predict(X_train))))
print('test R2 = ' + str(round(rf_reg9.score(X_test, y_test), 3)))
print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg9.predict(X_test))))

fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for HistGradient Boosting Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap(eV)",fontsize=12)
plt.ylabel("Predicted Values (eV)",fontsize=12)
ax.text(1, 4, 'test R2 = 0.858', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.text(1, 5, 'test RMSE = 0.848', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
ax.scatter(z,y_Pred9, s=60, alpha=0.7, edgecolors="k")
b, a = np.polyfit(z, y_Pred9, deg=1)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8) 
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()


from sklearn.ensemble import RandomForestRegressor

rf_reg = GradientBoostingRegressor()
sns.set_style("whitegrid", {'axes.grid' : False})

rf_reg.fit(X_train, y_train)
y_Pred = rf_reg.predict(X_test)
mae2 = mean_absolute_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
maep2 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg.predict(X_test))
print("RF mae2",mae2)
print("RF maep2",maep2)
l_pred = rf_reg.predict(X_train)

# get fit statistics
print('RF training R2 = ' + str(round(rf_reg.score(X_train, y_train), 3)))
print('RF training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train))))
print('RF test R2 = ' + str(round(rf_reg.score(X_test, y_test), 3)))
print('RF test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test))))
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Gradient Boosting Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap",fontsize=18)
plt.ylabel("Predicted Bandgap ",fontsize=18)
ax.text(-1, 6, 'test R2 = 0.923,training R2 = 0.998', style='italic', bbox={
       'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.text(-1, 6.75, 'test RMSE = 0.454,training RMSE = 0.0.189', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.scatter(z,y_Pred2, s=60, alpha=1.0, edgecolors="k",label='Testing Data')
ax.scatter(y_train,l_pred,s=60,alpha=0.5,color='red',edgecolors='k',marker='v',label='Training Data')
plt.legend(prop={'size': 16})

b, a = np.polyfit(z, y_Pred2, deg=1)
xseq = np.linspace(0, 10, num=100)
xseq2 = np.linspace(-1,10,num=100)
yseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq2, a + b * xseq2, color="k", lw=2.5);
plt.show()





rf_reg2 = RandomForestRegressor()
sns.set_style("whitegrid", {'axes.grid' : False})

rf_reg2.fit(X_train, y_train)
y_Pred2 = rf_reg2.predict(X_test)
mae2 = mean_absolute_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))
maep2 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))
print("RF mae2",mae2)
print("RF maep2",maep2)
l_pred = rf_reg2.predict(X_train)

# get fit statistics
print('RF training R2 = ' + str(round(rf_reg2.score(X_train, y_train), 3)))
print('RF training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg2.predict(X_train))))
print('RF test R2 = ' + str(round(rf_reg2.score(X_test, y_test), 3)))
print('RF test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg2.predict(X_test))))
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Random Forest Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap",fontsize=18)
plt.ylabel("Predicted Bandgap ",fontsize=18)
ax.text(-1, 6, 'test R2 = 0.953,training R2 = 0.998', style='italic', bbox={
       'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.text(-1, 6.75, 'test RMSE = 0.437,training RMSE = 0.466', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.scatter(z,y_Pred2, s=60, alpha=1.0, edgecolors="k",label='Testing Data')
ax.scatter(y_train,l_pred,s=60,alpha=0.5,color='red',edgecolors='k',marker='v',label='Training Data')
plt.legend(prop={'size': 16})

b, a = np.polyfit(z, y_Pred2, deg=1)
xseq = np.linspace(0, 10, num=100)
xseq2 = np.linspace(-1,10,num=100)
yseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq2, a + b * xseq2, color="k", lw=2.5);
plt.show()



rf_reg3= ExtraTreesRegressor()
sns.set_style("whitegrid", {'axes.grid' : False})

rf_reg3.fit(X_train, y_train)
y_Pred3 = rf_reg3.predict(X_test)
mae2 = mean_absolute_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))
maep2 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))
print("RF mae2",mae2)
print("RF maep2",maep2)
l_pred = rf_reg3.predict(X_train)

# get fit statistics
print('RF training R2 = ' + str(round(rf_reg3.score(X_train, y_train), 3)))
print('RF training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg3.predict(X_train))))
print('RF test R2 = ' + str(round(rf_reg3.score(X_test, y_test), 3)))
print('RF test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg3.predict(X_test))))
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Extra Trees Regressor ',fontsize=20)
plt.xlabel("Experimental Bandgap",fontsize=18)
plt.ylabel("Predicted Bandgap ",fontsize=18)
ax.text(-1, 6, 'test R2 = 0.969,training R2 = 0.999', style='italic', bbox={
       'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.text(-1, 6.75, 'test RMSE = 0.382,training RMSE = 0.079', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.scatter(z,y_Pred2, s=60, alpha=1.0, edgecolors="k",label='Testing Data')
ax.scatter(y_train,l_pred,s=60,alpha=0.5,color='red',edgecolors='k',marker='v',label='Training Data')
plt.legend(prop={'size': 16})

b, a = np.polyfit(z, y_Pred2, deg=1)
xseq = np.linspace(0, 10, num=100)
xseq2 = np.linspace(-1,10,num=100)
yseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq2, a + b * xseq2, color="k", lw=2.5);
plt.show()


rf_reg4 = LinearRegression()
sns.set_style("whitegrid", {'axes.grid' : False})

rf_reg4.fit(X_train, y_train)
y_Pred4 = rf_reg4.predict(X_test)
mae2 = mean_absolute_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))
maep2 = mean_absolute_percentage_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))
print("RF mae2",mae2)
print("RF maep2",maep2)
l_pred = rf_reg4.predict(X_train)

# get fit statistics
print('RF training R2 = ' + str(round(rf_reg4.score(X_train, y_train), 3)))
print('RF training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg4.predict(X_train))))
print('RF test R2 = ' + str(round(rf_reg4.score(X_test, y_test), 3)))
print('RF test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg4.predict(X_test))))
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize = (9, 9))
fig.suptitle('Validation for Linear Regressor',fontsize=20)
plt.xlabel("Experimental Bandgap",fontsize=18)
plt.ylabel("Predicted Bandgap ",fontsize=18)
ax.text(-1, 6, 'test R2 = 0.523,training R2 = 0.96', style='italic', bbox={
       'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.text(-1, 6.75, 'test RMSE = 1.454,training RMSE = 0.489', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
ax.scatter(z,y_Pred2, s=60, alpha=1.0, edgecolors="k",label='Testing Data')
ax.scatter(y_train,l_pred,s=60,alpha=0.5,color='red',edgecolors='k',marker='v',label='Training Data')
plt.legend(prop={'size': 16})

b, a = np.polyfit(z, y_Pred2, deg=1)
xseq = np.linspace(0, 10, num=100)
xseq2 = np.linspace(-1,10,num=100)
yseq = np.linspace(0, 10, num=100)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.plot(xseq2, a + b * xseq2, color="k", lw=2.5);
plt.show()


"""