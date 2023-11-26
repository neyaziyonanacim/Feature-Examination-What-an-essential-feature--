"""
Author : Mustafa Gürkan Çanakçi
LinkedIn : https://www.linkedin.com/in/mgurkanc/
"""

# Project Name : Diabet Feature Engineering

# Business Problem

# In this project, we will develop a machine learning model that
# can predict whether Pima Indian Women in the dataset have diabetes or not.
# Before modelling , we will make the exploratory data analysis and feature engineering for its dataset.


# Content of Variables:
# Pregnancies - Number of pregnancies
# Glucose - 2-hour plasma glucose concentration in the oral glucose tolerance test
# BloodPressure - Diastolic Blood Pressure
# SkinThickness - Thickness of Skin
# Insulin- 2-hour serum insulin
# DiabetesPedigreeFunction - indicates the function which scores likelihood of diabetes based on family history.
# BMI - Body Mass Index
# Age - Age
# Outcome - Diabetic ( 1 or 0 )

########################################################################################
#                           1.EXPLORATORY DATA ANALYSIS                                #
########################################################################################

# * 1.1.Importing necessary libraries*

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


####################################################################################
# * 1.2.Read the dataset*
####################################################################################
df = pd.read_csv("..\pythonProgramlama\python_for_data_science\dataset\diabetes.csv")


# * Checking the data*
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)


# Conclusion:
# There are only numerical variables in this dataset.
# 768 observations, 9 variable available(1 dependent)
# Under normal circumstances, it seems that there are no missing valuesin the data set,
# but there may be missing values hidden in the data of the variables here.

###############################################
# * 1.3. Checking the missing values in the dataset*
###############################################
dimension_variable = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[dimension_variable] = df[dimension_variable].replace(0,np.NaN)

df.isnull().sum()

# Pregnancies                   0
# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11
# DiabetesPedigreeFunction      0
# Age                           0
# Outcome                       0

####################################################################################
# * 1.4.Define a Function to grab the Numerical and Categorical variables of its dataset*
####################################################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 768
# Variables: 9
# cat_cols: 1
# num_cols: 8
# cat_but_car: 0
# num_but_cat: 1

cat_cols
# Out[14]: ['Outcome']

num_cols
# ['Pregnancies',
#  'Glucose',
#  'BloodPressure',
#  'SkinThickness',
#  'Insulin',
#  'BMI',
#  'DiabetesPedigreeFunction',
#  'Age']

#####################################
# * 1.5.Target Variable Analysis
#####################################

df["Outcome"].value_counts()
# 0    500
# 1    268
# Name: Outcome, dtype: int64


def target_summary_with_num(dataframe,target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n")
    print("###################################")

for col in num_cols:
    target_summary_with_num(df,"Outcome",col)

###############################################
# * 1.6.Outliers Analysis
###############################################

# Define a Function about outlier threshold for data columns
def outlier_th(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Define a Function about checking outlier for data columns
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Define a Function about replace with threshold for data columns
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
# Pregnancies True
# Glucose False
# BloodPressure True
# SkinThickness True
# Insulin True
# BMI True
# DiabetesPedigreeFunction True
# Age True

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))
# Pregnancies False
# Glucose False
# BloodPressure False
# SkinThickness False
# Insulin False
# BMI False
# DiabetesPedigreeFunction False
# Age False

####################################################################################
# * 1.7.The Missing Values Analysis
####################################################################################

# Define a Function about missing values for dataset columns
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
#                n_miss  ratio
# Insulin           374 48.700
# SkinThickness     227 29.560
# BloodPressure      35  4.560
# BMI                11  1.430
# Glucose             5  0.650


#########################
# * Correlation Analysis
#########################

dimension_variable = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
corr_matrix = df[dimension_variable].corr()
corr_matrix
#                Glucose  BloodPressure  SkinThickness  Insulin   BMI
# Glucose          1.000          0.225          0.217    0.614 0.235
# BloodPressure    0.225          1.000          0.241    0.115 0.295
# SkinThickness    0.217          0.241          1.000    0.200 0.675
# Insulin          0.614          0.115          0.200    1.000 0.266
# BMI              0.235          0.295          0.675    0.266 1.000


# Insulin - Glucose arasında pozitif yönlü ilişki (yüksek)
# BMI- SkinThickness arasında pozitif yönlü ilişki (yüksek)
# BMI - BloodPressure arasında pozitif yönlü ilişki(orta)

# Visulization of Correlation Matrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots()
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.coolwarm)

# making the colorbar on the side
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])

# making the labels
labels = ['']
for column in dimension_variable:
    labels.append(column)
    labels.append('')
ax.set_yticklabels(labels, minor=False)
ax.set_xticklabels(labels, minor=False)

plt.show(block=True)


########################################################################################
#                           2.FEATURE ENGINEERING                                      #
########################################################################################


###############################################
# * 2.1.Processing for Missing Values and Outliers
###############################################
df.isnull().sum()
# Out[46]:
# Pregnancies                   0
# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11
# DiabetesPedigreeFunction      0
# Age                           0
# Outcome                       0

na_cols = missing_values_table(df, True)
#                n_miss  ratio
# Insulin           374 48.700
# SkinThickness     227 29.560
# BloodPressure      35  4.560
# BMI                11  1.430
# Glucose             5  0.650


# Define a Function about comparing target variable with missing values
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)

# Conclusion:
# We examined the missing values of each variable according to the target variable.
# So we decided to apply different methods in order to fill na values according to state of each variable.

# Fill the missing values of some variables with the median
df["Glucose"] = df["Glucose"].fillna(df["Glucose"].median())
df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].median())
df["BMI"] = df["BMI"].fillna(df["BMI"].median())



# Fill the missing values of "Insulin" and "SkinThickness variables by implementing the KNN method

dff = pd.get_dummies(df[["Insulin","SkinThickness"]], drop_first=True)

dff.head()

# # Standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# # Implement the KNN method
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# # Undo the standardization of these variables
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["Insulin"] = dff["Insulin"]
df["SkinThickness"]= dff["SkinThickness"]

df.isnull().sum()


###############################################
# * 2.2.Creating New Feature Interactions
###############################################

df.head()

# # Create a Glucose Categorical variable
df.loc[(df['Glucose'] < 70), 'GLUCOSE_CAT'] ="hipoglisemi"
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100) , 'GLUCOSE_CAT'] ="normal"
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] < 126) , 'GLUCOSE_CAT'] ="imparied glucose"
df.loc[(df['Glucose'] >= 126), 'GLUCOSE_CAT'] ="hiperglisemi"

df.head()

df.groupby("GLUCOSE_CAT").agg({"Outcome": ["mean","count"]})

# Create the Age Categorical variable
df.loc[(df['Age'] >= 18) & (df['Age'] < 30) , 'AGE_CAT'] ="young_women_"
df.loc[(df['Age'] >= 30) & (df['Age'] < 45) , 'AGE_CAT'] ="mature_women"
df.loc[(df['Age'] >= 45) & (df['Age'] < 65) , 'AGE_CAT'] ="middle_age"
df.loc[(df['Age'] >= 65) & (df['Age'] < 75) , 'AGE_CAT'] ="old_age"
df.loc[(df['Age'] >= 75) , 'AGE_CAT'] ="elder_age"

df.groupby("AGE_CAT").agg({"Outcome": ["mean","count"]})
#              Outcome
#                 mean count
# AGE_CAT
# mature_adult   0.494   239
# middle_age     0.530   117
# old_age        0.250    16
# young_adult    0.212   396

# Create a Body Mass Index(BMI) Categorical variable
df.loc[(df['BMI'] < 16), 'BMI_CAT'] ="overweak"
df.loc[(df['BMI'] >= 16) & (df['BMI'] < 18.5) , 'BMI_CAT'] ="weak"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25) , 'BMI_CAT'] ="normal"
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30) , 'BMI_CAT'] ="overweight"
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 70) , 'BMI_CAT'] ="obese"


df.groupby("BMI_CAT").agg({"Outcome": ["mean","count"]})
#            Outcome
#               mean count
# BMI_CAT
# 1stObese     0.438   235
# 2ndObese     0.453   212
# 3rdObese     0.611    36
# normal       0.069   102
# overweight   0.223   179
# weak         0.000     4

df.head()

# # Create a Diastolic Blood Pressure Categorical variable
df.loc[(df['BloodPressure'] < 70)  , 'DIASTOLIC_CAT'] ="low"
df.loc[(df['BloodPressure'] >= 70) & (df['BMI'] < 90) , 'DIASTOLIC_CAT'] ="normal"
df.loc[(df['BloodPressure'] >= 90 ) , 'DIASTOLIC_CAT'] ="high"

df.groupby("DIASTOLIC_CAT").agg({"Outcome": ["mean","count"]})


df["Insulin"].unique()

# # Create a Insulin Categorical variable
df.loc[(df['Insulin'] < 120)  , 'INSULIN_CAT'] ="normal"
df.loc[(df['Insulin'] >= 120) , 'INSULIN_CAT'] ="abnormal"

df.groupby("INSULIN_CAT").agg({"Outcome": ["mean","count"]})


# # Create a Pregnancies Categorical variable
df.loc[(df['Pregnancies'] == 0)  , 'PREG_CAT'] ="unpregnant"
df.loc[(df['Pregnancies'] > 0 ) & (df['Pregnancies'] <= 5)  , 'PREG_CAT'] ="normal"
df.loc[(df['Pregnancies'] > 10 )  , 'PREG_CAT'] ="very high"

df.groupby("PREG_CAT").agg({"Outcome": ["mean","count"]})
# Out[43]:
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction    Age  Outcome   GLUCOSE_CAT       AGE_CAT     BMI_CAT DIASTOLIC_CAT INSULIN_CAT    PREG_CAT
# 0        6.000  148.000         72.000         35.000      NaN 33.600                     0.627 50.000        1  hiperglisemi    middle_age   1st_Obese        normal         NaN        high
# 1        1.000   85.000         66.000         29.000      NaN 26.600                     0.351 31.000        0        normal  mature_women  overweight           low         NaN      normal
# 2        8.000  183.000         64.000            NaN      NaN 23.300                     0.672 32.000        1  hiperglisemi  mature_women      normal           low         NaN        high
# 3        1.000   89.000         66.000         23.000   94.000 28.100                     0.167 21.000        0        normal  young_women_  overweight           low      normal      normal
# 4        0.000  137.000         40.000         35.000  168.000 43.100                     1.200 33.000        1  hiperglisemi  mature_women   2nd_Obese           low    abnormal  unpregnant



###############################################
# * 2.3.Processing Encoding and One-Hot Encoding
###############################################


le = LabelEncoder()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
# Out[48]:
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction    Age  Outcome  GLUCOSE_CAT_hipoglisemi  GLUCOSE_CAT_imparied glucose  GLUCOSE_CAT_normal  AGE_CAT_middle_age  AGE_CAT_old_age  AGE_CAT_young_women_  BMI_CAT_2nd_Obese  BMI_CAT_3rd_Obese  BMI_CAT_normal  BMI_CAT_overweight  BMI_CAT_weak  DIASTOLIC_CAT_low  DIASTOLIC_CAT_normal  INSULIN_CAT_1  INSULIN_CAT_2  PREG_CAT_normal  PREG_CAT_unpregnant  PREG_CAT_very high
# 0        6.000  148.000         72.000         35.000      NaN 33.600                     0.627 50.000        1                        0                             0                   0                   1                0                     0                  0                  0               0                   0             0                  0                     1              0              1                0                    0                   0
# 1        1.000   85.000         66.000         29.000      NaN 26.600                     0.351 31.000        0                        0                             0                   1                   0                0                     0                  0                  0               0                   1             0                  1                     0              0              1                1                    0                   0
# 2        8.000  183.000         64.000            NaN      NaN 23.300                     0.672 32.000        1                        0                             0                   0                   0                0                     0                  0                  0               1                   0             0                  1                     0              0              1                0                    0                   0
# 3        1.000   89.000         66.000         23.000   94.000 28.100                     0.167 21.000        0                        0                             0                   1                   0                0                     1                  0                  0               0                   1             0                  1                     0              1              0                1                    0                   0
# 4        0.000  137.000         40.000         35.000  168.000 43.100                     1.200 33.000        1                        0                             0                   0                   0                0                     0                  1                  0               0                   0             0                  1                     0              0              0                0                    1                   0


###############################################
# * 2.4.Standardization for numerical variables
###############################################

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()



cat_cols, num_cols, cat_but_car = grab_col_names(df)

#########################################################
# Bonus : Rare Analyser and Rare Encoding
#########################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df



df = rare_encoder(df, 0.01)

rare_analyser(df, "Outcome", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

df.drop(useless_cols, axis=1, inplace=True)

df.shape

#########################################################
# * 2.5.Create Modelling
#########################################################


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# Out[81]: 0.7705627705627706

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve

class_report=classification_report(y_test,y_pred)
print(class_report)


auc = roc_auc_score(y_test, y_pred)
auc

cm = confusion_matrix(y_test, y_pred)
predicted_probab_log = rf_model.predict_proba(X_test)
predicted_probab_log = predicted_probab_log[:, 1]
fpr, tpr, _ = roc_curve(y_test, predicted_probab_log)

from matplotlib import pyplot
pyplot.plot(fpr, tpr, marker='.', label='Random Forest')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show(block=True)

# Cross Validation
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier().fit(X_train, y_train)

cv_results = cross_validate(rf_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("test_accuracy: ", cv_results['test_accuracy'].mean())
print("test_f1: ", cv_results['test_f1'].mean())
print("test_roc_auc: ", cv_results['test_roc_auc'].mean())


########################################################################################
#                                   3.BONUS                                            #
########################################################################################

# Importing Sklearn Libraries and Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Logistic Regression
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
LR=LogisticRegression()
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)
class_report=classification_report(y_test,y_pred)
print(class_report)

auc = roc_auc_score(y_test, y_pred)
auc

cm = confusion_matrix(y_test, y_pred)
predicted_probab_log = LR.predict_proba(X_test)
predicted_probab_log = predicted_probab_log[:, 1]
fpr, tpr, _ = roc_curve(y_test, predicted_probab_log)

from matplotlib import pyplot
pyplot.plot(fpr, tpr, marker='.', label='Logistic Regression')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show(block=True)

log_model = LogisticRegression().fit(X, y)
y_pred = log_model.predict(X)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show(block=True)

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# 5- Fold Cross Validation
log_model = LogisticRegression().fit(X_train, y_train)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("test_accuracy: ", cv_results['test_accuracy'].mean())
print("test_f1: ", cv_results['test_f1'].mean())
print("test_roc_auc: ", cv_results['test_roc_auc'].mean())

################################################################
# Comparison 8-Machine Learning Algoritms and Feature Importance
################################################################

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
primitive_success=[]
model_names=[]

def ML(algName):
    # Model Building / Training
    model=algName().fit(X_train,y_train)
    model_name=algName.__name__
    model_names.append(model_name)
    # Prediction
    y_pred=model.predict(X_test)
    # primitive-Success / Verification Score
    from sklearn.metrics import accuracy_score
    primitiveSuccess=accuracy_score(y_test,y_pred)
    primitive_success.append(primitiveSuccess)
    return  primitive_success,model_names,model


models=[KNeighborsClassifier,SVC,MLPClassifier,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier,XGBClassifier,LGBMClassifier]
for i in models:
    ML(i)

classification=pd.DataFrame( primitive_success,columns=
                                 ['accuracy_Score'],index=[model_names]).sort_values(by='accuracy_Score',ascending=False)
classification


# Plot Importance
def plot_importance(model, features,modelName, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title('Features'+ ' - ' + modelName.__name__ )
    plt.tight_layout()
    plt.show(block=True)

for i in models[3:]:
    model=i().fit(X_train,y_train)
    plot_importance(model, X_train,i)



