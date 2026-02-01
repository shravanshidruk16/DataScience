# Step 0 - Understanding the Problem
    1. Find what kind of problem it is ? - Regression , Classification ,etc
    2. What is the target variable ? What should be the final output?
    3. What can be the final performance metrics ?
        a. Classification Problems - Accuray , Precision , F1-score , etc
        b. Regression Problems - RMSE , MAE 
    
# Step 1 - Basic inspection of DATA
```python
df.head()     # Check top 5 rows
df.info()     # Basic info of all the data columns and there data types
df.describe() # Descriptive statistics of the data
df.shape      # Check number of rows and columns in the dataset
df.tail()     # Check last 5 rows
df.columns()  # Check different columns and there names
```

# Step 2 - Data Cleaning

## 2.1 - Dealing with **missing values**
```python
df.isnull().sum() # calculate missing values per column
```
### 2.1.a - Use Imputations in order to handle missing values
```python
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy="mean") # This simply means that we add mean of the column to the place where  there is no value assignend in numerical column
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
cat_imputer = SimpleImputer(strategy="most_frequent") # This simply means that where the categorical values are missing add the most frequent values there
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
```

## 2.2 - Find duplicates and delete them
```python
df[df.duplicated()]  # to find duplicate rows
df.drop_duplicates() # to drop duplicate rows and return a new dataframe
```

## 2.3 - Fix data types
```python
df.info() # always check the data types first before fixing them
df['age'] = df['age'].astype(int) # sample use case
```

# Step 3 - EDA (Exploratory Data Analysis)

## 3.1 - Univariate Analysis
1. To check distribution = Plot **Histograms**
```python
df.hist(bins=50,figsize=(20,15))
plt.tight_layout()
plt.show()
```
2. To check outliers = Plot **Box-Plots** 
```python
sns.boxplot(
    data=df,
    x = 'target_feature', # this type of arrange gives readable box plot
    y = 'input_feature'
)
```

3. To check Output/Dependent feature
```python
plt.figure(figsize=(10,6))
plt.pie(df['target'].value_counts(),labels=['Yes','No'],autopct='%1.1f%%')
plt.title('Classes Distribtuion')
``` 

## 3.2 - Bivariate Analysis
1. Plot a **Correlation Heatmap**
```python 
plt.figure(figsize=(10,8))  # makes it readable

sns.heatmap(
    df_copy.corr(), 
    annot=True,          # shows correlation values
    fmt=".2f",           # show 2 decimal places
    cmap=plt.cm.CMRmap_r,# best colour theme for heatmap plot
    linewidths=0.5,      # lines between boxes
    linecolor='black'
)

plt.title("Correlation Matrix Heatmap", fontsize=14)
plt.show()
```
2. Plot **Target vs Features**
```python
sns.countplot(x='input_feature', hue='output_feature', data=df)
plt.title("Individual Feature vs Target Analysis")
plt.show()
```

## 3.3 - Outlier Detection
* Use **IQR , Z-score , Box-Plots , etc**
```python
# IQR
Q1 = df['marks'].quantile(0.25)
Q3 = df['marks'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['marks'] < lower_bound) | (df['marks'] > upper_bound)]
print(outliers_iqr)

# Z-Score
from scipy import stats
z_scores = np.abs(stats.zscore(df['marks']))
outliers_z = df[z_scores > 3]
print(outliers_z)

# BOX plots
sns.boxplot(x=df['marks'])
plt.title("Box Plot for Outlier Detection")
plt.show()
```

## 3.4 - Multivariate Analysis
* Plot **PAIRPLOTS**
```python
# Style settings (makes it look professional)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
# Example dataset (replace with your dataframe)
df = sns.load_dataset("iris")
# Pairplot
pair = sns.pairplot(
    df,
    hue="species",          # color by category (very powerful)
    diag_kind="kde",        # smooth distribution on diagonal
    markers=["o", "s", "D"],# different shapes per class
    plot_kws={"alpha":0.7, "s":60},  # point size & transparency
    corner=True             # only lower triangle (clean look)
)
pair.fig.suptitle("Pairplot of Features", y=1.02, fontsize=16)
plt.show()
```

# Step 4 - Feature Engineering

## 4.1 - Important Features Creation
    eg - Creating AGE column from DOB column

## 4.2 - Feature Transformation
    Skewed Data = Use Log transformation
    Wide Range Data = Use Power transformation

## 4.3 - Encoding Categorical Data
    Ordinal Data = Use Label Encoding
    Nominal Data = Use One-Hot-Encoding
    High Categories = Use frequency wise selection and Target Encoding

## 4.4 - Feature Scaling
    Linear/Logistic/SVM/KNN = Need of Scaling
    Tree Models / RF / XGBoost = No need of Scaling

## 4.5 - Handling Outliers
    Removal , Clip , Transformations

## 4.6 - Handling Imbalanced Dataset
    1. Undersampling
    2. Oversampling
    3. SMOTE (Synthetic Minority Oversampling Technique)

# Step 5 - Feature Selection

## 5.1 - Variance Thresholding (Based on Variance of a feature)

```python
# Removing features/columns which have a low Variance
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0)
```

## 5.2 - Pearson's Correlation (Correlation Based)
```python
# Removing highly correlated features
# Note : We remove the features which are correlated among themselves
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)

    return col_corr
correlated_features = correlation(x_train,0.7) # 70% of correlated features
# Note : Only apply the test on x_train after finding columns drop them from both i.e x_train , x_test
for feature in correlated_features:
    print(feature)
x_train.drop(correlated_features,axis=1,inplace=True)
x_test.drop(correlated_features,axis=1,inplace=True)
```

## 5.3 - Mutual Information gain 
```python
from sklearn.feature_selection import mutual_info_classif

info_gain = mutual_info_classif(df, y)

ig_scores = pd.Series(info_gain, index=df.columns).sort_values(ascending=False)
print(ig_scores.head())
# Note Higher Score = More Important Feature
```

## 5.4 - Mutual Information gain using Regression (Model Based)
```python
from sklearn.feature_selection import mutual_info_regression
import numpy as np

# Fake continuous target example
y_reg = np.random.rand(len(df))

mi_scores = mutual_info_regression(df, y_reg)

mi_series = pd.Series(mi_scores, index=df.columns).sort_values(ascending=False)
print(mi_series.head())
# Note
```

## 5.5 - Chi^2 test of Feature Selection (Statistical Based)
```python 
from sklearn.feature_selection import chi2
f_values , p_values = chi2(x_train,y_train)
# Note fscore is directly proportional to importance of the feature in ML model training wheres p_values are inversely proportional to the importance of a feature in training of a ML model
```

## 5.6 - Recursive feature elimination
```python 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)

rfe = RFE(model, n_features_to_select=5)
rfe.fit(df, y)

selected_features = df.columns[rfe.support_]
print(selected_features)
```

## 5.7 - Select K - Best
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(df, y)

selected = df.columns[selector.get_support()]
print(selected)
```

# Step 6 - Splliting data into Train and Test set

## 6.1 - Dividing features into **Independent** and **Dependent**
```python
# Independent Features division
x = df.drop('target_feature',axis=1) # if there are multiple input features
x = df[['input_feature']] # if there is single input feature use this

# Dependent Feature division
y = df['target_variable']
```

## 6.2 - Making a train test split 

```python
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)
```

## 6.3 - Scaling data as per the need - Looking to the approach or ML type
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test) # in order to avoid data leakage problem
```

# Step 7 - Model Building
eg. Building a Linear Regression Model
```python
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)
```

# Step 8 - Plot the results
```python
plt.scatter(x_train,y_train)
plt.plot(x_train,regression.predict(x_train))
plt.show()
```

# Step 9 - Prediction Analysis / Model Evaluation
    Now here we check the scores of our model accuracy

    Regression Models = Calculate MAE,RMSE,R2_score,Adj_R2_score
    Classification Models = Calculate Accuracy , Precision , Recall , F1_score, ROC , AUC
```python
# For a linear regression model we use : 
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
y_pred = regression.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
R2_score = r2_score(y_test,y_pred)
adj_r2_score = 1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print(f"MSE: {mse}\nMAE:{mae}\nRMSE:{rmse}\nR2 Score: {R2_score}\nAdjusted R2 Score: {adj_r2_score}")
```
# Step 10 - Start Using Pipelines
# Step 11 - Hyper Parameter Tuning
# Step 12 - Cross Validation
# Step 13 - Saving Final Model
# Step 14 - Deployment

### Connect With Me : Shravan Shidruk
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shravan-shidruk)
<br>
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/shravanshidruk16)


