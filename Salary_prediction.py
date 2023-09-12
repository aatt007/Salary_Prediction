import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error



df=pd.read_csv('Salary_Data.csv')
print(df)
print(df.columns)
print(df.info())

print(df.isna().sum())

df.dropna(inplace=True)
print(df.isna().sum())
print(df.describe())

#check the unique values of job titles
print(df['Job Title'].unique())
print(df['Job Title'].nunique())

#check the value counts
print(df['Job Title'].value_counts())

#creating the variables for reducing the number of job titles
job_title_stats=df['Job Title'].value_counts()
job_title_stats_less_than_50=job_title_stats[job_title_stats<=50]
print(job_title_stats_less_than_50.count())

#reducing the number of job titles
df['Job Title']=df['Job Title'].apply(lambda x:'Others' if x in job_title_stats_less_than_50 else x)
print(df['Job Title'])
print(df['Job Title'].nunique())

#check the unique values of Education Level
print(df['Education Level'].unique())
df['Education Level'].replace(["Bachelor's Degree","Master's Degree",'phD'], ["Bachelor's","Master's", 'PhD'], inplace=True)
print(df['Education Level'].unique())
print(df['Education Level'].isna().sum())
#df_educationa_lvel=df['Education Level'].to_frame()
#df_educationa_lvel.to_excel('educationa_level_1.xlsx', index=False)


#check the number of gender
print(df['Gender'].value_counts())

'''

#EDA

#Distribution of categorical variables
fig, ax=plt.subplots(1,2, figsize=(15,5))
sns.countplot(x='Gender', data=df, ax=ax[0])
sns.countplot(x='Education Level', data=df, ax=ax[1])
plt.show()
#first chart reveals that a significant portion of the employees are males, while the second chart indicates that the majority of employees have completed a bachelor's degree

#Distribution of continous variables
fig, ax=plt.subplots(1,3, figsize=(20,5))
sns.histplot(df['Age'], ax=ax[0])
sns.histplot(df['Years of Experience'], ax=ax[1])
sns.histplot(df['Salary'], ax=ax[2])
plt.show()
#chart 1 highlights that the majority of employees fall within the 23 to 37 years age range, emphasizing a youthful workforce.
#chart 2 illustrate employees experience levels with the majority having 1 to 10 years of experience.
#chart 3 demonstrates the salary distribution with most employees earning salaries between 50,000 to 2,00, 000


#Top 10 highest paid jobs
mean_salary_by_job=df.groupby('Job Title')['Salary'].mean().reset_index()
sorted_data=mean_salary_by_job.sort_values(by='Salary', ascending=False)
sns.barplot(x='Salary', y='Job Title', data=sorted_data.head(10)).set(title='Top 10 Highest paid jobs')
plt.show()
#based on this chart we can know Director of data science gets a highest mean salary


#Relationship with target variable
fig, ax=plt.subplots(1,2, figsize=(15, 5))
sns.barplot(x='Gender', y='Salary', data=df, ax=ax[0]).set(title='Relationship between Gender and Salary')
sns.boxplot(x='Education Level', y='Salary', data=df, ax=ax[1]).set(title='Relationship between Edcuation and Salary')
plt.show()
#chart 1 demonstrates the salary distribution among the genders. Employees fromt the other genders get a high salary as compared to the other two genders, but they are very less in count.
#chart2 we can ascertain PHD holders have a high median salary


sns.barplot(x='Education Level', y='Salary', data=df, hue='Gender').set(title='Education level vs Salary vs Gender')
plt.show()
#this chart shows education level and salary among the genders. In all education level category male gets high salary than female. In master's and high school category other gender gets a high salary than males and females.


plt.figure(figsize=(6,5))
sns.scatterplot(x='Age', y='Salary', data=df, hue='Gender').set(title='Relationship between age and Salary')
plt.show()
#this chart shows relationship between age and salary of employees. it illustrates that as age increase salary also increases. gender distribution are also equal.


plt.figure(figsize=(6,5))
sns.scatterplot(x='Years of Experience', y='Salary', data=df, hue='Gender').set(title='Relationship between experience and Salary')
plt.show()
#chart shows that as experience increases salary also increases. Gender distribution are also same.
'''

#Data preprocessing Part2
#Detecting the outliers in salary column usig IQR method
Q1=df.Salary.quantile(0.25)
Q3=df.Salary.quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
print(df[df.Salary>upper])
print(df[df.Salary<lower])
#No outliers found in Salary column

#mapping education level column
education_mapping={'High School':0, "Bachelor's":1, "Master's":2, 'PhD':3}
df['Education Level']=df['Education Level'].map(education_mapping)
#df_educationa_lvel=df['Education Level'].to_frame()
#df_educationa_lvel.to_excel('educationa_level.xlsx', index=False)
print(df['Education Level'])
print(df['Education Level'].isna().sum())

#label encoding the categorical variables
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
#correlation plot
sns.heatmap(df.corr(), annot=True)
#plt.show()
#through this heatmap we can know age, education level, experience are highly correlated to the salary.

#creating dummies for job titles
dummies=pd.get_dummies(df['Job Title'], drop_first=True)
df=pd.concat([df, dummies], axis=1)

#drop job title column
df.drop('Job Title', inplace=True, axis=1)
print(df.head())

#seperate the dataset into features and target
features=df.drop('Salary', axis=1)
print(features.isna().sum())
target=df['Salary']
print(target.isna().sum())

#train test split
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)
print(x_train.shape)
imputer=SimpleImputer(strategy='mean')
x_train_imputed=imputer.fit_transform(x_train)
# Create a dictionary for hyperparameter tuning
model_params = {
    'Linear_Regression': {
        'model': LinearRegression(),
        'params': {

        }
    },
    'Decision_Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [2, 4, 6, 8, 10],
            'random_state': [0, 42],
            'min_samples_split': [1, 5, 10, 20]
        }
    },
    'Random_Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 30, 20, 50, 80]
        }
    }
}

# Hyper parameter tuning through grid search cv
score = []

for model_name, m in model_params.items():
    clf = GridSearchCV(m['model'], m['params'], cv=5, scoring='neg_mean_squared_error')
    clf.fit(x_train, y_train)

    score.append({
        'Model': model_name,
        'Params': clf.best_params_,
        'MSE(-ve)': clf.best_score_
    })
print(pd.DataFrame(score))
#Random Forest has a lowest negative mean squared error which corresponds to the highest positive value of MSE.

#Order of the best models
s=pd.DataFrame(score)
sort=s.sort_values(by='MSE(-ve)', ascending=False)
print(sort)

#Model Evaluation
#Random Forest
'''
# Random Forest model
rfr = RandomForestRegressor(n_estimators=20)
rfr.fit(x_train,y_train)

RandomForestRegressor(n_estimators=20)


rfr.score(x_test,y_test)


y_pred_rfr = rfr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_rfr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_rfr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_rfr,squared=False))

#Mean Squared Error : 81824965.02115172
#Mean Absolute Error : 3522.7619318989396
#Root Mean Squared Error : 9045.715285213862


# Decision Tree model
dtr = DecisionTreeRegressor(max_depth=10,min_samples_split=1,random_state=0)
dtr.fit(x_train,y_train)

DecisionTreeRegressor(max_depth=10, min_samples_split=1, random_state=0)


dtr.score(x_test,y_test)

y_pred_dtr = dtr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_dtr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr,squared=False))

#Mean Squared Error : 161157915.3105976
#Mean Absolute Error : 7325.361557312708
#Root Mean Squared Error : 12694.798750299178
'''
# Linear regression model
lr = LinearRegression()
lr.fit(x_train,y_train)


lr.score(x_test,y_test)

y_pred_lr = lr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_lr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_lr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_lr,squared=False))

#Mean Squared Error : 488076581.1787823
#Mean Absolute Error : 16310.3481305285
#Root Mean Squared Error : 22092.455299915902

#Conclusion
#Among three models it appears that the Random forest model is performing the best in terms of the R2 score and other evaluation metrics used.
#The Random Forest model is the most accurate among these models, with a accuracy of 97.14%
# The Decision Tree model also performs well with a accuracy of 94.34%
#The Linear Regression model has the lowest score, suggesting it may not capture the underlying patterns in the data as effectively as the other models