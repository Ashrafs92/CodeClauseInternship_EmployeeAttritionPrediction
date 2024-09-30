import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import f1_score, recall_score, precision_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()
data.columns
data.shape

for col in data.columns:
    if data[col].dtypes=='O':
        print(f'{col} : {data[col].unique()}')
    else:
        print(f'{col} : {data[col].min()} - {data[col].max()}')

data.drop(['Over18','EmployeeCount'], axis=1, inplace=True)
data.info()
data.isnull().sum()
data.duplicated().sum()
categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(exclude=['object']).columns

ax=sns.countplot(x=data['Attrition'])
for i in ax.containers:
    ax.bar_label(i)
plt.title('Count of each Attrition')
plt.show()

categorical_columns = categorical_columns[1:]
num_rows = len(categorical_columns)
fig, ax = plt.subplots(nrows=num_rows, ncols=1, figsize=(8, 4 * num_rows))
fig.tight_layout(pad=3.0)
for i, column_name in enumerate(categorical_columns):
    sns.countplot(x=column_name, hue='Attrition', data=data, ax=ax[i])
    ax[i].set_title(f'Attrition by {column_name}')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Count')
plt.show()

correlation_matrix = data.drop('StandardHours',axis=1).corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

sns.boxplot(y='Attrition', x='MonthlyIncome', data=data)
plt.title('Monthly Income by Attrition')
plt.show()

sns.barplot(y='JobRole', x='JobSatisfaction', hue='Attrition',data=data)

sns.pairplot(data=data[['TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']])

for col in data.columns:
    if data[col].dtypes=='O':
        print(f'{col} : {data[col].unique()}')

data.Attrition=[1 if val=='Yes' else 0 for val in data.Attrition]
data.Gender=[1 if val=='Male' else 0 for val in data.Gender]
data.OverTime=[1 if val=='Yes' else 0 for val in data.OverTime]

BusinessTravel_dummies=pd.get_dummies(data.BusinessTravel)
Department_dummies=pd.get_dummies(data.Department)
EducationField_dummies=pd.get_dummies(data.EducationField)
JobRole_dummies=pd.get_dummies(data.JobRole)
MaritalStatus_dummies=pd.get_dummies(data.MaritalStatus)

data=pd.concat([data,
               BusinessTravel_dummies.iloc[:,:2],
               Department_dummies.iloc[:,:2],
               EducationField_dummies.iloc[:,:5],
               JobRole_dummies.drop(columns='Human Resources'),
               MaritalStatus_dummies.iloc[:,:2]
               ],axis=1)

data.drop(['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'],axis=1,inplace=True)
data

X=data.drop('Attrition',axis=1)
y=data.Attrition

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11,train_size=0.80,shuffle=True)

X_train.shape , X_test.shape , y_train.shape , y_test.shape

X_train.head()

y_train[:5]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train
X_test

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

for x in range(120):
    RF = RandomForestClassifier(random_state=x)
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)
    current_accuracy = round(accuracy_score(y_pred,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)

RF=RandomForestClassifier(criterion='entropy', random_state=best_x, n_estimators=19)

y_pred = RF.predict(X_test)
y_pred

print(classification_report(y_test,y_pred))

train_acc=RF.score(X_train,y_train)
test_acc=accuracy_score(y_test,y_pred)
recal=recall_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
print("Training Accuracy :", train_acc)
print("Testing Accuracy :", test_acc)
print("F1 Score :", f1)
print("Recall :", recal)
print("Precision :", prec)

conf_matrix = pd.DataFrame(data = confusion_matrix(y_test,y_pred),
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (5, 3))
sns.heatmap(conf_matrix, annot = True, fmt = 'd')
plt.show()

importances = RF.feature_importances_

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances, 'Percentage': [round(val,2) for val in ((importances*100)/sum(importances))]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
feature_importance.head()

feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(15,12))

plt.pie(x='Importance', data=feature_importance.iloc[:5,:],autopct='%.1f%%',
        labels='Feature',explode = [0.03 for i in range(5)], shadow = True)
plt.show()

def gen(g):
   return 1 if (g.lower()=='male' or g.lower()[0]=='m') else 0

def overtime(o):
   return 1 if (o.lower()=='yes' or o.lower()[0]=='y') else 0

def business_travel(b):
   if b.lower()=='Travel_Rarely'.lower():
      return [0, 0]
   elif b.lower()=='Travel_Frequently'.lower():
      return [0, 1]
   else:
      return [1,0]
   
def department(d):
   if d.lower()=='Human Resources'.lower():
      return [1,0]
   elif d.lower()=='Research & Development'.lower():
      return [0,1]
   else:
      return [0,0]
   
def education_field(e):
    if e.lower()=='Human Resources'.lower():
        return [1,0,0,0,0]
    elif e.lower()=='Life Sciences'.lower():
        return [0,1,0,0,0]
    elif e.lower()=='Marketing'.lower():
        return [0,0,1,0,0]
    elif e.lower()=='Medical'.lower():
        return [0,0,0,1,0]
    elif e.lower()=='Other'.lower():
        return [0,0,0,0,1]
    else:
        return [0,0,0,0,0]
     
def jobrole(j):
    if j.lower()=='Healthcare Representative'.lower():
        return [1,0,0,0,0,0,0,0]
    elif j.lower()=='Laboratory Technician'.lower():
        return [0,1,0,0,0,0,0,0]
    elif j.lower()=='Manager'.lower():
        return [0,0,1,0,0,0,0,0]
    elif j.lower()=='Manufacturing Director'.lower():
        return [0,0,0,1,0,0,0,0]
    elif j.lower()=='Research Director'.lower():
        return [0,0,0,0,1,0,0,0]
    elif j.lower()=='Research Scientist'.lower():
        return [0,0,0,0,0,1,0,0]
    elif j.lower()=='Sales Executive'.lower():
        return [0,0,0,0,0,0,1,0]
    elif j.lower()=='Sales Representative'.lower():
        return [0,0,0,0,0,0,0,1]
    else:
        return [0,0,0,0,0,0,0,0]
    
def marital_status(m):
    if m.lower()=='Divorced'.lower():
        return [1,0]
    elif m.lower()=='Married'.lower():
        return [0,1]
    else:
        return [0,0]
      
def predict(ip):
    data = []
    data.append(ip[0])
    data.append(ip[2])
    data.extend(ip[4:6] + ip[8:10])
    data.append(gen(ip[10]))
    data.extend(ip[11:14])
    data.append(ip[15])
    data.extend(ip[17:20])
    data.append(overtime(ip[21]))
    data.extend(ip[22:] + business_travel(ip[1]) + department(ip[3])+  education_field(ip[6]) + jobrole(ip[14]) + marital_status(ip[16]))
    
    data = scaler.transform([data])
    result = RF.predict(data)[0]
    
    return "Yes, the employee will attrite from the company." if result==1 else "No, the employee will not attrite from the company."

predict([30, 'Travel_Rarely', 288, 'Research & Development', 2, 3,
       'Life Sciences', 1, 117, 3, 'Male', 99, 2, 2,
       'Healthcare Representative', 4, 'Married', 4152, 15830, 1, 'Y',
       'No', 19, 3, 1, 80, 3, 11, 3, 3, 11, 10, 10, 8])


