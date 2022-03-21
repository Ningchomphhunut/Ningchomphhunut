import numpy
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pydot
import graphviz

warnings.filterwarnings('ignore')

df = pd.read_csv('car_evaluation.csv') #นำเข้าชุดข้อมูล
print('Dimensions:',df.shape)
df.head()
df.describe()  #ข้อมูลสรุปโดยย่อของชุดข้อมูล
df.info()  #ข้อมูลเพิ่มเติมเกี่ยวกับชุดข้อมูล

for i in df.columns:  #แสดงคอลัมน์ทั้งหมดเป็นหมวดหมู่ โดยตรวจสอบค่าที่ไม่ซ้ำกันของแต่ละคอลัมน์
    print(df[i].unique(),"\t",df[i].nunique())

for i in df.columns: #ตรวจสอบว่าหมวดหมู่ที่ไม่ซ้ำกันเหล่านี้มีการกระจายระหว่างคอลัมน์ต่าง ๆ อย่างไร
    print(df[i].value_counts())
    print()

# แสดงกราฟของแต่ละคลาสว่ามีจำนวนเท่าใด
plt.title("Class distribution", color="purple", fontsize=17)
sns.countplot(df['class'])
plt.show()

for i in df.columns[:-1]:  # แสดงกราฟที่จำแนกของแต่ละแอตทริบิวต์
    plt.figure(figsize=(12, 6))
    plt.title("For feature '%s'" % i)
    sns.countplot(df[i], hue=df['class'])

from sklearn.preprocessing import LabelEncoder #การเข้ารหัสจำลอง(ข้อมูลเป็นตัวเลข)
le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform(df[i])
    plt.show()

cor_mat = df.corr()  # แสดงตารางแมทริคสหสัมพันธ์
fig = plt.figure(figsize=(10,6)) #แสดงตารางแมทริคสหสัมพันธ์
sns.heatmap(df.corr(),annot=True)
plt.show()


x=df[df.columns[:-1]]  #การสร้างตัวแปร(x)และ(y)
y=df['class']
x.head(2)

from sklearn.model_selection import train_test_split #แยกข้อมูลออกเป็นชุดฝึกและชุดทดสอบ
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

#เปรียบเทียบ model ต่าง ๆ
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#1 KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
KNN =KNeighborsClassifier (n_jobs=-1)
KNN.fit(xtrain,ytrain)
y_pred = KNN.predict(xtest)
KNN.score(xtest,ytest)
from sklearn.metrics import accuracy_score
Model_accuracy = accuracy_score(y_pred,ytest)
print(Model_accuracy)
print(classification_report(ytest, y_pred))

avg_score = []
for k in range(2,30):
    KNN = KNeighborsClassifier(n_jobs=-1,n_neighbors=k)
    score = cross_val_score(KNN, xtrain, ytrain, cv=5, n_jobs=-1, scoring='accuracy')
    avg_score.append(score.mean())
plt.figure(figsize=(12,8))
plt.plot(range(2,30),avg_score)
plt.xlabel("n_neighbours")
plt.ylabel("Accuracy")
#plt.xticks(range(2,30,2))
plt.show()
#ดังนั้นการจำแนกด้วยอัลกอริธึม KNeighborsClassifier สามารถบรรลุความแม่นยำ 90%

#2 LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)
y_pred = logreg.predict(xtest)
logreg.score(xtest,ytest)
from sklearn.metrics import accuracy_score
Model_accuracy = accuracy_score(y_pred,ytest)
print(Model_accuracy)
print(classification_report(y_pred,ytest))

avg_score = []
for l in range(2,30):
    logreg = LogisticRegression()
    score = cross_val_score(logreg, xtrain, ytrain, cv=5, scoring='accuracy')
    avg_score.append(score.mean())
plt.figure(figsize=(12,8))
plt.plot(range(2,30),avg_score)
plt.xlabel("LogisticRegression")
plt.ylabel("Accuracy")
#plt.xticks(range(2,30,2))
plt.show()
#ดังนั้นการจำแนกด้วยอัลกอริธึม LogisticRegression สามารถบรรลุความแม่นยำ 66%

#3 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=450)
RF.fit(xtrain,ytrain)
y_pred = RF.predict(xtest)
RF.score(xtest,ytest)
from sklearn.metrics import accuracy_score
Model_accuracy = accuracy_score(y_pred,ytest)
print(Model_accuracy)
print(classification_report(ytest, y_pred))

avg_score = []
for r in range(2,30):
    RF = RandomForestClassifier(n_estimators=450)
    score = cross_val_score(RF, xtrain, ytrain, cv=5, scoring='accuracy')
    avg_score.append(score.mean())
plt.figure(figsize=(12,8))
plt.plot(range(2,30),avg_score)
plt.xlabel("RandomForestClassifier")
plt.ylabel("Accuracy")
#plt.xticks(range(2,30,2))
plt.show()
#ดังนั้นการจำแนกด้วยอัลกอริธึม RandomForestClassifier สามารถบรรลุความแม่นยำ 96%


#4 GaussianNB
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain,ytrain)
y_pred = NB.predict(xtest)
NB.score(xtest,ytest)
from sklearn.metrics import accuracy_score
Model_accuracy = accuracy_score(y_pred,ytest)
print(Model_accuracy)
print(classification_report(ytest, y_pred))

avg_score = []
for n in range(2,30):
    NB = GaussianNB()
    score = cross_val_score(NB, xtrain, ytrain, cv=5, scoring='accuracy')
    avg_score.append(score.mean())
plt.figure(figsize=(12,8))
plt.plot(range(2,30),avg_score)
plt.xlabel("GaussianNB")
plt.ylabel("Accuracy")
#plt.xticks(range(2,30,2))
plt.show()
#ดังนั้นการจำแนกด้วยอัลกอริธึม GaussianNB สามารถบรรลุความแม่นยำ 64%


#5 SVC
from sklearn.svm import SVC
SVC = SVC()
SVC.fit(xtrain,ytrain)
y_pred = SVC.predict(xtest)
SVC.score(xtest,ytest)
from sklearn.metrics import accuracy_score
Model_accuracy = accuracy_score(y_pred,ytest)
print(Model_accuracy)
print(classification_report(ytest, y_pred))

avg_score = []
for s in range(2,30):
    SVC = SVC
    score = cross_val_score(SVC, xtrain, ytrain, cv=5, scoring='accuracy')
    avg_score.append(score.mean())
plt.figure(figsize=(12,8))
plt.plot(range(2,30),avg_score)
plt.xlabel("SVC")
plt.ylabel("Accuracy")
#plt.xticks(range(2,30,2))
plt.show()
#ดังนั้นการจำแนกด้วยอัลกอริธึม SVC สามารถบรรลุความแม่นยำ 89%


fig, axes = plt.subplots(2,3, figsize=(13,8))  #แสดงกราฟคุณลักษณะแต่ละอย่างส่งผลต่อคลาสของรถยนต์ทั้ง6แอททริบิว
axes[0,0].set_title('Buying')
sns.countplot(x = 'class', hue='buying', data = df, ax=axes[0,0])
axes[0,1].set_title('Maintenance')
sns.countplot(x = 'class', hue='maint', data = df, ax=axes[0,1])
axes[1,1].set_title('Doors')
sns.countplot(x = 'class', hue='doors', data = df, ax=axes[1,1])
axes[1,0].set_title('Safety')
sns.countplot(x = 'class', hue='safety', data = df, ax=axes[1,0])
axes[1,2].set_title('Luggage Boot')
sns.countplot(x = 'class', hue='lug_boot', data = df, ax=axes[1,2])
axes[0,2].set_title('Persons')
sns.countplot(x = 'class', hue='persons', data = df, ax=axes[0,2])

plt.tight_layout()
plt.show()



