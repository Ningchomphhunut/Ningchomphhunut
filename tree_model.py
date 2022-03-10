import numpy
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pydot
import graphviz
import pickle

warnings.filterwarnings('ignore')

df = pd.read_csv('car_evaluation.csv') #นำเข้าชุดข้อมูล
print('Dimensions:',df.shape)

from sklearn.preprocessing import LabelEncoder #การเข้ารหัสจำลอง(ข้อมูลเป็นตัวเลข)
le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform(df[i])
    plt.show()


x=df[df.columns[:-1]]  #การสร้างตัวแปร(x)และ(y)
y=df['class']
x.head(2)

from sklearn.model_selection import train_test_split #แยกข้อมูลออกเป็นชุดฝึกและชุดทดสอบ
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

#เปรียบเทียบ model ต่าง ๆ
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
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
#ดังนั้นการจำแนกด้วยอัลกอริธึม RandomForestClassifier สามารถบรรลุความแม่นยำ 96%
with open('Tree_model.pkl','wb') as model:
    pickle.dump(RF,model)
