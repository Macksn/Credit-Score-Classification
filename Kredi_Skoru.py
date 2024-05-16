# Karar Ağacı Algoritması ile Kredi Skoru Sınıflandırması

#Kullanacağımız kütüphanelerimizi yüklüyoruz
from sklearn import tree
import pandas as pd
import os
from sklearn import preprocessing

print(os.getcwd())

## 1.Veri Toplama

dataset=pd.read_csv("CreditScoreClassificationDataset.csv")

## 2.Veri Analizi

dataset

## 3.Sonuçların Görselleştirilmesi

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kfold=KFold(10,shuffle=True,random_state=True)

string_to_int= preprocessing.LabelEncoder()
dataset.Gender=string_to_int.fit_transform(dataset.Gender)
dataset.MaritalStatus=string_to_int.fit_transform(dataset.MaritalStatus)
dataset.HomeOwnership=string_to_int.fit_transform(dataset.HomeOwnership)
dataset.CreditScore=string_to_int.fit_transform(dataset.CreditScore)
dataset.Education=string_to_int.fit_transform(dataset.Education)

X=dataset.drop("CreditScore",axis=1)
y=dataset.CreditScore
k=0
accuracy=[]
for train,test in kfold.split(dataset):
    print(k,len(train),len(test))
    k+=1
    trainSet=dataset.iloc[train]
    testSet=dataset.iloc[test]

    dt=tree.DecisionTreeClassifier(criterion="gini")
    dt.fit(trainSet.drop("CreditScore",axis=1),trainSet.CreditScore)

    pred=dt.predict(testSet.drop("CreditScore",axis=1))
    score=accuracy_score(testSet.CreditScore,pred)
    accuracy.append(score)
    print("Accuracy:"+str(score))


## 4.Sonuçların Yorumlanması 

print(sum(accuracy)/len(accuracy))
from matplotlib import pyplot as plt
tree.plot_tree(dt)
plt.show()

# Ortalama Doğruluğun Hesağlanması
avg_accuracy = sum(accuracy) / len(accuracy)
print(f"Average Accuracy: {avg_accuracy:.3f}")

# Karar ağacı çizimi
plt.figure(figsize=(10, 8))
tree.plot_tree(dt, filled=True)
plt.show()

