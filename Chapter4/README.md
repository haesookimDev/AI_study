# Chapter4  
Building Good Training Datasets – Data Preprocessing  
  
이 장에서 다룰 주제는 다음과 같습니다.  
• 데이터 집합에서 결측값 제거 및 귀속  
• 기계 학습 알고리즘을 위한 범주형 데이터 구체화  
• 모델 구성에 대한 관련 기능 선택
  
## Dealing with missing data  
  
이 섹션에서는 데이터 세트에서 항목을 제거하거나 다른 교육 예제 및 기능에서 누락된 값을 귀속하여 결측값을 처리하기 위한 몇 가지 실용적인 기술을 통해 작업할 것이다.  
  
#### Identifying missing values in tabular data
  
  .csv 확장자를 갖는 Dataframe을 하나 만든다.
  
```python
import pandas as pd
from io import StringIO
csv_data = \
... '''A,B,C,D
... 1.0,2.0,3.0,4.0
... 5.0,6.0,,8.0
... 10.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
df
```
```python
df.isnull().sum()
A   0
B   0
C   1
D   1
dtype: int64
```
#### Eliminating training examples or features with missing values
  
  결측 데이터를 처리하는 가장 쉬운 방법 중 하나는 데이터 세트에서 해당 기능(열) 또는 교육 예제(행)를 완전히 제거하는 것이다.  
  결측값이 있는 행은 dropna 방법을 통해 쉽게 삭제할 수 있다.
  
```python
df.dropna(axis=0)
A B C D
0 1.0 2.0 3.0 4.0
```
```python
df.dropna(axis=1)
A B
0 1.0 2.0
1 5.0 6.0
2 10.0 11.0
```
```python
# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN)
df.dropna(how='all')
A B C D
0 1.0 2.0 3.0 4.0
1 5.0 6.0 NaN 8.0
2 10.0 11.0 12.0 NaN
```
```python
# drop rows that have fewer than 4 real values
df.dropna(thresh=4)
A B C D
0 1.0 2.0 3.0 4.0
# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])
A B C D
0 1.0 2.0 3.0 4.0
2 10.0 11.0 12.0 NaN
```
  
  너무 많은 피쳐 열을 제거하면 분류기가 클래스를 구별하는 데 필요한 귀중한 정보가 손실될 위험이 있다.  
  다음 장에서는 결측값을 처리하기 위해 가장 일반적으로 사용되는 대안 중 하나인 보간 기법을 살펴보겠다.
  
#### Imputing missing values
  
  가장 일반적인 보간 기술 중 하나는 `mean imputation` 이며, 여기서는 결측값을 전체 형상 열의 평균값으로 간단히 대치한다.
  
```python
from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data
array([[ 1., 2., 3., 4.],
[ 5., 6., 7.5, 8.],
[ 10., 11., 12., 6.]])
```
```python
df.fillna(df.mean())
```
#### Understanding the scikit-learn estimator API
  
  이전 섹션에서는 `skickit-learn`의 `SimpleImputer` 클래스를 사용하여 데이터 세트의 결측값을 대치하였다.  
  `estimators`의 두 가지 `essential methods`는 `fit` 와 `transform`이다.  
  `fit`은 훈련 데이터에서 매개 변수를 학습하는 데 사용되며 `transform`은 이러한 매개 변수를 사용하여 데이터를 변환합니다.  
  `transform`할 데이터 배열은 모형을 `fit`시키는 데 사용된 데이터 배열과 동일한 수의 피쳐를 가져야 합니다.  
  ![image](https://user-images.githubusercontent.com/63633387/192139744-8aef565e-f4be-4432-80f8-8f780af14b3c.png)
  
  ![image](https://user-images.githubusercontent.com/63633387/192139765-5f387b41-b428-4afe-ae9e-0a2631cf1b72.png)  
  
  
## Handling categorical data
실제 데이터 세트에서 하나 이상의 `categorical feature` 열을 포함하는 것은 드문 일이 아니다

`Ordinal features`는 정렬 또는 순서가 있는 `categorical values`
이와 반대되는 것을 `Nominal features`라고 한다.
#### Categorical data encoding with pandas
```python
import pandas as pd
df = pd.DataFrame([
... ['green', 'M', 10.1, 'class2'],
... ['red', 'L', 13.5, 'class1'],
... ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
df
  color   size  price   classlabel
0 green   M     10.1    class2
1 red     L     13.5    class1
2 blue    XL    15.3    class2
```
#### Mapping ordinal features
```python
size_mapping = {'XL': 3,
...             'L': 2,
...             'M': 1}
```
```python
df['size'] = df['size'].map(size_mapping)
df
  color   size  price   classlabel
0 green   1     10.1    class2
1 red     2     13.5    class1
2 blue    3     15.3    class2
```
```python
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)
0 M
1 L
2 XL
Name: size, dtype: object
```
#### Encoding class labels
```python
import numpy as np
class_mapping = {label: idx for idx, label in
... enumerate(np.unique(df['classlabel']))}
class_mapping
{'class1': 0, 'class2': 1}
```
```python
df['classlabel'] = df['classlabel'].map(class_mapping)
df
  color   size  price   classlabel
0 green   1     10.1    1
1 red     2     13.5    0
2 blue    3     15.3    1
```
```python
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
  color size price classlabel
0 green 1    10.1   class2
1 red   2    13.5   class1
2 blue  3    15.3   class2
```
```python
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y
array([1, 0, 1])
```
```python
class_le.inverse_transform(y)
array(['class2', 'class1', 'class2'], dtype=object)
```
#### Performing one-hot encoding on nominal features
```python
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X
array([[1, 1, 10.1],
      [2, 2, 13.5],
      [0, 3, 15.3]], dtype=object)
```
```python
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
array([[0., 1., 0.],
      [0., 0., 1.],
      [1., 0., 0.]])
```
```python
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
...     ('onehot', OneHotEncoder(), [0]),
...     ('nothing', 'passthrough', [1, 2])
... ])
c_transf.fit_transform(X).astype(float)
array([[0.0, 1.0, 0.0, 1, 10.1],
      [0.0, 0.0, 1.0, 2, 13.5],
      [1.0, 0.0, 0.0, 3, 15.3]])
```
```python
pd.get_dummies(df[['price', 'color', 'size']])
  price size color_blue color_green color_red
0 10.1  1     0           1           0
1 13.5  2     0           0           1
2 15.3  3     1           0           0
```
```python
pd.get_dummies(df[['price', 'color', 'size']],
... drop_first=True)
   price size color_green color_red
0 10.1    1     1           0
1 13.5    2     0           1
2 15.3    3     0           0
```
```python
olor_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
...     ('onehot', color_ohe, [0]),
...     ('nothing', 'passthrough', [1, 2])
... ])
c_transf.fit_transform(X).astype(float)
array([[ 1. , 0. , 1. , 10.1],
        [ 0. , 1. , 2. , 13.5],
        [ 0. , 0. , 3. , 15.3]])
```
## Partitioning a dataset into separate training and test datasets
```python
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
...                   'ml/machine-learning-databases/'
...                   'wine/wine.data', header=None)
```
```python
df_wine.columns = ['Class label', 'Alcohol',
...                'Malic acid', 'Ash',
...                'Alcalinity of ash', 'Magnesium',
...                'Total phenols', 'Flavanoids',
...                'Nonflavanoid phenols',
...                'Proanthocyanins',
...                'Color intensity', 'Hue',
...                'OD280/OD315 of diluted wines',
...                'Proline']
print('Class labels', np.unique(df_wine['Class label']))
Class labels [1 2 3]
df_wine.head()
```  
![image](https://user-images.githubusercontent.com/63633387/192140492-b17160f9-b4b3-4c34-b5a1-a09f61bfffca.png)
  
```python
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
...    train_test_split(X, y,
...                     test_size=0.3,
...                     random_state=0,
...                     stratify=y)
```
## Bringing features onto the same scale
  
  ![image](https://user-images.githubusercontent.com/63633387/192140535-ab0bdf63-cf1a-4034-934d-6ca045d94073.png)
  
```python
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
```
  
  ![image](https://user-images.githubusercontent.com/63633387/192140543-a4d608b1-e922-48ef-ab75-3bdc24ddcf75.png)  
  ![image](https://user-images.githubusercontent.com/63633387/192140556-2f20b2b9-d63d-49b8-9e02-a4864112e556.png)  
   
```python
ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean()) / ex.std())
standardized: [-1.46385011 -0.87831007 -0.29277002 0.29277002
0.87831007 1.46385011]
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))
normalized: [ 0. 0.2 0.4 0.6 0.8 1. ]
```
```python
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```
## Selecting meaningful features
#### L1 and L2 regularization as penalties against model complexity
#### A geometric interpretation of L2 regularization
#### Sparse solutions with L1 regularization
  
  ![image](https://user-images.githubusercontent.com/63633387/192140612-da801d22-3130-4327-9e01-589313f11a97.png)
  
```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1',
...                 solver='liblinear',
...                 multi_class='ovr')
```
```python
lr = LogisticRegression(penalty='l1',
...                     C=1.0,
...                     solver='liblinear',
...                     multi_class='ovr')
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regularization effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
Training accuracy: 1.0
print('Test accuracy:', lr.score(X_test_std, y_test))
Test accuracy: 1.0
```
```python
lr.intercept_
array([-1.26317363, -1.21537306, -2.37111954])
```
```python
lr.coef_
array([[ 1.24647953, 0.18050894, 0.74540443, -1.16301108,
         0.        , 0.        , 1.16243821, 0.         ,
         0.        , 0.        , 0.        , 0.55620267,
         2.50890638],
       [-1.53919461, -0.38562247, -0.99565934, 0.36390047,
        -0.05892612, 0.         , 0.66710883,  0.        ,
        0.         , -1.9318798 , 1.23775092,  0.        ,
        -2.23280039], 
       [ 0.13557571, 0.16848763, 0.35710712,   0.        ,
        0.         , 0.        , -2.43804744,  0.        ,
        0.         , 1.56388787, -0.81881015, -0.49217022,
        0. ]])
```
```python
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
...       'magenta', 'yellow', 'black',
...       'pink', 'lightgreen', 'lightblue',
...       'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
...   lr = LogisticRegression(penalty='l1', C=10.**c,
...                           solver='liblinear',
...                           multi_class='ovr', random_state=0)
...   lr.fit(X_train_std, y_train)
...   weights.append(lr.coef_[1])
...   params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
...   plt.plot(params, weights[:, column],
...            label=df_wine.columns[column + 1],
...            color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
...       bbox_to_anchor=(1.38, 1.03),
...       ncol=1, fancybox=True)
plt.show()
```
  
  ![image](https://user-images.githubusercontent.com/63633387/192140792-6cb8149e-e3a5-41bb-999d-b395bce8d90d.png)
  
#### Sequential feature selection algorithms
  
  모델의 복잡성을 줄이고 과적합을 방지하는 방법은 `feature selection`을 통한 `dimensionality reduction`이며, 
  이는 특히 비정규화된 모델에 유용 
  `dimensionality reduction` 기술에는 `feature selection`과 `feature extraction`이라는 두 가지 주요 범주가 있다. 
  
  `feature selection`은 원래 피쳐의 하위 집합을 선택하는 반면, 
  `feature extraction`에서는 피쳐 집합에서 정보를 도출하여 새 `feature subspace`을 구성한다.
  
  1. k = d로 알고리즘을 초기화. 여기서 d는 `전체 feature space` $X_d$의 `dimensionality`
  2. 기준을 최대화하는 특징 $x^–$을 구한다: $x^– = argmaxJ(X_k – x)$, $x \in X_k$
  3. `feature set`에서 `feature` x^–을 제거한다: $X_k–1 = X_k – x^–; k = k – 1.$
  4. k가 원하는 피쳐의 수와 같으면 종료하고, 그렇지 않으면 2단계로 이동합니다.
  
```python
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
        
    def transform(self, X):
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
```
```python
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
```
```python
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
```
  
  ![image](https://user-images.githubusercontent.com/63633387/192140967-18cf4346-4181-4b39-916a-c2947ef693ad.png)
  
```python
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])
Index(['Alcohol', 'Malic acid', 'OD280/OD315 of diluted wines'],
dtype='object')
```
```python
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
Training accuracy: 0.967741935484
print('Test accuracy:', knn.score(X_test_std, y_test))
Test accuracy: 0.962962962963
```
```python
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:',
...     knn.score(X_train_std[:, k3], y_train))
Training accuracy: 0.951612903226
print('Test accuracy:',
...     knn.score(X_test_std[:, k3], y_test))
Test accuracy: 0.925925925926
```
## Assessing feature importance with random forests
```python
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,
...                             random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
... print("%2d) %-*s %f" % (f + 1, 30,
...                         feat_labels[indices[f]],
...                          importances[indices[f]]))
plt.title('Feature importance')
plt.bar(range(X_train.shape[1]),
...     importances[indices],
...     align='center')
plt.xticks(range(X_train.shape[1]),
...        feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
1) Proline 0.185453
2) Flavanoids 0.174751
3) Color intensity 0.143920
4) OD280/OD315 of diluted wines 0.136162
5) Alcohol 0.118529
6) Hue 0.058739
7) Total phenols 0.050872
8) Magnesium 0.031357
9) Malic acid 0.025648
10) Proanthocyanins 0.025570
11) Alcalinity of ash 0.022366
12) Nonflavanoid phenols 0.013354
13) Ash 0.013279
```
  와인 데이터 세트의 다양한 기능을 상대적 중요도에 따라 순위를 매기는 그림 
  기능 중요도 값은 최대 1.0까지 합치도록 표준화
  ![image](https://user-images.githubusercontent.com/63633387/192140159-4d17b3ca-a214-4548-afa9-5c7e6be3760c.png)
  
```python
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold',
... 'criterion:', X_selected.shape[1])
Number of features that meet this threshold criterion: 5
for f in range(X_selected.shape[1]):
... print("%2d) %-*s %f" % (f + 1, 30,
... feat_labels[indices[f]],
... importances[indices[f]]))
1) Proline 0.185453
2) Flavanoids 0.174751
3) Color intensity 0.143920
4) OD280/OD315 of diluted wines 0.136162
5) Alcohol 0.118529
```
## Summary
