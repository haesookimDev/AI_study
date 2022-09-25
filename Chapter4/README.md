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
  
  가장 일반적인 보간 기술 중 하나는 `mean imputation` 이며, 여기서는 결측값을 전체 형상 열의 평균값으로 간단히 교체한다.
  
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
```python

```
## Handling categorical data
#### Categorical data encoding with pandas
#### Mapping ordinal features
#### Encoding class labels
#### Performing one-hot encoding on nominal features
> Optional: encoding ordinal features
## Partitioning a dataset into separate training and test datasets
## Bringing features onto the same scale
## Selecting meaningful features
#### L1 and L2 regularization as penalties against model complexity
#### A geometric interpretation of L2 regularization
#### Sparse solutions with L1 regularization
#### Sequential feature selection algorithms
## Assessing feature importance with random forests
## Summary
