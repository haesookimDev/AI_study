# Chapter4  
Building Good Training Datasets – Data Preprocessing  
  
이 장에서 다룰 주제는 다음과 같습니다.  
• 데이터 집합에서 결측값 제거 및 귀속  
• 기계 학습 알고리즘을 위한 범주형 데이터 구체화  
• 모델 구성에 대한 관련 기능 선택
  
## Dealing with missing data  
  
이 섹션에서는 데이터 세트에서 항목을 제거하거나 다른 교육 예제 및 기능에서 누락된 값을 귀속하여 결측값을 처리하기 위한 몇 가지 실용적인 기술을 통해 작업할 것이다.  
  
#### Identifying missing values in tabular data
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
#### Imputing missing values
#### Understanding the scikit-learn estimator API
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
