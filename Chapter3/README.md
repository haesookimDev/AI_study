# Chapter 3
Chapter 3: A Tour of Machine Learning Classifiers Using Scikit-Learn


## Choosing a classification algorithm  
1. 특징 선택 및 레이블링된 학습 데이터 수집  
2. 성능 메트릭 선택  
3. 학습 알고리즘 선택 및 모델 교육  
4. 모델의 성능 평가  
5. 알고리즘의 설정 변경 및 모델 튜닝  
## First steps with scikit-learn – training a perceptron  
## Modeling class probabilities via logistic regression  
#### Logistic regression and conditional probabilities  
>   
> ![image](https://user-images.githubusercontent.com/63633387/190896820-ccf2b591-c8b5-48c7-bc82-176092a4d640.png)
>    
> 로짓 함수는 0 ~ 1 범위의 입력 값을 가져와서 전체 실수 범위에 걸쳐 값으로 변환
> $$logit(p) = log\frac{p}{1-p}$$
>   
> 아래는 로짓 함수와 선형 입력간의 관계가 있다고 가정한 수식
> $$logit(p) = w_1x_1 + w_2x_2 + \cdots + w_mx_m + b = w^Tx + b$$
>   
> ![image](https://user-images.githubusercontent.com/63633387/190896834-31833ce8-95bb-4ef9-8844-3b435711b922.png)  
>  
>  $$𝜎(𝑧)=\frac{1}{1+e^-z}$$
>  
>  $$z = w^Tx + b$$
>   
#### Learning the model weights via the logistic loss function  
![image](https://user-images.githubusercontent.com/63633387/190898034-922d66c4-d11e-44a8-ab80-b2f3ae97ac8b.png)  
로짓함수를 베르누이 분포를 통해 최대 가능도 추정법을 계산한 로지스틱 회귀에서 사용되는 손실 함수이다. 
![image](https://user-images.githubusercontent.com/63633387/190898058-e7ce0d9d-defe-495f-9f7f-cc0afd6c79d3.png)  
$𝜎(𝑧)$는 변수 $x$를 모델이 $y$로 예측할 확률이다.   
로그를 취하는 이유는 산술연산의 결과가 취급할 수 있는 수의 범위 보다 작아지는 상태인 산술 언더플로의 가능성을 줄이기 위함이다.
#### Converting an Adaline implementation into an algorithm for logistic regression  
#### Training a logistic regression model with scikit-learn  
#### Tackling overfitting via regularization  
![image](https://user-images.githubusercontent.com/63633387/190898417-16158175-2f38-4af6-b3c7-cf48612ddfa5.png)  
과대적합은 학습 데이터에서는 높은 정확도를 보이지만 테스트(실제)데이터에서 일반화되지 않아 성능이 떨어지는 것을 말한다.  
해당문제는 학습 데이터를 학습 할 만큼 모델의 파라미터가 충분하지 않거나 모델의 데이터의 패턴을 학습 할만큼 데이터의 질이 낮다는 것을 의미한다.  
## Maximum margin classification with support vector machines  
![image](https://user-images.githubusercontent.com/63633387/190898627-a644546d-54b3-4cb6-bd91-d7a1b0fd0563.png)  
서포트 벡터 머신은 결정 경계와 서포트 벡터 사이의 마진을 최대화하는 최적의 결정 경계를 찾는 분류모델이다. 
#### Maximum margin intuition  
#### Dealing with a nonlinearly separable case using slack variables  
![image](https://user-images.githubusercontent.com/63633387/190898826-7d2af00c-56ee-437a-a540-10ae916c1893.png)  
C 파라미터를 통해 오분류에 대한 패널티를 제어한다. C파라미터가 크면 오분류를 엄격하게 관리하지만 과대적합의 위험이 있을 수 있다.

#### Alternative implementations in scikit-learn  
## Solving nonlinear problems using a kernel SVM  
서포트 벡터 머신은 비선형 데이터에서도 분류를 가능하게한다.
#### Kernel methods for linearly inseparable data  
#### Using the kernel trick to find separating hyperplanes in a high-dimensional space  
서포트 벡터머신은 비선형데이터를 고차원 특징공간으로 투영하여 나타난 데이터 차원에서 선형 평면을 만들어 분류한다.  
![image](https://user-images.githubusercontent.com/63633387/190899073-398374da-4e48-4b56-8b2f-dfbc2c058cdb.png)  
![image](https://user-images.githubusercontent.com/63633387/190899087-d4120a30-d0fb-4fce-8916-f81d795a5d82.png)  
  
  위처럼 저차원의 데이터를 고차원으로 투영하는 것은 어려운 일이다.   
  따라서 커널 트릭이라는 것을 이용한다.  
  ![image](https://user-images.githubusercontent.com/63633387/190899396-37fdd0f2-2e36-4dff-9813-077e53c269bc.png)  
  주로 사용하는 커널은 가우시안 커널(한쌍의 특징사이의 거리를 측정하는 유사도 함수)이라는 것을 사용한다.   
  ![image](https://user-images.githubusercontent.com/63633387/190899520-103c7e19-6dd2-4c2a-bd99-c41c1cf952d6.png)  
  지수항을 통해 거리값의 범위는 0(유사도 낮음)과 1(유사도 높음)사이가 된다.

## Decision tree learning  
  ![image](https://user-images.githubusercontent.com/63633387/190899836-e866b4a6-d43b-4a9b-b1c7-b4b531fe82e3.png)
  위의 그림과 같이 나의 가지처럼 뻗어나가며 결과를 결정하는 학습 모델이다.
  도출된 결과를 해석 할 수 있기 때문에 예측과정에 대한 정보를 얻을 수 있다.
  노드가 많아지고 깊이가 깊어지면 학습과정에서는 높은 정확도를 얻을 수 있지만 
  실제 데이터를 이용할 때 일반화가 되지않는 과대적합으로 이어질 수 있다.

#### Maximizing IG – getting the most bang for your buck  
  ![image](https://user-images.githubusercontent.com/63633387/190900069-32050695-bec1-4e73-8011-0e398dbe33ae.png)  
  위의 함수는 결정트리에서 최적화에 사용하는 함수이다.  
  $D$는 데이터 $N$은 노드에서 학습되는 $x$의 수 이며 $I$는 impurity 측정법이며 $p$는 부모, $j$는 자식이다.  
  정보이득을 최대화 하는 방법은 자식노드에서의 impurity가 낮을 수록 높다
  ![image](https://user-images.githubusercontent.com/63633387/190900159-6442941d-59a8-4b52-95a6-ba01f70dcd31.png)  
  위는 대부분의 라이브러리에서 구현되는 이진 결정트리의 정보이득 계산식이다.
  
  이진 결정트리에서 사용되는 분류 기준은 Gini impurity $(I_G)$, entropy $(I_H)$ classification error $(I_E)$이다.
  
  ![image](https://user-images.githubusercontent.com/63633387/190900352-c73184a8-e5e9-4153-a330-254a309dda85.png)  
  ![image](https://user-images.githubusercontent.com/63633387/190900380-2a0b5b20-1ef2-47d8-ad4d-022b8c568d44.png)  
  ![image](https://user-images.githubusercontent.com/63633387/190900391-7a2d3a34-00af-40dc-a578-058f0a1098cb.png)  
  
  $p(i|t)$는 특정 노드, $t$에 대한 클래스 $i$에 속하는 $x$들의 비율이다.
  
#### Building a decision tree  
#### Combining multiple decision trees via random forests  
  여러 결정트리의 결과를 합쳐서 하나의 결론을 내는 것을 랜덤 포레스트라 부르며 모델의 앙상블이라고 할 수 있다.  
  방법은 아래와 같다. 
  1. 크기가 n인 무작위 샘플을 추출한다(학습 데이터 세트에서 무작위로 n개의 $x$를 선택한다).  
  2. 샘플에서 의사결정 트리를 학습한다.
    각 노드:  
    a. 교체하지 않고 무작위로 선택한 특징
    b. 목적 함수에 따라 최적의 분할을 제공하는 기능, 예를 들어 정보 이득을 최대화하는 기능을 사용하여 노드를 분할  
  3. 1 ~ 2 k회 단계를 반복한다.  
  4. 각 트리별 예측을 집계하여 다수결로 클래스 레이블을 할당한다.

## K-nearest neighbors – a lazy learning algorithm  
  단순함 때문이 아니라 훈련 데이터에서 차별적 기능을 학습하지 않고 훈련 데이터 세트를 대신 외우기 때문에 'a lazy learning algorithm'으로 불린다.  
  
  KNN 알고리즘 자체는 매우 간단하며 다음과 같은 단계로 요약할 수 있다.  
  1. k의 수와 거리 메트릭을 선택합니다.  
  2. 분류하고자 하는 데이터 레코드의 가장 가까운 이웃 찾기  
  3. 다수결로 클래스를 지정한다.  
  
  ![image](https://user-images.githubusercontent.com/63633387/190900812-f98d7448-36bb-4df6-9451-9722303d3950.png)  
  
  KNN 알고리즘은 분류하고자 하는 지점에 가장 가까운(가장 유사한) 훈련 데이터 세트에서 k개의 예를 찾는다.  
  데이터 포인트의 클래스 라벨은 k개의 가장 가까운 이웃들 사이의 다수결에 의해 결정

## Summary 
 
