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
로지스틱 회귀에서 사용되는 손실 함수이다. 
![image](https://user-images.githubusercontent.com/63633387/190898058-e7ce0d9d-defe-495f-9f7f-cc0afd6c79d3.png)
$𝜎(𝑧)$는 변수 $x$를 모델이 $y$로 예측할 확률이다.   
로그를 취하는 이유는 산술연산의 결과가 취급할 수 있는 수의 범위 보다 작아지는 상태인 산술 언더플로의 가능성을 줄이기 위함이다.
#### Converting an Adaline implementation into an algorithm for logistic regression  
#### Training a logistic regression model with scikit-learn  
#### Tackling overfitting via regularization  
## Maximum margin classification with support vector machines  
#### Maximum margin intuition  
#### Dealing with a nonlinearly separable case using slack variables  
#### Alternative implementations in scikit-learn  
## Solving nonlinear problems using a kernel SVM  
#### Kernel methods for linearly inseparable data  
#### Using the kernel trick to find separating hyperplanes in a high-dimensional space  
## Decision tree learning  
#### Maximizing IG – getting the most bang for your buck  
#### Building a decision tree  
#### Combining multiple decision trees via random forests  
## K-nearest neighbors – a lazy learning algorithm  
## Summary 
 
