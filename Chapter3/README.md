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
> $$logit(p) = w_1x_1 + w_2x_2 + \cdots + w_mx_m + b = w^Tx + b$$
>   
> ![image](https://user-images.githubusercontent.com/63633387/190896834-31833ce8-95bb-4ef9-8844-3b435711b922.png)  
>   
>   
#### Learning the model weights via the logistic loss function  
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
 
