# Chapter2
Chapter 2: Training Simple Machine Learning Algorithms for Classification


## Artificial neurons – a brief glimpse into the early history of machine learning 
> ![image](https://user-images.githubusercontent.com/63633387/190891701-d296950e-5e27-4109-83a8-b66aeb5fd0e6.png)
> 
> 생물학적 뉴런은 화학적, 전기적 신호의 처리와 전달에 관여하는 뇌의 상호 연결된 신경 세포
>  
> McCulloch 와 Pitts는 이러한 신경세포를 이진 출력을 갖는 단순한 논리 게이트라고 설명
> 여러 신호가 가지돌기에 도달한 다음 세포 본체에 통합되고, 축적된 신호가 특정 임계값을 초과하면 축삭에 의해 전달될 출력 신호가 생성  

#### The formal definition of an artificial neuron  
> 
> 인공 뉴런의 아이디어를 0과 1의 두 가지 클래스가 있는 이진 분류에 적용  
> 결정 함수 $𝜎(z)$는 특정 입력 값 $x$와 해당 가중치 벡터 $w$의 선형 조합  
> $z$는 $w_1x_1 + w_2x_2 + \cdots + w_mx_m$ 이다.  
>   
> 특정 입력값 $x_i$가 정해진 임계치 $𝜃$보다 크면 1 아니면 0으로 분류하며  
> 퍼셉트론 알고리즘에서 정의된 결정 함수 $𝜎(∙)$는 단위 계단 함수(아래 그림의 왼쪽형태의 함수를 단위계단 함수라고 함)의 변형이다. 
>   
> $$𝜎(z) = 
> \begin{cases} 
> 1\ if\ z\geq0\\ 
> 0\ otherwise 
> \end{cases}$$
>   
> ![image](https://user-images.githubusercontent.com/63633387/190891854-a2673e88-8862-49ff-8146-853c02806173.png)
>   
> 입력 $z = w^Tx + b$가 퍼셉트론의 결정 함수에 의해 이진 출력(0 또는 1)이 방법과 선형 결정 경계에 의해 분리될 수 있는 두 가지 클래스로 구별하는 방법
#### The perceptron learning rule  
>   
> Rosenblatt’s 고전적인 퍼셉트론 규칙은 매우 간단하며 퍼셉트론 알고리즘은 다음과 같은 단계와 같다.
>   
> 1. 가중치와 바이어스를 0 또는 작은 수의 난수로 초기화
> 2. 각 예시 $x^(i)$를 학습  
>   a. 출력값 $\widehat{y}^(i)$ 계산  
>   b. 가중치와 바이어스 단위 업데이트
>   
>   
> ![image](https://user-images.githubusercontent.com/63633387/190893585-87889d00-173d-4bf3-9108-f974ee6977b3.png)
>   
>   
> 두 클래스가 선형 결정 경계로 분리될 수 없는 경우,  
> 훈련 데이터 세트(에포크)에 대한 최대 허용 횟수 및/또는 허용되는 잘못 분류된 입력의 개수에 대한 임계값을 설정할 수 있다.  
> 그렇지 않으면 퍼셉트론은 가중치 업데이트를 멈추지 않을 것  
>   
>   
> ![image](https://user-images.githubusercontent.com/63633387/190893602-f7bcd2f4-e57b-4a8d-a5d1-952b2cf684ed.png)
>   
>   
## Implementing a perceptron learning algorithm in Python (해당 파이썬 코드는 [여기](https://github.com/ww232330/AI_study/blob/main/Chapter2/Chapter_2_Training_Simple_Machine_Learning_Algorithms_for_Classification.ipynb)에 있음)
#### An object-oriented perceptron API  
#### Training a perceptron model on the Iris dataset  
## Adaptive linear neurons and the convergence of learning 
>   
> ![image](https://user-images.githubusercontent.com/63633387/190893789-641655f9-d2bc-483f-8508-9ef6f84c16aa.png)
>   
#### Minimizing loss functions with gradient descent  
>   
> 결과와 실제 값의 mean squared error (MSE)를 통해 파라미터를 조정한다.
> $$\frac{1}{2n} \displaystyle\sum{(y^(i)-\widehat{y}^(i))^2}$$
>   
#### Implementing Adaline in Python  
#### Improving gradient descent through feature scaling  
#### Large-scale machine learning and stochastic gradient descent  
## Summary 
