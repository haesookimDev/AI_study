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
> $z$는 $w_1x_1 + w_2x_2 + ... + w_mx_m$ 이다.  
>   
> 특정 입력값 $x_i$가 정해진 임계치 $𝜃$보다 크면 1 아니면 0으로 분류하며  
> 퍼셉트론 알고리즘에서 정의된 결정 함수 $𝜎(∙)$는 단위 계단 함수(아래 그림의 왼쪽형태의 함수를 단위계단 함수라고 함)의 변형이다. 
>   
> $$𝜎(z) = 
> \begin{cases} 
> 1\;if\;z\geq0\\ 
> 0\;otherwise 
> \end{cases}$$
>   
> ![image](https://user-images.githubusercontent.com/63633387/190891854-a2673e88-8862-49ff-8146-853c02806173.png)
>   
> 입력 $z = w^Tx + b$가 퍼셉트론의 결정 함수에 의해 이진 출력(0 또는 1)이 방법과 선형 결정 경계에 의해 분리될 수 있는 두 가지 클래스로 구별하는 방법
#### The perceptron learning rule  
## Implementing a perceptron learning algorithm in Python
#### An object-oriented perceptron API  
#### Training a perceptron model on the Iris dataset  
## Adaptive linear neurons and the convergence of learning 
#### Minimizing loss functions with gradient descent  
#### Implementing Adaline in Python  
#### Improving gradient descent through feature scaling  
#### Large-scale machine learning and stochastic gradient descent  
## Summary 
