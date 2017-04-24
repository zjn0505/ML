# Human Learning Machine Learning -- HLML

> I learn and I forget, hopefully the machine won't.

#### perception 
&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/perceptron.py)

#### linear regression
- Cost Function

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492693028/render.png" alt="J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2"/>
</p>


where the hypothesis ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) is given by the linear model

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492692889/render.png" alt="h_\theta(x)=\theta^Tx=\theta_0+\theta_1x_1+...+\theta_nx_n"/>
</p>

The vectorized version is:

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492747002/render.png" alt="J(\theta)=\frac{1}{2m}(X\theta-\overrightarrow{y})^T(X\theta-\overrightarrow{y})"/>
</p>

- Gradient Descent

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492581297/render.png" alt="(\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}-y^{(i)})x_j^{(i)}))"/>
</p>


The vectorized version is:

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492747223/render.png" alt="\theta := \theta-\frac{\alpha}{m}X^T(X\theta-\overrightarrow{y})"/>
</p>

- Normal Equation

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492752074/render.png" alt="\theta=(X^TX)^{-1}X^T \overrightarrow{y}"/>
</p>

##### Comparison of Gradient Descent and Normal Equation

| Gradient Descent                    | Normal Equation           |
| :---:                               | :----:                    |
| \- Need to choose α                 | \+ No need to choose α    |
| \- Needs many iterations            | \+ Don't need to iterate  |
| \+ Works well even when n is large  | \- Need to compute ![(X^TX)^{-1}X^T](http://www.sciweavers.org/upload/Tex2Img_1492752105/render.png), slow if n is very large |
| Chosen when n > 10k                 | Chosen when n < 10k       |
| May need Feature Scaling            | No need of Feature Scaling|

##### Feature Scaling
returns a normalized version of X where the mean value of each feature is 0 and the standard deviation or range is 1. This is often a good preprocessing step to do when working with learning algorithms.

&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/gradient_descent.py)

Q1: Is there a proper choice of learning rate and iteration so that we can say "Yes, we'd like to ensure J(Θ) tend to converge at after 'X' iteration for most of input datasets"?

Q2: Any ideas on derivation of Normal Equation?

A2:
- [Derivation of the Normal Equation for linear regression](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)

- [Linear least squares (Wikipedia)](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics))

Note1: It seems to be a good habbit to declare necessary variables (vectors, matrices, return values) before using them. One will easily remember their dimensions in review.



#### Logistic Regression

- cost function

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492765827/render.png" alt="J(\theta)=\sum_{i=1}^{m}cost(h_\theta(x^{(i)}),y^{(i)})"/>
</p>

where the hypothesis ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) is given by the Sigmoid Function

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492767594/render.png" alt="h_\theta (x) = g ( \theta^T x )"/>
  <br>
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492778224/render.png" alt="z = \theta^Tx"/>
  <br>
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492767678/render.png" alt="g(z) = \dfrac{1}{1 + e^{-z}}"/>
</p>

and the cost() function is

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492767109/render.png" alt="cost(h_\theta(x),y) =\begin{cases}-logh_\theta(x) & y = 1\\-log(1-h_\theta(x)) & y = 0\end{cases}"/>
</p>

It means

- If ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) = y, then cost(![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png),y)=0 for both y=0 and y=1. (A good prediction)

- If y=0, then cost(![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) , y) → ∞ as ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) → 1. (A bad prediction)

- If y=1, then cost(![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) , y) → ∞ as ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) → 0. (A bad prediction)



Written in one line: y ∈ {0,1}

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492765286/render.png" alt="cost(h_\theta(x),y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))"/>
</p>

The entire cost function is:
<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492766145/render.png" alt="J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]"/>
</p>


The vectorized version is:
<p align="center">
    h=g(XΘ)
    <br>
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492766410/render.png" alt="J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right)"/>
</p>


- Gradient Descent

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492766977/render.png" alt="\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}-y^{(i)})x_j^{(i)})"/>
</p>


The vectorized version is:

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492767012/render.png" alt="\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})"/>
</p>

##### Regularization

To avoid the hypothesis function overfitting to training data. Penalize the cost function to make it more general.

[Regularized Linear Regression -- Coursera](https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression)

[Regularized Logisitic Regression -- Coursera](https://www.coursera.org/learn/machine-learning/supplement/v51eg/regularized-logistic-regression)


<!--- LaTeX generated in http://www.sciweavers.org/free-online-latex-equation-editor -->
