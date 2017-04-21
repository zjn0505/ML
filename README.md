# Human Learning Machine Learning -- HLML
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


&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/gradient_descent.py)

Q1: Is there a proper choice of learning rate and iteration so that we can say "Yes, we'd like to ensure J(Θ) tend to converge at after 'X' iteration for most of input datasets"?

Q2: Any ideas on derivation of Normal Equation?

A2:
- [Derivation of the Normal Equation for linear regression](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)

- [Linear least squares (Wikipedia)](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics))

Note1: It seems to be a good habbit to declare necessary variables (vectors, matrices, return values) before using them. One will easily remember their dimensions in review.

<!--- LaTeX generated in http://www.sciweavers.org/free-online-latex-equation-editor -->
