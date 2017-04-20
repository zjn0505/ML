# Human Learning Machine Learning -- HLML
#### perception 
&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/perceptron.py)

#### linear regression
- Cost Function

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492693028/render.png"/>
  <!-- J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2 -->
</p>


where the hypothesis ![h_\theta(x)](http://www.sciweavers.org/upload/Tex2Img_1492693075/render.png) is given by the linear model

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492692889/render.png"/>
  <!-- h_\theta(x)=\theta^Tx=\theta_0+\theta_1x_1+...+\theta_nx_n  -->
</p>

- Gradient Descent

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492581297/render.png"/>
  <!--- (\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}-y^{(i)})x_j^{(i)})) -->
</p>


- Normal Equation

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492694590/render.png"/>
  <!--- \theta=(X^TX)^{-1}X^T \overrightarrow{y}  -->
</p>

##### Comparison of Gradient Descent and Normal Equation

| Gradient Descent                    | Normal Equation           |
| :---:                               | :----:                    |
| \- Need to choose α                 | \+ No need to choose α    |
| \- Needs many iterations            | \+ Don't need to iterate  |
| \+ Works well even when n is large  | \- Need to compute ![(X^TX)^{-1}X^T](http://www.sciweavers.org/upload/Tex2Img_1492695138/render.png), slow if n is very large |
| Chosen when n > 10k                 | Chosen when n < 10k       |


&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/gradient_descent.py)

Q1: Is there a proper choice of learning rate and iteration so that we can say "Yes, we'd like to ensure J(Θ) tend to converge at after 'X' iteration for most of input datasets"?

Q2: Any ideas on derivation of Normal Equation?

A2:
- [Five Ways to Derive the Normal Equation](http://blog.xiangjiang.live/derivations-of-the-normal-equation/)

- [掰开揉碎推导Normal Equation](https://zhuanlan.zhihu.com/p/22757336)

Note1: It seems to be a good habbit to declare necessary variables (vectors, matrices, return values) before using them. One will easily remember their dimensions in review.

<!--- LaTeX generated in http://www.sciweavers.org/free-online-latex-equation-editor -->
