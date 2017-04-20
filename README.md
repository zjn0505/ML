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

- Gradient Descent for Multiple Variables

<p align="center">
  <img src="http://www.sciweavers.org/upload/Tex2Img_1492581297/render.png"/>
  <!--- (\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}-y^{(i)})x_j^{(i)})) -->
</p>


&nbsp; [python](https://github.com/zjn0505/ML/blob/master/Python/gradient_descent.py)

Q1: Is there a proper choice of learning rate and iteration so that we can say "Yes, we'd like to ensure J(Θ) tend to converge at after 'X' iteration for most of input datasets"?

Note1: It seems to be a good habbit to declare necessary variables (vectors, matrices, return values) before using them. One will easily remember their dimensions in review.

<!--- LaTeX generated in http://www.sciweavers.org/free-online-latex-equation-editor -->
