## Introduction to Machine Learning


---

### How to talk about data

 - Tabular data
  - columns, rows

 - Statistical terms
  - independent, dependent

 - Computer science
  - input, output; instance, prediction

 - Model and algorithms
  - model = algorithm(data)

---

## Mapping input to output

Machine learning algorithms are described as learning a target function (f) that best maps input variables (X) to an output variable (Y ). Y =f(X) + e

Machine learning algorithms are techniques for estimating the target function (f) to predict the output variable (Y ) given input variables (X).

---

## Parametric and Nonparametric

 - That parametric machine learning algorithms simplify the mapping to a known functional form.

 - That nonparametric algorithms can learn any mapping from inputs to outputs.

 - That all algorithms can be organized into parametric or nonparametric groups.

---

## Parametrics
 - A parametric algorithm involve two steps:
  1. Select a form for the function.
  2. Learn the coefficients for the function from the training data.

 - Some more examples of parametric machine learning algorithms include:   
  - Logistic Regression   
  - Linear Discriminant Analysis  
  - Perceptron

 - Benefits of Parametric Machine Learning Algorithms:   
  - Simpler   
  - Speed
  - Less Data

 - Limitations of Parametric Machine Learning Algorithms:  
  - Constrained  
  - Limited Complexity   
  - Poor Fit

---

## NonParametrics

 - Some more examples of popular nonparametric machine learning algorithms are:   
  - Decision Trees like CART and C4.5  
  - Naive Bayes   
  - Support Vector Machines   
  - Neural Networks

 - Benefits of Nonparametric Machine Learning Algorithms:   
  - Flexibility   
  - Power   
  - Performance

 - Limitations of Nonparametric Machine Learning Algorithms:   
  - More data   
  - Slower
  - Overfitting

---

## Supervised, Unsupervised, Semi-supervised



---

## Supervised

 - The goal is to approximate the mapping function so well that when you have new input data (X) that you can predict the output variables (Y ) for that data.

 - Grouped into:
  - Classification: A classification problem is when the output variable is a category, such as red or blue or disease and no disease.   
  - Regression: A regression problem is when the output variable is a real value, such as dollars or weight.

 - Some popular examples of supervised machine learning algorithms are:   
  - Linear regression for regression problems.   
  - Random forest for classification and regression problems.   
  - Support vector machines for classification problems.

---

## Unsupervised

 - The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.

 - Grouped into:
  - Clustering: A clustering problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.   
  - Association: An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy A also tend to buy B.

 - Some popular examples of unsupervised learning algorithms are:
  - k-means for clustering problems.
  - Apriori algorithm for association rule learning problems.

---

## Semi-Supervised

 - Problems where you have a large amount of input data (X) and only some of the data is labeled (Y ) are called semi-supervised learning problems.

 - You can use unsupervised learning techniques to discover and learn the structure in the input variables. You can also use unsupervised learning techniques to make best guess predictions for the unlabeled data, feed that data back into the supervised learning algorithm as training data and use the model to make predictions on new unseen data.

 - A good example is a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

 - Many real world machine learning problems fall into this area.
 - This is because it can be expensive or time consuming to label data as it may require access to domain experts. Whereas unlabeled data is cheap and easy to collect and store.

---

## Learning error

 - Bias refers to the simplifying assumptions made by the algorithm to make the problem easier to solve.

 - Variance refers to the sensitivity of a model to changes to the training data.

 - Irreducible Error cannot be reduced regardless of what algorithm is used.

---

## Bias

 - Bias are the simplifying assumptions made by a model to make the target function easier to learn.

 - Low Bias: Suggests less assumptions about the form of the target function.
 - High-Bias: Suggests more assumptions about the form of the target function.

 - Examples of low-bias machine learning algorithms include:
  - Decision Trees,
  - k-Nearest Neighbors
  - Support Vector Machines.

  - Examples of high-bias machine learning algorithms include:
   - Linear Regression
   - Linear Discriminant Analysis
   - Logistic Regression.

 - Generally parametric algorithms have a high bias making them fast to learn and easier to understand but generally less flexible.
 - In turn they have lower predictive performance on complex problems that fail to meet the simplifying assumptions of the algorithms bias.

---

## Variance

 - Variance is the amount that the estimate of the target function will change if different training data was used.

 - Low Variance: Suggests small changes to the estimate of the target function with changes to the training dataset.   
 - High Variance: Suggests large changes to the estimate of the target function with changes to the training dataset.

 - Generally nonparametric machine learning algorithms that have a lot of flexibility have a high variance.

 - For example decision trees have a high variance, that is even higher if the trees are not pruned before use.

 - Examples of low-variance machine learning algorithms include:
  - Linear Regression
  - Linear Discriminant Analysis
  - Logistic Regression.

 - Examples of high-variance machine learning algorithms include:
  - Decision Trees
  - k-Nearest Neighbors
  - Support Vector Machines.

---

## Bias & Variance relationship

 - The goal of any supervised machine learning algorithm is to achieve low bias and low variance.
 - Parametric or linear machine learning algorithms often have a high bias but a low variance.
 - Nonparametric or nonlinear machine learning algorithms often have a low bias but a high variance.

 - The k-nearest neighbors algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbors that contribute to the prediction and in turn increases the bias of the model.  

 - The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

  - There is no escaping the relationship between bias and variance in machine learning.  Increasing the bias will decrease the variance.  Increasing the variance will decrease the bias.

 - In reality we cannot calculate the real bias and variance error terms because we do not know the actual underlying target function.

 - Nevertheless, as a framework, bias and variance provide the tools to understand the behavior of machine learning algorithms in the pursuit of predictive performance.

---

## Overfitting & Underfitting

 - The cause of poor performance in machine learning is either overfitting or underfitting the data.

 - That overfitting refers to learning the training data too well at the expense of not generalizing well to new data.   

 - That underfitting refers to failing to learn the problem from the training data sufficiently.   

 - That overfitting is the most common problem in practice and can be addressed by using resampling methods and a held-back verification dataset.

 - Overfitting refers to a model that models the training data too well. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize.

 - Overfitting is more likely with nonparametric and nonlinear models that have more flexibility when learning a target function.

 - Underfitting refers to a model that can neither model the training data nor generalize to new data.
 - The remedy is to move on and try alternate machine learning algorithms.

---

## Inductive, Deductive, Generalization, Statistical fit

 - In machine learning we describe the learning of the target function from training data as inductive learning.  

 - Induction refers to learning general concepts from specific examples which is exactly the problem that supervised machine learning problems aim to solve.

 - Deduction that is the other way around and seeks to learn specific concepts from general rules.

 - Generalization refers to how well the concepts learned by a machine learning model apply to specific examples not seen by the model when it was learning. The goal of a good machine learning model is to generalize well from the training data to any data from the problem domain.

 - Statistical Fit refers to how well you approximate a target function. This is good terminology to use in machine learning, because supervised machine learning algorithms seek to approximate the unknown underlying mapping function for the output variables given the input variables.


---

## Sweet spot

 - The sweet spot is the point just before the error on the test dataset starts to increase where the model has good skill on both the training dataset and the unseen test dataset.

 - There are two important techniques that you can use when evaluating machine learning algorithms to limit overfitting:
  1. Use a resampling technique to estimate model accuracy.
  2. Hold back a validation dataset.

 - Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

---

## MAE and RMSE

[MAE and RMSE — Which Metric is Better?](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d)
 - RMSE has the benefit of penalizing large errors more so can be more appropriate in some cases, for example, if being off by 10 is more than twice as bad as being off by 5.

 - But if being off by 10 is just twice as bad as being off by 5, then MAE is more appropriate
