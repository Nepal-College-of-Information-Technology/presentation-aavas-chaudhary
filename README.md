# presentation-aavas-chaudhary
presentation-aavas-chaudhary created by GitHub Classroom

name- Aavas Chaudhary
rollno- 211701
class- BESE (day)



# Logistic Regression

Introduction / History / Significance

Introduction:

Logistic regression is a statistical method used for binary classification problems. It predicts the probability that a given input belongs to a particular category, 
typically denoted as 0 or 1. Unlike linear regression, which predicts continuous values, logistic regression outputs probabilities constrained between 0 and 1 using
a logistic function (sigmoid function).

History:

The origins of logistic regression can be traced back to the work of Pierre François Verhulst in the 1840s, who introduced the logistic function to describe population
growth. In the context of classification, logistic regression was developed in the 1940s and 1950s, with notable contributions from David Cox in 1958, who applied the 
logistic function to binary data.

Significance:

Logistic regression is significant due to its simplicity, interpretability, and effectiveness in binary classification problems. It is widely used in various fields, 
including medical research, social sciences, marketing, and machine learning. Its probabilistic framework allows for a better understanding of the influence of 
different predictors on the outcome.

Architecture / Diagram / Network

Architecture:

Logistic regression can be represented as a single-layer neural network, where the input features are linearly combined using weights and a bias term, and then passed 
through a sigmoid activation function.

Input Features (X) --> Linear Combination (Z = WX + B) --> Sigmoid Function (σ(Z)) --> Output (P(Y=1|X))

Where:
•	XXX represents the input features.
•	WWW represents the weights.
•	BBB represents the bias.
•	σσσ represents the sigmoid function.

Mathematical Model:

The logistic regression model can be expressed mathematically as follows:
Linear Combination: Z=WX+BZ = WX + BZ=WX+B
Sigmoid Function: P(Y=1∣X)=σ(Z)=11+e−ZP(Y=1|X) = \sigma(Z) = \frac{1}{1 + e^{-Z}}P(Y=1∣X)=σ(Z)=1+e−Z1 Where eee is the base of the natural logarithm.
The sigmoid function ensures that the output is between 0 and 1, representing the probability of the positive class.

Algorithm:

The algorithm for training a logistic regression model involves optimizing the weights and bias to minimize the error between the predicted probabilities and the
actual outcomes. This is typically done using maximum likelihood estimation and gradient descent.
1.	Initialization:
	Initialize the weights WWW and bias BBB to small random values or zeros.
2.	Forward Pass:
	Compute the linear combination Z=WX+BZ = WX + BZ=WX+B.
	Apply the sigmoid function to get the predicted probabilities P(Y=1∣X)P(Y=1|X)P(Y=1∣X).
3.	Loss Calculation:
	Compute the binary cross-entropy loss: L=−1m∑i=1m[yilog⁡(pi)+(1−yi)log⁡(1−pi)]L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]L=−m1∑i=1m[yilog(pi)+(1−yi)log(1−pi)] Where mmm is the number of samples, yiy_iyi is the actual label, and pip_ipi is the predicted probability.
4.	Backpropagation:
	Compute the gradients of the loss with respect to the weights and bias.
5.	Update Weights and Bias:
	Update the weights and bias using gradient descent: W:=W−η∂L∂WW := W - \eta \frac{\partial L}{\partial W}W:=W−η∂W∂L B:=B−η∂L∂BB := B - \eta \frac{\partial L}{\partial B}B:=B−η∂B∂L Where η\etaη is the learning rate.
6.	Iterate:
	Repeat the forward pass, loss calculation, backpropagation, and weight updates until convergence.


Source Code
Here is a simple implementation of logistic regression using Python and NumPy:

program:
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return [1 if p > 0.5 else 0 for p in predictions]

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print(predictions)

Application Areas / Real World Example

Application Areas:

Medical Diagnosis: Predicting the presence or absence of a disease based on patient data.
Marketing: Determining the likelihood of a customer responding to a marketing campaign.
Finance: Predicting credit default or loan approval based on applicant information.
Social Sciences: Analyzing survey data to understand factors influencing binary outcomes (e.g., voting behavior).

Real World Example:

A healthcare provider uses logistic regression to predict whether a patient has diabetes based on features such as age, BMI, blood pressure, and glucose levels. 
The model outputs a probability that the patient has diabetes, allowing doctors to make informed decisions about further testing or treatment.

References:

Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232.
Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.
Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
Kleinbaum, D. G., & Klein, M. (2010). Logistic Regression: A Self-Learning Text (3rd ed.). Springer.
