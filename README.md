# California Housing Price Prediction: Regularization Techniques

This project demonstrates and compares standard Linear Regression with three key regularization techniques: Ridge (L2), Lasso (L1), and ElasticNet, using Scikit-Learn on the California Housing dataset.

## Project Overview
* Dataset: California Housing dataset (fetch_california_housing) with 8 numerical features such as median income, house age, average rooms, population, and location coordinates.
* Goal: Predict the median house value (MedHouseVal) and analyze how regularization penalizes or eliminates feature coefficients.
* Techniques Compared:
  * Linear Regression: Standard Ordinary Least Squares.
  * Ridge Regression (L2): Adds squared penalty (alpha = 1.0) to shrink coefficients and manage multicollinearity.
  * Lasso Regression (L1): Adds absolute penalty (alpha = 0.1) to shrink less significant coefficients to zero for automated feature selection.
  * ElasticNet: Combines both L1 and L2 penalties (alpha = 0.1).

## Results and Findings

* Linear Regression: MSE = 0.181, R2 Score = 0.695 (baseline weights without penalty)
* Ridge Regression: MSE = 0.205, R2 Score = 0.655 (shrunk large coefficients while keeping all features active)
* Lasso Regression: MSE = 0.237, R2 Score = 0.601 (eliminated multiple non-critical features to 0.0)
* ElasticNet: MSE = 0.220, R2 Score = 0.629 (balanced shrinkage and feature elimination)

## Tools Used
* Language: Python
* Libraries: Scikit-Learn, Pandas, Matplotlib
