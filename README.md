# PySpark Linear Regression Modeling

This repository contains code and documentation for building and evaluating various regression models using PySpark. The project focuses on predicting the number of crew members on a cruise ship using data-driven approaches. Multiple algorithms, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor, are used to compare model performances.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup and Installation](#setup)
- [Implementation](#implementation)
- [Model Evaluation](#model-evaluation)
- [Key Insights](#key-insights)
- [Conclusion](#conclusion)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The aim of this project is to use PySpark's MLlib library to build, evaluate, and compare various regression models to predict the number of crew members on cruise ships based on multiple features. Key assumptions of linear regression are assessed and validated as part of the analysis.

## Dataset

The dataset used in this project, `cruise_ship_info.csv`, contains various features of cruise ships, such as:
- Age
- Tonnage
- Number of passengers
- Length of the ship
- Number of cabins
- Passenger density

### Target Variable
- **Crew**: The number of crew members on the cruise ship.

## Setup

### Requirements
- Python
- PySpark
- Java 8
- Jupyter Notebook (optional, for running the code interactively)
- I have used Colab for this project

### Setup and Installation
To run this project, you need to set up PySpark and Java. Below are the steps to set up the environment in Colab:

```bash
# Download Java and Spark
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!tar xf spark-3.2.1-bin-hadoop3.2.tgz
!pip install -q findspark

# Set up the paths
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"

# Create a Spark session
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) # Property used to format output tables better
spark.conf.set("spark.sql.caseSensitive", True) # Avoid error "Found duplicate column(s) in the data schema"
spark

```

## Implementation

### Key Steps:
1. **Data Preprocessing**: Loaded the data, handled categorical variables, scaled numerical features, and prepared the data for modeling.
2. **Model Training**: Trained multiple regression models using PySpark’s MLlib.
3. **Assumptions Validation**: Checked assumptions of linear regression like linearity, normality of residuals, homoscedasticity, and absence of multicollinearity.
4. **Model Evaluation**: Evaluated models using R-squared, Adjusted R-squared, RMSE, MAE, AIC, and BIC.

## Model Evaluation

The models were evaluated based on the following metrics:

- **R-squared**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
- **Adjusted R-squared**: Adjusts R-squared for the number of predictors in the model.
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors in predictions.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between actual and predicted values.
- **AIC (Akaike Information Criterion)**: Evaluates model complexity by penalizing overfitting.
- **BIC (Bayesian Information Criterion)**: Similar to AIC but with a stronger penalty for models with more parameters.

### Summary of Results

| Model                       | R-squared | Adjusted R-squared | RMSE     | MAE     | AIC       | BIC       |
|-----------------------------|-----------|--------------------|----------|---------|-----------|-----------|
| Linear Regression           | 0.797906  | 0.747382           | 1.540703 | 0.700623| 38.798822 | 47.402745 |
| Gradient Boosting Regressor | 0.775518  | 0.719397           | 1.623801 | 0.726914| 42.055727 | 50.659650 |
| Decision Tree Regressor     | 0.774926  | 0.718658           | 1.625940 | 0.836743| 42.137355 | 50.741278 |
| Random Forest Regressor     | 0.749559  | 0.686949           | 1.715121 | 0.881034| 45.447983 | 54.051906 |

### Key Insights

- **Linear Regression** showed the highest R-squared and Adjusted R-squared values, indicating better performance in terms of explained variance compared to other models.
- **Gradient Boosting Regressor** and **Decision Tree Regressor** performed similarly, with slightly lower R-squared and higher RMSE than Linear Regression, but they offer robustness against non-linear patterns.
- **Random Forest Regressor** showed the lowest performance among the evaluated models, with higher RMSE and MAE values, indicating less accurate predictions.

These results suggest that while simpler models like Linear Regression can provide good baseline performance, more complex ensemble methods can offer additional robustness against overfitting and non-linear relationships, albeit sometimes at the cost of slightly higher prediction errors.

## Conclusion

The project demonstrates the power of PySpark for large-scale data processing and machine learning. ensemble methods can offer additional robustness against overfitting and non-linear relationships, albeit sometimes at the cost of slightly higher prediction errors.

## References
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [MLlib Regression Guide](https://spark.apache.org/docs/latest/ml-classification-regression.html)

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
