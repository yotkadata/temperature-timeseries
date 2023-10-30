# Temperature Forecast using Time Series Analysis

**Predicting tomorrow's temperature based on a time series analysis of temperature data from 1876 to today.**

This Jupyter Notebook showcases a comprehensive analysis and modeling of historical temperature data for Berlin Tempelhof (1876 to 2022) to predict temperatures for the first weeks of 2023. The notebook is organized into multiple sections, each focusing on a specific aspect of time series analysis. Here is a brief overview:

- **Introduction and Setup:** Introduces the notebook and its objectives. Sets up the required libraries and dependencies. Libraries such as pandas, numpy, matplotlib, and others are imported.
- **Data Loading:** Describes how to load the time series data from external sources, in this case the [European Climate Assessment & Dataset](https://www.ecad.eu/utils/showselection.php?oo8sk8dr3n14m8tb21ejl2hckn). The data is loaded into a pandas DataFrame, and initial explorations are conducted to understand its structure and content.
- **Data Preprocessing:** Focuses on cleaning and preprocessing the data to make it suitable for analysis. Includes handling missing values, converting data types, and other preprocessing steps.
- **Exploratory Data Analysis (EDA):** Conducts a very brief exploratory data analysis to uncover patterns, trends, and anomalies in the data. Utilizes matplotlib and seaborn for visualizing different aspects of the time series data.
- **Time Series Decomposition:** After denoising, decomposes the time series data into its constituent components: trend, seasonality, and residual. Employs statistical methods and algorithms for time series decomposition.
- **Modeling:** Builds models to capture the trend, seasonality, and noise in the data. The resulting models are saved in the `models` directory.
- **Machine Learning Models and Techniques used:** Linear Regression, Auto ARIMA, Pipeline, Grid Search Cross-Validation

_This project was part of the Data Science Bootcamp at SPICED Academy from April to June 2023._
