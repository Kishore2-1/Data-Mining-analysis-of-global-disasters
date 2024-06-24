# Data-Mining-Analysis-of-Global-Disasters

## Overview

The objective of this project is to perform an in-depth analysis of global disaster data from specific countries utilizing statistical and machine learning approaches. Concentrating on countries such as Cambodia, Ethiopia, Ghana, and others, the research is directed towards investigating trends in occurrences of disasters, their categorizations, as well as consequences like fatalities and financial damages. Through the preprocessing of data, data visualisation, and the application of advanced analytics which include regression analysis, classification, and principal component analysis (PCA), the project aims to unveil insights into the correlations among various factors related to disasters. Through the utilization of these methodologies, the project seeks to contribute to strategies for reducing disaster risk by identifying critical elements that influence the impacts of disasters and facilitating predictive modeling to enhance preparedness and response activities.

## Data Source

The data for the case study was sourced from the DesInventar database, which is a systematic record of disasters maintained by the United Nations Office for Disaster Risk Reduction (UNDRR) and supported by various organizations.

- DesInventar database: https://www.desinventar.net/DesInventar/

This database is utilized to analyze the impact of different types of disasters in different countries, providing critical insights into disaster management, risk reduction strategies, and the socio-economic implications of disasters globally. The data includes information on disaster occurrences, their types, impacts such as deaths and economic losses, and other relevant variables that are essential for conducting comprehensive analyses as outlined in the case study.

## Data Cleaning and Preprocessing

### Data Cleaning
- Manual Cleaning: Initially, columns with more than 90% null values were removed from the dataset to streamline analysis and ensure data integrity. This step reduces unnecessary noise and focuses on relevant variables.

### Handling Missing Values
- Method Used: The `missForest` package in R was employed to impute missing values specifically for columns related to disaster impacts, such as deaths and economic losses. This imputation technique was chosen to maintain data integrity while addressing missing values effectively.

### Standardizing Variables
- Categorical Columns: After cleaning and imputation, categorical columns were standardized to ensure consistency across different countries' data. This step involved mapping various local disaster types (e.g., floods, earthquakes) to a common set of categories.

### Transforming Data Formats
- Imputation Process: Imputed values from `missForest` were integrated back into the dataset, ensuring that each country's disaster impact data was complete and ready for further analysis.
  
### Outlier Removal
- Identification: Outliers in numerical variables were identified using Z-scores. This method helps in identifying extreme values that could distort statistical analyses.
- Filtering: Identified outliers were filtered out from the dataset to ensure subsequent statistical modeling and visualization accurately represent typical disaster impact scenarios.

## Data visualisation

### Data Visualization Techniques Used:

1. Bar Plot (Grouped and Stacked):
  - Grouped Bar Plot: Visualizes counts of deaths, missing persons, and injured individuals across different events using `ggplot2`. Each event category (`Event_Comb`) has bars grouped by these variables (`Deaths`, `Missing`, `Injured`).
  - Shows how `ggplot2` is used to create a grouped bar plot (`geom_bar(stat = "identity", position = "dodge")`).
  - Data is transformed into long format using `gather` from `tidyr` for plotting multiple variables (`Deaths`, `Missing`, `Injured`) against each event (`Event_Comb`).
   - Stacked Bar Plot: Displays the total number of events grouped by decade (`Decade`), showing a stacked bar where each segment represents the count of events in that decade.
  - `ggplot2` is used to create a stacked bar plot for summarizing total events by decade.
  - Decade information (`Decade`) is derived from the `Year` column, and totals are calculated using `dplyr` (`group_by` and `summarise`).

2. Pie Chart:
  - Represents the percentage distribution of infrastructure damage across different event categories (`Event_Comb`). Each category (`Climate-related`, `Natural Disasters`, etc.) is represented by a segment of the pie chart, colored using custom colors.
  - Calculates sums and percentages of infrastructure damage across different event categories (`Event_Comb`).
  - `pie()` function in base R is used to create the pie chart with custom colors (`custom_colors`).

3. Bar Plot (Accuracy Comparison):
  - For Country Classification: Compares accuracy (%) of different classification methods (`K-Means`, `Random Forest`, `XGBoost`) using a grouped bar plot.
  - For Event Classification: Compares accuracy (%) of different classification methods (`K-Means`, `Random Forest`, `XGBoost`) for `Event_Comb` using another grouped bar plot.
  - Compares accuracy metrics for different classification methods (`K-Means`, `Random Forest`, `XGBoost`) for both country and event classification tasks.
  - `ggplot2` is used to visualize the accuracy metrics (`geom_bar(stat = "identity", position = "dodge")`), and custom colors are set for clarity.

4. Scatter Plot with Regression Line:
  - Plots a scatter plot of `x` against `y` and overlays a regression line (`abline`) on the plot. This shows the relationship between two catogories with a fitted linear model.
  - Base R functions (`plot()` and `abline()`) are used to create a scatter plot and overlay a regression line (`model`) to demonstrate the relationship between infrastructure and damages in roads.

## Data Analysis Methods

This analysis leverages several advanced statistical and machine learning methods to understand and predict patterns in the dataset. The following sections explain the usage of each method:

### Linear Regression

Linear regression is used to model the relationship between a dependent variable and one independent variable. 

In this analysis:
- Purpose: To understand how `Losses_USD` (economic losses) affects the number of `Injured` individuals.
- Implementation: The simple linear regression model was fit using `lm(y ~ x, data = data)` where `x` is `Losses_USD` and `y` is `Injured`.
- Visualization: The relationship was visualized with a scatter plot and a regression line.

### Multi-Linear Regression

Multi-linear regression extends linear regression by modeling the relationship between a dependent variable and multiple independent variables.

- Purpose: To predict `Deaths` and `Losses_USD` using principal components derived from PCA.
- Implementation: The model was built using `lm(Deaths ~ PC1 + PC2 + PC3, data = train_data)` and similarly for `Losses_USD`.
- Evaluation: The model's performance was assessed using Root Mean Squared Error (RMSE).

### XGBoost

XGBoost is a powerful machine learning algorithm based on gradient boosting. 

In this analysis:
- Purpose: To improve prediction accuracy for `Deaths` and `Losses_USD`.
- Implementation: The model was trained using the `xgb.train` function with specified parameters such as learning rate, max depth, and number of rounds.
- Evaluation: RMSE and R-squared values were calculated to evaluate model performance.

### Random Forest

Random Forest is an ensemble learning method that operates by constructing multiple decision trees. Although not explicitly used in the final provided code, it was mentioned for imputation of missing values:

- Purpose: To handle missing values in the dataset effectively.
- Implementation: The `missForest` package was used to perform imputation, improving the dataset's completeness for further analysis.

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms a large set of variables into a smaller one while retaining most of the variance in the data.

In this analysis:
- Purpose: To reduce the dimensionality of the dataset and identify the most significant components.
- Implementation: The `prcomp` function was used to compute principal components.
- Result: The first three principal components were retained for subsequent modeling.

### Classification

While the provided code primarily focuses on regression and dimensionality reduction, classification could be integrated to categorize events or predict categorical outcomes. This could involve using algorithms like logistic regression, decision trees, or SVM, depending on the target variable.

### Correlation

Correlation analysis helps identify relationships between variables.

- Purpose: To explore the interdependencies between various numerical variables in the dataset.
- Implementation: The correlation matrix was computed and visualized using `corrplot` to identify significant correlations.
- Insight: This analysis informs the selection of variables for further modeling.

## Results and Key Findings

This section highlights the significant findings and insights derived from our analyses, focusing on predictive models, correlations, and patterns identified in the disaster-related dataset.

### Multi-Linear Regression

1. Model to Predict Deaths:
   - Adjusted R-squared: 0.5035
   - Insight: The multi-linear regression model explains approximately 50.35% of the variance in the number of deaths. This indicates a moderate level of predictability using the selected principal components.

2. Model to Predict USD Losses:
   - Adjusted R-squared: 0.7353
   - Insight: This model accounts for 73.53% of the variance in economic losses (USD). The higher adjusted R-squared value suggests that the selected features are more effective in predicting economic losses than deaths.

### XGBoost

1. Model to Predict Deaths:
   - R-squared: 0.6623
   - Insight: The XGBoost model demonstrates improved performance over the multi-linear regression model, explaining 66.23% of the variance in deaths. This highlights the efficacy of XGBoost in capturing complex relationships in the data.

2. Model to Predict USD Losses:
   - R-squared: 0.8741
   - Insight: XGBoost significantly outperforms the multi-linear regression model in predicting economic losses, with an R-squared value of 87.41%. This indicates a strong predictive capability, making it a valuable tool for estimating economic impacts of disasters.

### Correlation Analysis

1. Correlation between USD Losses and Injured:
   - Adjusted R-squared: 0.4374
   - Insight: The correlation analysis reveals that about 43.74% of the variance in the number of injured individuals can be explained by economic losses (USD). This moderate correlation suggests that as economic losses increase, the number of injuries also tends to rise.

### Classification Models

1. Random Forest:
   - Event Combination Accuracy: 52.13%
   - Country Accuracy: 67.91%
   - Insight: The random forest classifier shows moderate accuracy in predicting event combinations and better performance in classifying events by country. This indicates its potential for categorizing events based on geographical regions.

2. XGBoost:
   - Country Accuracy: 79.92%
   - Event Combination Accuracy: 99.12%
   - Insight: XGBoost demonstrates high accuracy, particularly in classifying event combinations with an impressive 99.12% accuracy. This underscores its robustness and reliability for classification tasks.

3. K-Means Clustering:
   - Country Accuracy: 6.28%
   - Event Combination Accuracy: 17.71%
   - Insight: K-Means clustering shows limited accuracy in both country and event combination classifications. This suggests that clustering may not be as effective as supervised learning methods for this dataset.

## Conclusion

This project represents a comprehensive analysis of global disaster data, focusing on key countries such as Cambodia, Ethiopia, and Ghana, among others. Through meticulous data cleaning, preprocessing, and advanced statistical and machine learning techniques, we have gained valuable insights into disaster patterns, their impacts, and critical factors influencing these impacts. Here are the key takeaways from our analysis:

### Insights from Data Analysis

1. Trend Analysis:
   - Our visualizations, including grouped and stacked bar plots, provided clear trends in disaster occurrences, fatalities, and injuries across different decades and event categories. This highlights the temporal and categorical shifts in disaster impacts.

2. Impact of Economic Losses:
   - Linear regression analysis showed a moderate relationship between economic losses (`Losses_USD`) and the number of injured individuals. This indicates that higher economic damages often correlate with increased injuries.

3. Predictive Modeling:
   - Multi-linear regression models demonstrated that principal components derived from PCA could moderately predict deaths and economic losses, with adjusted R-squared values of 0.5035 and 0.7353, respectively.
   - XGBoost models significantly improved prediction accuracy, with R-squared values of 0.6623 for deaths and 0.8741 for economic losses. This underscores the power of advanced machine learning techniques in capturing complex relationships in disaster data.

4. Correlation Analysis:
   - A notable correlation between economic losses and the number of injured individuals was identified, explaining 43.74% of the variance. This finding suggests that economic impact assessments can provide insights into human impacts during disasters.

5. Classification Performance:
   - Random Forest and XGBoost classifiers showed varying degrees of accuracy in predicting disaster events and country classifications. XGBoost, in particular, achieved remarkable accuracy in classifying event combinations (99.12%) and countries (79.92%), demonstrating its robustness for classification tasks.

## Future Work

Future research could focus on expanding the dataset to include more countries and disaster types, integrating real-time data for dynamic modeling, and exploring additional machine learning techniques for improved predictive accuracy. Collaborations with local governments and international agencies can further enhance the practical applications of these findings.

In conclusion, this project underscores the critical role of data-driven approaches in understanding and mitigating the impacts of disasters. By leveraging statistical and machine learning methodologies, we can uncover valuable insights that contribute to more effective disaster risk reduction and management strategies globally.

---

This comprehensive analysis and its findings have been documented and made available on GitHub, providing a valuable resource for researchers, policymakers, and practitioners in the field of disaster management. The repository includes detailed code, data visualizations, and results, promoting transparency and facilitating further research and collaboration. 

Feel free to explore the project on GitHub [here](https://github.com/Kishore2-1/Data-Mining-analysis-of-global-disasters/)



