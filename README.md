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
