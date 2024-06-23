# This script installs necessary packages
install.packages(c("missForest", "openxlsx", "caret", "readxl", "ggplot2", "corrplot", "tidyr", "xgboost", "randomForest", "cluster", "scales"))

##### Data Preparation

# Load Required Libraries
library(missForest)
library(openxlsx)
library(caret)
library(readxl)

# Read in the data
data <- read.xlsx("processed.xlsx")

# Data Imputation for All_country Data to Fill NA Values Using Random Forest Imputation
# Number of Columns where Random Forest is Imputed
columns_to_impute <- 5:18

# Impute Missing Values
imputed_data <- missForest(data[, columns_to_impute])
data[, columns_to_impute] <- imputed_data$ximp


##### Data Cleaning

# Read in the data
data <- read.xlsx("processed.xlsx")

# Extract Numerical Variables for Z-score Calculation
numerical_data <- df[, sapply(df, is.numeric)]

# Calculate Z-scores for Each Numerical Variable
z_scores <- scale(numerical_data)

# Set a Threshold for Z-scores (e.g., 3 Standard Deviations)
z_threshold <- 3

# Identify Rows with Outliers Based on Z-scores
outliers <- apply(abs(z_scores) > z_threshold, 1, any)

# Remove Rows with Outliers
data <- data[!outliers, ]

# Print the Number of Removed Outliers
cat("Number of removed outliers:", sum(outliers), "\n")

# Print Data Types
data_types <- sapply(data, typeof)
print(data_types)

# Print Column Names
column_names <- names(data)
print(column_names)



##### Data Visulisation

## BarPlot
# Load necessary libraries
library(ggplot2)
library(tidyr)

# Select relevant columns
Events <- df$Event_Comb
bargraph <- data.frame(Event = Events, Deaths = df$Deaths, Missing = df$Missing, Injured = df$Injured)

# Convert data frame to long format
df_long <- gather(bargraph, key = "Variable", value = "Value", -Event)

# Plotting with ggplot2
ggplot(df_long, aes(x = Event, y = Value, fill = Variable)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.8) +
  labs(title = "Deaths, Missing, Injured v/s Events",
       x = "Events",
       y = "Count") +
  scale_fill_manual(values = c("Deaths" = "#0A9396", "Missing" = "#EE9B00", "Injured" = "#BB3E03")) +
  theme_minimal()

# Save the plot (optional)
ggsave("plots/deaths_missing_injured.png", plot = last_plot(), width = 10, height = 6, units = "in", dpi = 300)


## Stacked bar
# Load necessary libraries
library(ggplot2)
library(dplyr)

# Assuming 'df' has 'Year' and 'Event' columns
# Extract the decade from the 'Year' column
df$Decade <- as.factor((as.numeric(substr(df$Year, 1, 3)) * 10))

# Group by decade and calculate total number of events
event_counts <- df %>%
  group_by(Decade) %>%
  summarise(Total_Events = n())

# Plot the bar chart
ggplot(event_counts, aes(x = Decade, y = Total_Events, fill = Decade)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Total_Events), vjust = -0.5, size = 3) +
  labs(title = "Total Number of Events Decade-wise",
       x = "Decade",
       y = "Total Events") +
  theme_minimal()

# Save the plot (optional)
ggsave("plots/total_events_decade_wise.png", plot = last_plot(), width = 10, height = 6, units = "in", dpi = 300)


## Pie Charts
# Load necessary libraries
library(ggplot2)

# Calculate sums for Infrastructure damage by Event_Comb category
s1 <- sum(df$Infrastructure[df$Event_Comb == "Climate-related"])
s2 <- sum(df$Infrastructure[df$Event_Comb == "Natural Disasters"])
s3 <- sum(df$Infrastructure[df$Event_Comb == "Man-Made"])
s4 <- sum(df$Infrastructure[df$Event_Comb == "Environmental Changes and Accidents"])
s5 <- sum(df$Infrastructure[df$Event_Comb == "Biological and Health-related"])
s6 <- sum(df$Infrastructure[df$Event_Comb == "Weather-related"])
s7 <- sum(df$Infrastructure[df$Event_Comb == "Water-related"])
s8 <- sum(df$Infrastructure[df$Event_Comb == "Miscellaneous"])

# Create a data frame for pie chart
Event <- c(
  "Climate-related Events",
  "Natural Disasters",
  "Man-Made Incidents",
  "Environmental Changes and Accidents",
  "Biological and Health-related Events",
  "Weather-related Events",
  "Miscellaneous Events",
  "Water-related Incident"
)
InfrastructureDamage <- c(s1, s2, s3, s4, s5, s6, s8, s7)
data <- data.frame(Event, InfrastructureDamage)

# Calculate percentages
data$Percentage <- (data$InfrastructureDamage / sum(data$InfrastructureDamage)) * 100

# Define custom colors
custom_colors <- c("#001219", "#005F73", "#0A9396", "#94D2BD", "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03")

# Create a pie chart
pie(data$Percentage, labels = paste(data$Event, "\n", sprintf("%.1f%%", data$Percentage)),
    col = custom_colors,
    main = "Infrastructure Damage by Events")

# Save the plot (optional)
dev.copy(png, "plots/infrastructure_damage_pie.png")
dev.off()


##### Principal Component Analysis(PCA)

# Extract numerical variables for PCA
numerical_data <- df[, sapply(df, is.numeric)]

# Check for missing values and replace them if needed
numerical_data[is.na(numerical_data)] <- 0  # Replace missing values with 0 (you may choose a different strategy)

# Standardize the numerical variables
scaled_data <- scale(numerical_data)

# Apply PCA
pca_result <- prcomp(scaled_data)

# Explore PCA results
summary(pca_result)

# Create a scree plot with different colors for points and black line
plot(cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2),
     type = "b",
     xlab = "Number of Principal Components",
     ylab = "Proportion of Variance Explained",
     col = "#AE2012",  # Specify color for points
     lty = 1,  # Specify line type for lines
     main = "Scree Plot with Black Line")

# Choose the number of principal components based on the plot and your requirements
num_components <- 3  # You can adjust this based on the plot

# Extract the principal components
principal_components <- as.data.frame(pca_result$x[, 1:num_components])

# Merge principal components with the original dataframe if needed
final_data <- cbind(df, principal_components)

# Print final data
final_data


#### Multi-Linear Regression
## Model to predict Death using Multi-Linear Regression

# Data preprocessing: Remove any missing values
final_data <- na.omit(final_data)

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
splitIndex <- createDataPartition(final_data$Deaths, p = 0.7, list = FALSE)
train_data <- final_data[splitIndex, ]
test_data <- final_data[-splitIndex, ]

# Build a linear regression model using principal components
model <- lm(Deaths ~ PC1 + PC2 + PC3 , data = train_data)  # Include the selected principal components

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Evaluate the model
rmse <- sqrt(mean((test_data$Deaths - predictions)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# View the summary of the multi-linear regression model
summary(model)

## Model to predict USD Losses using Multi-Linear Regression

# Data preprocessing: Remove any missing values
final_data <- na.omit(final_data)

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
splitIndex <- createDataPartition(final_data$Losses_USD, p = 0.7, list = FALSE)
train_data <- final_data[splitIndex, ]
test_data <- final_data[-splitIndex, ]

# Build a linear regression model using principal components
model <- lm(Losses_USD ~ PC1 + PC2 + PC3 , data = train_data)  # Include the selected principal components

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Evaluate the model
rmse <- sqrt(mean((test_data$Losses_USD - predictions)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# View the summary of the linear regression model
summary(model)

#### XGBoost

## Model to predict Death using XGBoost
# Load Required Libraries
library(xgboost)
library(data.table)

# Assuming your dataframe with principal components is named final_data
# Assuming PC1, PC2, PC3, ... are the actual names of the principal components you selected

# Convert the data to a DMatrix (a format suitable for XGBoost)
dtrain <- xgb.DMatrix(as.matrix(train_data[, c("PC1", "PC2", "PC3")]), label = train_data$Deaths)
dtest <- xgb.DMatrix(as.matrix(test_data[, c("PC1", "PC2", "PC3")]))

# Set up XGBoost parameters
params <- list(
  objective = "reg:squarederror",  # For regression tasks
  eval_metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  booster = "gbtree",  # Use tree-based models
  eta = 0.1,  # Learning rate
  max_depth = 6,  # Maximum depth of a tree
  subsample = 0.8,  # Subsample ratio of the training instances
  colsample_bytree = 0.8  # Subsample ratio of columns when constructing each tree
)

# Train the XGBoost model
model_xgb <- xgb.train(params, dtrain, nrounds = 100)

# Make predictions on the test set
predictions_xgb <- predict(model_xgb, dtest)

# Evaluate the XGBoost model
rmse_xgb <- sqrt(mean((test_data$Deaths - predictions_xgb)^2))
cat("XGBoost - Root Mean Squared Error (RMSE):", rmse_xgb, "\n")

# Get R-squared for XGBoost
rsquared_xgb <- 1 - (sum((test_data$Deaths - predictions_xgb)^2) / sum((test_data$Deaths - mean(test_data$Deaths))^2))
cat("XGBoost - R-squared:", rsquared_xgb, "\n")


## Model to predict USD Losses using XGBoost
# Load Required Libraries
library(xgboost)
library(data.table)

# Assuming your dataframe with principal components is named final_data
# Assuming PC1, PC2, PC3, ... are the actual names of the principal components you selected
# Convert the data to a DMatrix (a format suitable for XGBoost)

dtrain <- xgb.DMatrix(as.matrix(train_data[, c("PC1", "PC2", "PC3")]), label = train_data$Losses_USD)
dtest <- xgb.DMatrix(as.matrix(test_data[, c("PC1", "PC2", "PC3")]))

# Set up XGBoost parameters
params <- list(
  objective = "reg:squarederror",  # For regression tasks
  eval_metric = "rmse",  # Root Mean Squared Error as the evaluation metric
  booster = "gbtree",  # Use tree-based models
  eta = 0.1,  # Learning rate
  max_depth = 6,  # Maximum depth of a tree
  subsample = 0.8,  # Subsample ratio of the training instances
  colsample_bytree = 0.8  # Subsample ratio of columns when constructing each tree
)

# Train the XGBoost model
model_xgb <- xgb.train(params, dtrain, nrounds = 100)

# Make predictions on the test set
predictions_xgb <- predict(model_xgb, dtest)

# Evaluate the XGBoost model
rmse_xgb <- sqrt(mean((test_data$Losses_USD - predictions_xgb)^2))
cat("XGBoost - Root Mean Squared Error (RMSE):", rmse_xgb, "\n")

# Get R-squared for XGBoost
rsquared_xgb <- 1 - (sum((test_data$Losses_USD - predictions_xgb)^2) / sum((test_data$Losses_USD - mean(test_data$Losses_USD))^2))
cat("XGBoost - R-squared:", rsquared_xgb, "\n")


#### Corelation graph between different Variables
# Load Required Libraries
library(corrplot)

colsec <- 5:18

str(df[, colsec])

# Assuming 'final_data' is your data frame
correlation_matrix <- cor(df[, colsec])
# Plot the correlation matrix
corrplot(correlation_matrix, method = "color")

#### Linear Regression

## Model for corelation between USD Losses and Injured

x <- df$Losses_USD
y <- df$Injured

# Create a data frame
data <- data.frame(x, y)

# Split the data into training and testing sets
train_indices <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train a linear regression model
model <- lm(y ~ x, data = train_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Calculate the model performance (e.g., Mean Squared Error)
mse <- mean((test_data$y - predictions)^2)

cat("Mean Squared Error:", mse, "\n")
summary(model)

# Scatter Plot and Regression Line
plot(data$x, data$y, main = "Scatter Plot with Regression Line", 
     xlab = "Losses in USD", ylab = "Injured", col = "#0A9396")
abline(model, col = "#EE9B00")


## Model for corelation between Duration and Evacuated

x <- df$Duration
y <- df$Evacuated

# Create a data frame
data <- data.frame(x, y)

# Split the data into training and testing sets
train_indices <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train a linear regression model
model <- lm(y ~ x, data = train_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Calculate the model performance (e.g., Mean Squared Error)
mse <- mean((test_data$y - predictions)^2)
cat("Mean Squared Error:", mse, "\n")
summary(model)

# Scatter plot and Regression Line
plot(data$x, data$y, main = "Scatter Plot with Regression Line", 
     xlab = "Duration", ylab = "Evacuated", col = "#94D2BD")
abline(model, col = "#BB3E03")


## Model for corelation between Infrastructure and Damages in Roads

x <- df$Infrastructure
y <- df$Damages_in_roads_Mts

# Create a data frame
data <- data.frame(x, y)

# Split the data into training and testing sets
train_indices <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train a linear regression model
model <- lm(y ~ x, data = train_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Calculate the model performance (e.g., Mean Squared Error)
mse <- mean((test_data$y - predictions)^2)
cat("Mean Squared Error:", mse, "\n")

summary(model)

# Scatter plot and Regression Line
plot(data$x, data$y, main = "Scatter Plot with Regression Line", 
     xlab = "Infrastructure", ylab = "Damages in Roads (mts)", col = "#BB3E03")
abline(model, col = "#005F73")


#### Classification

## Different types of Classification
# Load Required Libraries
library(readxl)

# Assuming 'your_data' is your data frame
column_names <- names(final_data)

your_data <- final_data


## classification of country using Random Forest
# Load Required Libraries
library(caret)
library(randomForest)

# Assuming 'your_data' is your data frame
# Replace 'Country' with the actual name of your target variable
your_data$Country <- as.factor(your_data$Country)

# Create a formula for the classification model
formula <- as.formula("Country ~ Event_Comb + Event + Continent + Year + Month + Losses_USD + Damages_in_crops_Ha + Lost_Cattle + Damages_in_roads_Mts + Duration + With_Victims + Deaths + Injured + Missing + Affected + Evacuated + Infrastructure")

# Create a training dataset
set.seed(123)  # for reproducibility
splitIndex <- createDataPartition(your_data$Country, p = 0.8, list = FALSE)
train_data <- your_data[splitIndex, ]
test_data <- your_data[-splitIndex, ]

# Train the random forest model
model <- randomForest(formula, data = train_data)

# Make predictions on the test set
predictions <- predict(model, newdata = test_data)

# Evaluate the model
confusion_matrix <- confusionMatrix(predictions, test_data$Country)
print(confusion_matrix)
accuracy_Country <- confusion_matrix$overall["Accuracy"]

# Accuracy
randomforest_accuracy_country <- accuracy_Country * 100


##  Classification of Events using Random Forest
# Load Required Libraries
library(caret)
library(randomForest)

# Replace 'Event_Comb' with the actual name of your target variable
your_data$Event_Comb <- as.factor(your_data$Event_Comb)

# Create a formula for the classification model
formula_event_comb <- as.formula("Event_Comb ~ Country + Event + Continent + Year + Month + Losses_USD + Damages_in_crops_Ha + Lost_Cattle + Damages_in_roads_Mts + Duration + With_Victims + Deaths + Injured + Missing + Affected + Evacuated + Infrastructure")

# Create a training dataset
set.seed(123)  # for reproducibility
splitIndex_event_comb <- createDataPartition(your_data$Event_Comb, p = 0.8, list = FALSE)
train_data_event_comb <- your_data[splitIndex_event_comb, ]
test_data_event_comb <- your_data[-splitIndex_event_comb, ]

# Train the random forest model for Event_Comb
model_event_comb <- randomForest(formula_event_comb, data = train_data_event_comb)

# Make predictions on the test set for Event_Comb
predictions_event_comb <- predict(model_event_comb, newdata = test_data_event_comb)

# Evaluate the model for Event_Comb
confusion_matrix_event_comb <- confusionMatrix(predictions_event_comb, test_data_event_comb$Event_Comb)
print(confusion_matrix_event_comb)


# Extract accuracy from the confusion matrix
accuracy_event_comb <- confusion_matrix_event_comb$overall["Accuracy"]

randomforest_Event_comb_Accuracy <- accuracy_event_comb * 100

##  Classification of Events using K-Means
# Load Required Libraries
library(caret)
library(cluster)

# Replace 'Event_Comb' with the actual name of your target variable
your_data$Event_Comb <- as.factor(your_data$Event_Comb)

# Extract relevant variables for clustering
cluster_data_event_comb <- your_data[, c("Country", "Event", "Continent", "Year", "Month", "Losses_USD", "Damages_in_crops_Ha", "Lost_Cattle", "Damages_in_roads_Mts", "Duration", "With_Victims", "Deaths", "Injured", "Missing", "Affected", "Evacuated", "Infrastructure")]

# Check for missing values and replace them if needed
cluster_data_event_comb[is.na(cluster_data_event_comb)] <- 0  # Replace missing values with 0 (you may choose a different strategy)

# Identify numeric columns
numeric_cols <- sapply(cluster_data_event_comb, is.numeric)

# Standardize the numerical variables
scaled_cluster_data_event_comb <- scale(cluster_data_event_comb[, numeric_cols])

# Apply k-means clustering
k <- 3  # You can adjust the number of clusters based on your analysis
kmeans_result_event_comb <- kmeans(scaled_cluster_data_event_comb, centers = k)

# Add the cluster assignments to the original dataframe
your_data$Cluster_Event_Comb <- as.factor(kmeans_result_event_comb$cluster)

# Print the cluster centers
print(kmeans_result_event_comb$centers)

# Evaluate the clustering results
conf_matrix <- table(your_data$Event_Comb, your_data$Cluster_Event_Comb)

# Calculate accuracy-like measure
accuracy_like_measure <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print the accuracy-like measure
Kmeans_Event_Comb_Accuracy <-accuracy_like_measure*100


##  Classification of country using K-Means
# Load Required Libraries
library(caret)
library(cluster)

# Replace 'Country' with the actual name of your target variable
your_data$Country <- as.factor(your_data$Country)

# Extract relevant variables for clustering
cluster_data_country <- your_data[, c("Event_Comb", "Event", "Continent", "Year", "Month", "Losses_USD", "Damages_in_crops_Ha", "Lost_Cattle", "Damages_in_roads_Mts", "Duration", "With_Victims", "Deaths", "Injured", "Missing", "Affected", "Evacuated", "Infrastructure")]

# Check for missing values and replace them if needed
cluster_data_country[is.na(cluster_data_country)] <- 0  # Replace missing values with 0 (you may choose a different strategy)

# Identify numeric columns
numeric_cols_country <- sapply(cluster_data_country, is.numeric)

# Standardize the numerical variables
scaled_cluster_data_country <- scale(cluster_data_country[, numeric_cols_country])

# Apply k-means clustering
k_country <- 3  # You can adjust the number of clusters based on your analysis
kmeans_result_country <- kmeans(scaled_cluster_data_country, centers = k_country)

# Add the cluster assignments to the original dataframe
your_data$Cluster_Country <- as.factor(kmeans_result_country$cluster)

# Print the cluster centers
print(kmeans_result_country$centers)

# Evaluate the clustering results
conf_matrix_country <- table(your_data$Country, your_data$Cluster_Country)

# Calculate accuracy-like measure
accuracy_like_measure_country <- sum(diag(conf_matrix_country)) / sum(conf_matrix_country)

# Print the accuracy-like measure
Kmeans_Country_Accuracy <- accuracy_like_measure_country * 100


##  Classification of Events using XGBoost
# Load Required Libraries
library(xgboost)

# Assuming 'Event_Comb' is the target variable for classification
your_data$Event_Comb <- as.factor(your_data$Event_Comb)

# Extract relevant variables for XGBoost classification
xgboost_data_event_comb <- your_data[, c("Country", "Event", "Continent", "Year", "Month", "Losses_USD", "Damages_in_crops_Ha", "Lost_Cattle", "Damages_in_roads_Mts", "Duration", "With_Victims", "Deaths", "Injured", "Missing", "Affected", "Evacuated", "Infrastructure")]

# Check for missing values and replace them if needed
xgboost_data_event_comb[is.na(xgboost_data_event_comb)] <- 0  # Replace missing values with 0 (you may choose a different strategy)

# Create dummy variables for categorical features
xgboost_data_event_comb <- model.matrix(~.-1, data = xgboost_data_event_comb)

# Prepare the target variable
labels_event_comb <- as.integer(your_data$Event_Comb) - 1  # XGBoost expects labels starting from 0

# Create a DMatrix for XGBoost
dtrain_event_comb <- xgb.DMatrix(data = as.matrix(xgboost_data_event_comb), label = labels_event_comb)

# Set XGBoost parameters
params <- list(
  objective = "multi:softmax",  # Multiclass classification
  num_class = length(levels(your_data$Event_Comb)),  # Number of classes
  eval_metric = "mlogloss"  # Multiclass logloss as evaluation metric
)

# Train the XGBoost model
xgb_model_event_comb <- xgboost(params = params, data = dtrain_event_comb, nrounds = 10)

# Make predictions on the training set
predictions_xgb_event_comb <- predict(xgb_model_event_comb, as.matrix(xgboost_data_event_comb))

# Convert predictions to integer labels
predictions_xgb_event_comb <- as.integer(predictions_xgb_event_comb) + 1

# Evaluate the accuracy of the model
accuracy_xgb_event_comb <- sum(predictions_xgb_event_comb == as.integer(your_data$Event_Comb)) / length(your_data$Event_Comb)

# Print the accuracy
XGBoost_Accuracy_Event_Comb <- accuracy_xgb_event_comb*100


##  Classification of Country using XGBoost
# Load Required Libraries
library(xgboost)

your_data$Country <- as.factor(your_data$Country)

# Extract relevant variables for XGBoost classification
xgboost_data_country <- your_data[, c("Event_Comb", "Event", "Continent", "Year", "Month", "Losses_USD", "Damages_in_crops_Ha", "Lost_Cattle", "Damages_in_roads_Mts", "Duration", "With_Victims", "Deaths", "Injured", "Missing", "Affected", "Evacuated", "Infrastructure")]

# Check for missing values and replace them if needed
xgboost_data_country[is.na(xgboost_data_country)] <- 0 

# Create dummy variables for categorical features
xgboost_data_country <- model.matrix(~.-1, data = xgboost_data_country)

# Prepare the target variable
labels_country <- as.integer(your_data$Country) - 1  

# Create a DMatrix for XGBoost
dtrain_country <- xgb.DMatrix(data = as.matrix(xgboost_data_country), label = labels_country)

# Set XGBoost parameters
params <- list(
  objective = "multi:softmax",  # Multiclass classification
  num_class = length(levels(your_data$Country)),  # Number of classes
  eval_metric = "mlogloss"  # Multiclass logloss as evaluation metric
)

# Train the XGBoost model
xgb_model_country <- xgboost(params = params, data = dtrain_country, nrounds = 10)

# Make predictions on the training set
predictions_xgb_country <- predict(xgb_model_country, as.matrix(xgboost_data_country))

# Convert predictions to integer labels
predictions_xgb_country <- as.integer(predictions_xgb_country) + 1

# Evaluate the accuracy of the model
accuracy_xgb_country <- sum(predictions_xgb_country == as.integer(your_data$Country)) / length(your_data$Country)

# Print the accuracy
XGBoost_Accuracy_for_Country <- accuracy_xgb_country*100


#### Data Visualisation to show the accuracy for different classification

##  Bar Graph to show the accuracy For Country
# Load Required Libraries
library(ggplot2)

# Accuracy values
accuracy_values <- c(
  Kmeans_Country_Accuracy,
  randomforest_accuracy_country,
  XGBoost_Accuracy_for_Country
)

# Classification methods
methods <- c("K-Means", "Random Forest", "XGBoost")

# Create a data frame
accuracy_df <- data.frame(Method = methods, Accuracy = accuracy_values)

# Assuming 'accuracy_df' is your data frame

custom_colors <- c("#19AADE", "#1AC9C6", "#C7F9EE")

# Plot the bar plot with different colors
ggplot(accuracy_df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  labs(title = "Accuracy Plot for Country Classification",
       x = "Classification Method",
       y = "Accuracy (%)") +
  scale_fill_manual(values = custom_colors) +  # Set custom colors
  theme_minimal()

##  Bar Graph to show the accuracy For Events
# Load Required Libraries
library(ggplot2)

# Accuracy values for Event_Comb
accuracy_values_event_comb <- c(
  Kmeans_Event_Comb_Accuracy,
  randomforest_Event_comb_Accuracy,
  XGBoost_Accuracy_Event_Comb
)

# Classification methods for Event_Comb
methods_event_comb <- c("K-Means", "Random Forest", "XGBoost")

# Create a data frame for Event_Comb
accuracy_df_event_comb <- data.frame(Method = methods_event_comb, Accuracy = accuracy_values_event_comb)

# Assuming 'accuracy_df_event_comb' is your data frame
custom_colors_event_comb <- c("#142459", "#1AC9E6", "#6CF0D2")

# Plot the bar plot with different colors for Event_Comb
ggplot(accuracy_df_event_comb, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  labs(title = "Accuracy Plot for Event_Comb Classification",
       x = "Classification Method",
       y = "Accuracy (%)") +
  scale_fill_manual(values = custom_colors_event_comb) +  # Set custom colors
  theme_minimal()
