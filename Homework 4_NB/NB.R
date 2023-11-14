# Load necessary libraries
library(e1071)
library(caret)
library(ROCR)

# Load the dataset
dataset <- read.csv('breast-cancer-wisconsin.csv', header = TRUE)

# Display the first few rows of the dataset
head(dataset)

# Summary of statistics (min, max, mean)
summary <- summary(dataset)
print(summary)

# Replace '?' with NA
dataset[dataset == "?"] <- NA

# Drop rows with missing values
dataset <- na.omit(dataset)

# Split the dataset into training and testing sets
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(dataset$diagnosis, p = 0.7, 
                                   list = FALSE, 
                                   times = 1)
trainingData <- dataset[trainIndex, ]
testingData <- dataset[-trainIndex, ]

# Create the Naïve Bayes model
nb_model <- naiveBayes(diagnosis ~ ., data = trainingData)

# Make predictions on the testing data
Y_pred <- predict(nb_model, newdata = testingData)

# Evaluate the model
accuracy <- sum(Y_pred == testingData$diagnosis) / length(testingData$diagnosis)
conf_matrix <- table(predicted = Y_pred, actual = testingData$diagnosis)

# Display accuracy and confusion matrix
print(paste("Accuracy:", accuracy))
print("Confusion Matrix:")
print(conf_matrix)

# Create a ROC curve
prediction_probabilities <- predict(nb_model, newdata = testingData, type = "raw")
prediction_scores <- as.numeric(prediction_probabilities[, "Malignant"])
actual_values <- ifelse(testingData$diagnosis == "Malignant", 1, 0)
prediction_obj <- prediction(prediction_scores, actual_values)
performance <- performance(prediction_obj, "tpr", "fpr")

# Plot the ROC curve
plot(performance, main = "ROC Curve for Naïve Bayes Model")
