# --- 1. Data Loading and Preparation ---

# Load the dataset.
# Using a fixed name is better for scripts. Make sure your file is named this.
data <- read.csv("spotify_data.csv")
# Or, to use the interactive chooser:
# data <- read.csv(file.choose())

# Display the full dataset in a new window
View(data)

# Select specific columns (features) by their index for modeling
# This selects columns 12, 13, 15, and 17 through 23
data1 <- data[, c(12, 13, 15, c(17:23))]

# View the selected subset of data
View(data1)

# Define a min-max normalization function (scales all data between 0 and 1)
d2 <- function(x) {
  ((x - min(x)) / (max(x) - min(x)))
}

# Apply the normalization function 'd2' to every column in 'data1'
d3 <- as.data.frame(lapply(data1, d2))

# View the normalized data
View(d3)

# Create the target variable 'popularity_category' based on the *original* data
# Using 'data$track_popularity' instead of 'a$track_popularity'
d3$popularity_category <- ifelse(data$track_popularity <= 50, "Low",
                                 ifelse(data$track_popularity <= 70, "Medium", "High")
)

# Convert the new category column to a factor (required for classification models)
d3$popularity_category <- as.factor(d3$popularity_category)

# View the final preprocessed data with the target variable
View(d3)


# --- 2. Train/Test Split ---

# Load the caTools library for splitting data
library(caTools)

# Set a seed for reproducibility, so you get the same "random" split every time
set.seed(123)

# Create a split, 70% for training.
# Splitting based on the target variable 'd3$popularity_category' helps maintain
# the same proportion of "Low", "Medium", and "High" in both train and test sets.
split_d <- sample.split(d3$popularity_category, SplitRatio = 0.70)

# Create the training set (where split_d is TRUE)
split_d_train <- subset(d3, split_d == TRUE)

# Create the testing set (where split_d is FALSE)
split_d_test <- subset(d3, split_d == FALSE)


# --- 3. Model Evaluation Function ---

# Define a function to calculate performance metrics from a confusion matrix (cm)
f1 <- function(cm) {
  # Accuracy: (Correct predictions) / (Total predictions)
  accuracy <- sum(diag(cm)) / sum(cm) * 100
  
  # Precision: (True Positives) / (All *predicted* as positive) - calculated per class
  precision <- diag(cm) / rowSums(cm) * 100
  
  # Recall: (True Positives) / (All *actual* positives) - calculated per class
  recall <- diag(cm) / colSums(cm) * 100
  
  # F1 Score: Harmonic mean of precision and recall - calculated per class
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Error Rate: 100% - Accuracy
  error_rate <- 100 - accuracy
  
  # Return a list of all metrics (using the *mean* for multi-class precision, recall, f1)
  list(
    accuracy = accuracy,
    precision = mean(precision, na.rm = TRUE),
    recall = mean(recall, na.rm = TRUE),
    f1_score = mean(f1_score, na.rm = TRUE),
    error_rate = error_rate
  )
}


# --- 4. Model Training and Evaluation ---

# Model 1: K-Nearest Neighbors (KNN)
library(class) # Load the 'class' library for KNN

# Predict using KNN (k=5)
# Assumes columns 1-10 are features and 11 is the target
y_pred <- knn(split_d_train[, c(1:10)], split_d_test[, c(1:10)], split_d_train[, 11], 5)

# Create the confusion matrix by comparing predicted vs. actual
cm <- table(y_pred, split_d_test[, 11])

# Calculate metrics using the custom function
knn1 <- f1(cm)

# Print KNN results
knn1

# Model 2: Naive Bayes
library(e1071) # Load the 'e1071' library

# Train the Naive Bayes model (using formula: target ~ all other features)
nb <- naiveBayes(popularity_category ~ ., split_d_train)

# Make predictions on the test set
y_pred1 <- predict(nb, split_d_test)

# Create the confusion matrix
cm1 <- table(y_pred1, split_d_test$popularity_category)

# Calculate metrics
nb1 <- f1(cm1)

# Print Naive Bayes results
nb1

# Model 3: Decision Tree
library(rpart) # Load the 'rpart' library

# Train the Decision Tree model
dt <- rpart(popularity_category ~ ., split_d_train, method = "class")

# Make predictions (type="class" returns the predicted category)
# Using 'dt' (the model object) instead of 'tree'
y_pred2 <- predict(dt, split_d_test, type = "class")

# Create the confusion matrix
cm2 <- table(y_pred2, split_d_test$popularity_category)

# Calculate metrics
dt1 <- f1(cm2)

# Print Decision Tree results
dt1

# Model 4: Neural Network
library(neuralnet) # Load the 'neuralnet' library

# Train the neural network
# Note: This is complex. The formula method with a factor target is tricky.
# We'll use the formula and assume it creates dummy variables internally.
nn <- neuralnet(popularity_category ~ ., split_d_train, linear.output = FALSE, threshold = 0.01)

# Make predictions (returns probabilities for each class: Low, Medium, High)
y_pred3_props <- predict(nn, split_d_test)

# Find the class with the highest probability (returns an index: 1, 2, or 3)
aa <- apply(y_pred3_props, 1, which.max)

# Convert the predicted index back to the class label (e.g., 1 -> "High", 2 -> "Low")
# This is a crucial step to match the factor levels.
predicted_levels <- levels(split_d_train$popularity_category)[aa]

# Create the confusion matrix using the *labels*, not the index
cm3 <- table(predicted_levels, split_d_test$popularity_category)

# Calculate metrics
nn1 <- f1(cm3)

# Print Neural Network results
nn1


# --- 5. Results Comparison and Plotting ---

# Load ggplot2 for plotting
library(ggplot2)

# Combine all results into a single data frame
results <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "Neural Network"),
  Accuracy = c(knn1$accuracy, nb1$accuracy, dt1$accuracy, nn1$accuracy),
  Precision = c(knn1$precision, nb1$precision, dt1$precision, nn1$precision),
  Recall = c(knn1$recall, nb1$recall, dt1$recall, nn1$recall),
  F1_Score = c(knn1$f1_score, nb1$f1_score, dt1$f1_score, nn1$f1_score),
  Error_Rate = c(knn1$error_rate, nb1$error_rate, dt1$error_rate, nn1$error_rate)
)

# Print the results table to the console
print(results)

# Define a function to create a bar plot for a given metric
# Renamed to 'plot_metric' to avoid overriding the base 'plot' function
plot_metric <- function(x) {
  # Use ggplot, .data[[x]] allows using the string 'x' as a column name
  ggplot(results, aes(x = Model, y = .data[[x]], fill = Model)) +
    # Create a bar chart
    geom_bar(stat = "identity") +
    # Add text labels (the metric value) above the bars
    geom_text(aes(label = round(.data[[x]], 1)), vjust = -0.25, color = "black") +
    # Set dynamic labels for the title and axes
    labs(title = paste("Model Comparison -", x), y = paste(x, "(%)"), x = "Model") +
    # Use a clean, minimal theme
    theme_minimal()
}

# Generate and display the plot for Accuracy
plot_metric("Accuracy")

# Generate and display the plot for Precision
plot_metric("Precision")

# Generate and display the plot for Recall
plot_metric("Recall")

# Generate and display the plot for F1_Score
plot_metric("F1_Score")

# Generate and display the plot for Error_Rate
plot_metric("Error_Rate")