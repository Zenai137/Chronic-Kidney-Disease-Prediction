#------------------------------------------------------------------------------#
#   Predicting Chronic Kidney Disease (Machine Learning)
#   Logistic Regression, k-Nearest Neighbors, and k-Means Clustering
#------------------------------------------------------------------------------#

#set Working Directory
getwd()
setwd("C:/Users/medma/Documents/MS/Capstone/rcode")

#load library
library(caTools)

#ingest the data
ckd <- read.csv("../output/kidney_disease_clean.csv", header = T)
#view data
head(ckd)

#inspect the data
str(ckd)

#convert categorical variables to factors
ckd$specific_gravity <- as.factor(ckd$specific_gravity)
ckd$albumin <- as.factor(ckd$albumin)
ckd$hypertension <- as.factor(ckd$hypertension)
ckd$classification <- as.factor(ckd$classification)

#--------------------Class Bias--------------------#
#check the balance of the response variable
table(ckd$classification)
#class is imbalanced, so sample the
#observations in approximately equal
#proportions to get better models

#--------------------Create Training and Test Samples--------------------#
#create the training data
ckd_ones <- ckd[which(ckd$classification ==1),]
ckd_zeros <- ckd[which(ckd$classification ==0),]
#set.seed for repeatability of samples
set.seed(100)
#1's for training
ckd_ones_training_rows <- sample(1:nrow(ckd_ones), 0.48*nrow(ckd_ones))
#0's for training (zeros and ones must be equal)
ckd_zeros_training_rows <- sample(1:nrow(ckd_zeros), 0.48*nrow(ckd_ones))

training_ones <- ckd_ones[ckd_ones_training_rows, ]  
training_zeros <- ckd_zeros[ckd_zeros_training_rows, ]
#row bind the 1's and 0's
trainingData <- rbind(training_ones, training_zeros) 

# create Test Data
test_ones <- ckd_ones[-ckd_ones_training_rows, ]
test_zeros <- ckd_zeros[-ckd_zeros_training_rows, ]
#row bind the 1's and 0's
testData <- rbind(test_ones, test_zeros)

# create labels
cl <- trainingData[1:240,9]
test_labels <- testData[1:160,9]

#--------------------Build Logit Model--------------------#
logitMod <- glm(classification ~ ., data=trainingData, 
                family=binomial(link="logit"), maxit=100)

#predicted scores
predicted <- predict(logitMod, testData, type="response")

model_pred_class = rep("0", 160)
model_pred_class[predicted > 0.5] = "1"

#model diagnostics
summary(logitMod)
#high p values because IVs are uncorrelated; correlation is a restriction
#of linear relationships, and its independence that is crucial to logistical 
#probability

#stepwise backward elimination
logitMod = step(logitMod, direction = "backward")


#model diagnostics for new model
summary(logitMod)

#--------------------Logit Model Evaluation--------------------#
#performance
library(performance)
r2_nagelkerke(logitMod)

#misclassification error
library(InformationValue)
misClassError(testData$classification, predicted)

#ROC plot
plotROC(testData$classification, predicted)

#goodness-of-fit
with(logitMod, pchisq(null.deviance-deviance, df.null-df.residual, lower.tail = F))

#confusion matrix
#actuals = rows; predicted = columns
library(caret)
library(e1071)
table(actual = testData$classification, prediction = model_pred_class)

#Current Classification Accuracy(CCR)
# accuracy = (TP+TN)/(TP+TN+FP+FN)
(130 + 30) / (130 + 30 + 0 + 8)

#Sensitivity and Specificity (correct 1s and correct 0s)
sensitivity(testData$classification, predicted)
specificity(testData$classification, predicted)

#--------------------Logit Model Visualization--------------------#
logitMod2 <- glm(classification ~ ., data=ckd, 
                family=binomial(link="logit"), maxit=100)

predicted.data <- data.frame(probability.of.ckd=logitMod2$fitted.values, 
                             ckd=ckd$classification)

predicted.data <- predicted.data[order(predicted.data$probability.of.ckd, decreasing = FALSE),]
predicted.data$rank <- 1:nrow(predicted.data)

library(ggplot2)
library(cowplot)

ggplot(data=predicted.data, aes(x=rank, y=probability.of.ckd)) +
  geom_point(aes(color=ckd), alpha=1,shape=4, stroke=2) +
  xlab("Index") + ylab("Predicted Probability of CKD")

#--------------------Build kNN Model--------------------#
# Load the "class" AND "gmodels" packages to use them.
library(class)
library(gmodels)


# Create predictions using the training data with k=15 (sq root of 240 rounded)
pred <- knn(trainingData, testData, cl, k=15)


# CrossTable compares the actual to the predicted values
CrossTable(x=test_labels, y=pred, prop.chisq = FALSE)


#---------- IMPROVING THE MODEL ------------------------------------------
# several different values of k
pred <- knn(trainingData, testData, cl, k=10)
CrossTable(x=test_labels, y=pred, prop.chisq = FALSE)

pred <- knn(trainingData, testData, cl, k=20)
CrossTable(x=test_labels, y=pred, prop.chisq = FALSE)

plot(pred)

#--------------------kNN Model Evaluation--------------------#
# CCR = ((TP+TN)/(TP+TN+FP+FN))*100
(68 + 27) / (68 +27 + 3 + 62)
# Sensitivity = ((TP)/(TP+FN))*100
(68) / (68 + 62)
#Specificity = ((TN)/(TN+FP))*100
(27)/(27 + 3)

#--------------------Build k-Means Clustering Model--------------------#
#create a features object with the features columns
features <- ckd[1:8]

#create the clusters using kmeans; set.seed randomizes cluster centers
RNGversion("3.5.2")
set.seed(2345)
ckd_clusters <- kmeans(features, 3)

#--------------------k-Means Clustering Model Evaluation--------------------#
#look at the size of the clusters
ckd_clusters$size

# look at the cluster centers.
# each row shows that cluster's average value for each feature
ckd_clusters$centers

# apply the cluster IDs to the original data frame
ckd$cluster <- ckd_clusters$cluster

# look at the first ten records
ckd[1:10, c("cluster", "classification")]

# table of classification and clusters
table(ckd$classification, ckd_clusters$cluster)

# plot of clusters and attributes
plot(ckd[c("red_blood_cell_count", "white_blood_cell_count")], 
     col = ckd_clusters$cluster)
# plot of clusters and class; clusters are meshed together, so they aren't
# highly unique
plot(ckd[c("red_blood_cell_count", "white_blood_cell_count")], 
     col = ckd$classification)

#--------------------Build k-Means Clustering Model--------------------#
# load cluster package
library(cluster)
#create a features object with the features columns
features <- ckd[1:8]
set.seed(1)
#store the result
kResult <- pam(features, k = 3)

# table of classification and clusters
table(class = ckd$classification, cluster = kResult$cluster)

#create a summary
summary(kResult)

#--------------------k-Means Clustering Visualizations--------------------#
#create the cluster and silhouette plots
plot(kResult)

windows()

#store the cluster ids back into the original dataframe
ckd <- data.frame(ckd, kResult$clustering)

# look at the first ten records
ckd[1:10, c("clustering", "classification")]

# look at a cluster summary
summary(subset(ckd, kResult.clustering == 1))