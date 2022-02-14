#------------------------------------------------------------------------------#
#   Predicting Chronic Kidney Disease (Machine Learning)
#   Decision Tree and Random Forest
#------------------------------------------------------------------------------#

#set Working Directory
getwd()
setwd("**/*****/*******/******/*****/******/****")

#load libraries
library(caret) #train and datapartition
library(smotefamily) #ADASYN
library(rpart) #decision tree
library(randomForest)# random forest
library(ROSE) # oversampling and undersampling

#visualizations
library(ggplot2) #bar plots
library(ggthemes)
library(ggpubr) #multiple plots on one graph
library(imbalance) #plot comparisons
library(rpart.plot) #decision tree visuals
library(rfviz) #random forest visuals

#ingest the data
ckd <- read.csv("../output/ckd.csv", header = T)
#view data
head(ckd)
#change haemoglobin to hemoglobin
colnames(ckd)[4] <- "hemoglobin"
#inspect the data
str(ckd)

#convert the response variable to a factor
ckd$classification <- as.factor(ckd$classification)

#inspect the data again
str(ckd)

#--------------------Class Balance--------------------#
#check the balance of the response variable
#class is imbalanced, so sample the
#observations in approximately equal
#proportions to get better models

df <- ggplot(data=ckd, aes(x=classification, 
  fill=classification)) + geom_bar(show.legend=FALSE) + 
  geom_text(stat="count", size=7, aes(label=..count..), vjust=3) + 
  ggtitle("Classification") + labs(y="Count", x="Classification") + 
  scale_fill_manual(values = c("darkred",
  "grey19")) + theme(plot.title = element_text(face="bold", size=14))

ggarrange(df, ncol=1, nrow=1, align = "v", common.legend=FALSE,
          legend="bottom")

#--------------------Create Training and Test Datasets--------------------#
#stratified sampling by class
#set.seed for repeatability of samples
set.seed(1545)
train <- createDataPartition(ckd$classification,
                             p = 0.7, # % of data going to training
                             times = 1,
                             list = F)
trainingData <- ckd[ train,]
testData       <- ckd[-train,]

#check that all levels of categorical variables are present in both datasets
levels(factor(trainingData$specific_gravity))
levels(factor(trainingData$albumin))
levels(factor(trainingData$hypertension))

levels(factor(testData$specific_gravity))
levels(factor(testData$albumin))
levels(factor(testData$hypertension))


#--------------------Class Balance Re-Check--------------------#
df <- ggplot(data=ckd, aes(x=classification, 
  fill=classification)) + geom_bar(show.legend=FALSE) + 
  geom_text(stat="count", size=7, aes(label=..count..), vjust=3) + 
  ggtitle("Classification") + labs(y="Count", x="Classification") + 
  scale_fill_manual(values = c("darkred","grey19")) + 
  theme(plot.title = element_text(face="bold", size=14))

trs <- ggplot(data=trainingData, aes(x=classification, 
  fill=classification)) + geom_bar(show.legend=FALSE) + 
  geom_text(stat="count", size=7, aes(label=..count..), vjust=3) + 
  ggtitle("Original Training Sample") + labs(y="Count", x="Classification") + 
  scale_fill_manual(values = c("darkred", "grey19")) + 
  theme(plot.title = element_text(face="bold", size=14)) + ylim(0,250)

tes <- ggplot(data=testData, aes(x=classification, 
  fill=classification)) + geom_bar(show.legend=FALSE) + 
  geom_text(stat="count", size=7, aes(label=..count..), vjust=3) + 
  ggtitle("Original Test Sample") + labs(y="Count", x="Classification") + 
  scale_fill_manual(values = c("darkred", "grey19")) + 
  theme(plot.title = element_text(face="bold", size=14)) + ylim(0,250)

ggarrange(trs, tes, ncol=2, nrow=1, align = "v", common.legend=FALSE,
          legend="bottom")

#--------------------Over Sampling--------------------#
trainingData.os <- ovun.sample(classification ~ ., data=trainingData,
                               method="over", N=350)$data
table(trainingData.os$classification)

prop.table(table(trainingData.os$classification))

#--------------------Under Sampling--------------------#
trainingData.us <- ovun.sample(classification ~ ., data=trainingData,
                               method="under", N=210)$data
table(trainingData.us$classification)

prop.table(table(trainingData.us$classification))

#--------------------SMOTE Oversampling--------------------#
trainingData.smote <- SMOTE(trainingData[,-9],trainingData$classification,
                          K=5, dup_size = 1)$data
str(trainingData.smote)

trainingData.smote$class <- as.factor(trainingData.smote$class)

table(trainingData.smote$class)
prop.table(table(trainingData.smote$class))

#--------------------ADASYN Oversampling--------------------#
trainingData.adas <- ADAS(trainingData[,-9],trainingData$classification,
                            K=5)$data
str(trainingData.adas)

trainingData.adas$class <- as.factor(trainingData.adas$class)

table(trainingData.adas$class)
prop.table(table(trainingData.adas$class))

#--------------------Classification Barplot Comparisons--------------------#
og <- ggplot(data=trainingData, aes(x=classification, 
  fill=classification)) + geom_bar(show.legend=FALSE) + 
  geom_text(stat="count", size=7, aes(label=..count..), vjust=3) + 
  ggtitle("Classification\nOriginal Training Sample") + 
  labs(y="Count", x="Classification") + 
  scale_fill_manual(values = c("darkred","grey19")) + 
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,250)

os <- ggplot(data=trainingData.os, aes(x=classification, fill=classification)) + 
  geom_bar(show.legend=FALSE) + geom_text(stat="count", size=7, 
  aes(label=..count..), vjust=3) +ggtitle("\nOver-sampling") + labs(y="Count", 
  x="Classification") + scale_fill_manual(values = 
  c("grey19", "darkred")) + guides(fill = 
  guide_legend(reverse = FALSE)) + theme(plot.title = element_text(face="bold", 
  size=14)) + scale_x_discrete(limits=rev) +
  ylim(0,250)

us <- ggplot(data=trainingData.us, aes(x=classification, fill=classification)) + 
  geom_bar(show.legend=FALSE) + geom_text(stat="count", size=7, 
  aes(label=..count..), vjust=3) +ggtitle("\nUnder-sampling") + labs(y="Count", 
  x="Classification") + scale_fill_manual(values = 
  c("grey19", "darkred")) + guides(fill = 
  guide_legend(reverse = FALSE)) + theme(plot.title = element_text(face="bold", 
  size=14)) + scale_x_discrete(limits=rev) +
  ylim(0,250)

smote <- ggplot(data=trainingData.smote, aes(x=class, fill=class)) + 
  geom_bar(show.legend=FALSE) + geom_text(stat="count", size=7, 
  aes(label=..count..), vjust=3) +ggtitle("\nSMOTE Over-sampling") + 
  labs(y="Count", x="Classification") +
  scale_fill_manual(values = c("darkred","grey19")) +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,250)

adas <- ggplot(data=trainingData.adas, aes(x=class, fill=class)) + 
  geom_bar(show.legend=FALSE) + geom_text(stat="count", size=7, 
  aes(label=..count..), vjust=3) +ggtitle("\nADAS Over-sampling") + 
  labs(y="Count", x="Classification") +
  scale_fill_manual(values = c("darkred","grey19")) +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,250)

ggarrange(os, us, smote, adas, ncol=3, nrow=2, align = "v", common.legend=FALSE,
  legend="bottom")

#--------------------Numerical Scatterplot Comparisons--------------------#
#function for mean labels
mean.n <- function(x){
  return(c(y = median(x)*1.12, label = round(mean(x),2)))
}
sd.n <- function(x){
  return(c(y = median(x)*.9, label = round(sd(x),2)))
}

#store the means, standard deviations, and medians for serum creatinine
mean_og <- aggregate(serum_creatinine ~ classification, trainingData, mean)
sd_og <- aggregate(serum_creatinine ~ classification, trainingData, sd)
med_og <- aggregate(serum_creatinine ~ classification, trainingData, median)

mean_os <- aggregate(serum_creatinine ~ classification, trainingData.os, mean)
sd_os <- aggregate(serum_creatinine ~ classification, trainingData.os, sd)
med_os <- aggregate(serum_creatinine ~ classification, trainingData.os, median)

mean_us <- aggregate(serum_creatinine ~ classification, trainingData.us, mean)
sd_us <- aggregate(serum_creatinine ~ classification, trainingData.us, sd)
med_us <- aggregate(serum_creatinine ~ classification, trainingData.us, median)

mean_smote <- aggregate(serum_creatinine ~ class, trainingData.smote, mean)
sd_smote <- aggregate(serum_creatinine ~ class, trainingData.smote, sd)
med_smote <- aggregate(serum_creatinine ~ class, trainingData.smote, median)

mean_adas <- aggregate(serum_creatinine ~ class, trainingData.adas, mean)
sd_adas <- aggregate(serum_creatinine ~ class, trainingData.adas, sd)
med_adas <- aggregate(serum_creatinine ~ class, trainingData.adas, median)


#serum creatinine
og <- ggplot(data=trainingData, aes(x=classification, 
  y=serum_creatinine, fill=classification)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) +
  ggtitle("Serum Creatinine\nOriginal Training Sample") + 
  labs(x="Classification", y="Milligrams per Deciliter\n (mg/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,2) +
  scale_fill_manual(values = c("orange3", "turquoise3")) +
  stat_summary(geom = "text", fun = quantile,
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

os <- ggplot(data=trainingData.os, aes(x=classification, 
  y=serum_creatinine, fill=classification)) +  stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) +
  ggtitle("\nOver-sampling") + 
  labs(x="Classification", y="Milligrams per Deciliter\n (mg/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,2) +
  scale_fill_manual(values = c("turquoise3", "orange3")) + 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) +
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

us <- ggplot(data=trainingData.us, aes(x=classification, 
  y=serum_creatinine, fill=classification)) +  stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) +
  ggtitle("\nUnder-sampling") + 
  labs(x="Classification", y="Milligrams per Deciliter\n (mg/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,2) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

smote <- ggplot(data=trainingData.smote, aes(x=class, 
  y=serum_creatinine, fill=class)) +  stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) +
  ggtitle("\nSMOTE Over-sampling") + 
  labs(x="Classification", y="Milligrams per Deciliter\n (mg/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,2) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

adas <- ggplot(data=trainingData.adas, aes(x=class, 
  y=serum_creatinine, fill=class)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nADAS Over-sampling") + 
  labs(x="Classification", y="Milligrams per Deciliter\n (mg/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  ylim(0,2) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", 
          common.legend=FALSE,
          legend="bottom")

#function for mean labels
mean.n <- function(x){
  return(c(y = median(x)*1.045, label = round(mean(x),2)))
}

sd.n <- function(x){
  return(c(y = median(x)*.97, label = round(sd(x),2)))
}

#hemoglobin
og <- ggplot(data=trainingData, aes(x=classification, 
  y=hemoglobin, fill=classification)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + ggtitle("Hemoglobin\nOriginal Training Sample") + 
  labs(x="Classification", y="Grams per Deciliter\n (g/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

os <- ggplot(data=trainingData.os, aes(x=classification, 
  y=hemoglobin, fill=classification)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + ggtitle("\nOver-sampling") + 
  labs(x="Classification", y="Grams per Deciliter\n (g/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

us <- ggplot(data=trainingData.us, aes(x=classification, 
  y=hemoglobin, fill=classification)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + ggtitle("\nUnder-sampling") + 
  labs(x="Classification", y="Grams per Deciliter\n (g/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

smote <- ggplot(data=trainingData.smote, aes(x=class, 
  y=hemoglobin, fill=class)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + ggtitle("\nSMOTE Over-sampling") + 
  labs(x="Classification", y="Grams per Deciliter\n (g/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

adas <- ggplot(data=trainingData.adas, aes(x=class, 
  y=hemoglobin, fill=class)) + stat_boxplot(geom = 'errorbar') +
  geom_boxplot(show.legend=FALSE) + ggtitle("\nADAS Over-sampling") + 
  labs(x="Classification", y="Grams per Deciliter\n (g/dL)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", 
          common.legend=FALSE,
          legend="bottom")

#function for mean labels
mean.n <- function(x){
  return(c(y = median(x)*1.048, label = round(mean(x),2)))
}

sd.n <- function(x){
  return(c(y = median(x)*.975, label = round(sd(x),2)))
}

#packed cell volume
og <- ggplot(data=trainingData, aes(classification, 
  y=packed_cell_volume, 
  fill=classification)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("Packed Cell Volume\nOriginal Training Sample") + 
  labs(x="Classification", y="Percentage\n (pct)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

os <- ggplot(data=trainingData.os, aes(classification, 
  y=packed_cell_volume, fill=classification)) + 
  stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + ggtitle("\nOver-sampling") + 
  labs(x="Classification", y="Percentage\n (pct)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

us <- ggplot(data=trainingData.us, aes(x=classification, 
  y=packed_cell_volume, fill=classification)) + 
  stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + ggtitle("\nUnder-sampling") + 
  labs(x="Classification", y="Percentage\n (pct)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

smote <- ggplot(data=trainingData.smote, aes(x=class, 
  y=packed_cell_volume, 
  fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nSMOTE Over-sampling") + 
  labs(x="Classification", y="Percentage\n (pct)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

adas <- ggplot(data=trainingData.adas, aes(x=class, 
  y=packed_cell_volume, 
  fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nADAS Over-sampling") + 
  labs(x="Classification", y="Percentage\n (pct)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", 
          common.legend=FALSE,
          legend="bottom")

#function for mean labels
mean.n <- function(x){
  return(c(y = median(x)*1.09, label = round(mean(x),2)))
}

sd.n <- function(x){
  return(c(y = median(x)*.95, label = round(sd(x),2)))
}

#white blood cell count
og <- ggplot(data=trainingData, aes(x=classification, 
  y=white_blood_cell_count, 
  fill=classification)) + stat_boxplot(geom = 'errorbar') + 
  geom_boxplot(show.legend=FALSE) + 
  ggtitle("White Blood Cell Count\nOriginal Training Sample") + 
  labs(x="Classification", y="Cells per Cubic Millimeter\n (cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

os <- ggplot(data=trainingData.os, aes(x=classification, 
  y=white_blood_cell_count, fill=classification)) + stat_boxplot(geom = 'errorbar') + 
  geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nOver-sampling") + 
  labs(x="Classification", y="Cells per Cubic Millimeter\n (cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

us <- ggplot(data=trainingData.us, aes(x=classification, 
  y=white_blood_cell_count, fill=classification)) + stat_boxplot(geom = 'errorbar') + 
  geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nUnder-sampling") + 
  labs(x="Classification", y="Cells per Cubic Millimeter\n (cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

smote <- ggplot(data=trainingData.smote, aes(x=class, 
  y=white_blood_cell_count, 
  fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nSMOTE Over-sampling") + 
  labs(x="Classification", y="Cells per Cubic Millimeter\n (cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

adas <- ggplot(data=trainingData.adas, aes(x=class, 
  y=white_blood_cell_count, 
  fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nADAS Over-sampling") + 
  labs(x="Classification", y="Cells per Cubic Millimeter\n (cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", 
          common.legend=FALSE,
          legend="bottom")

#function for mean labels
mean.n <- function(x){
  return(c(y = median(x)*1.06, label = round(mean(x),2)))
}

sd.n <- function(x){
  return(c(y = median(x)*.97, label = round(sd(x),2)))
}

#red blood cell count
og <- ggplot(data=trainingData, aes(x=classification, 
  y=red_blood_cell_count, fill=classification)) + 
  stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("Red Blood Cell Count\nOriginal Training Sample") + 
  labs(x="Classification", y="Millions of Cells per Cubic Millimeter\n (million cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

os <- ggplot(data=trainingData.os, aes(x=classification, 
  y=red_blood_cell_count, fill=classification)) + 
  stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nOver-sampling") + 
  labs(x="Classification", y="Millions of Cells per Cubic Millimeter\n (million cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3"))+ 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

us <- ggplot(data=trainingData.us, aes(x=classification, 
  y=red_blood_cell_count, fill=classification)) + 
  stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nUnder-sampling") + 
  labs(x="Classification", y="Millions of Cells per Cubic Millimeter\n (million cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("turquoise3", "orange3")) + 
  guides(fill = guide_legend(reverse = TRUE)) + scale_x_discrete(limits=rev) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

smote <- ggplot(data=trainingData.smote, aes(x=class, 
  y=red_blood_cell_count, fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nSMOTE Over-sampling") + 
  labs(x="Classification", y="Millions of Cells per Cubic Millimeter\n (million cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

adas <- ggplot(data=trainingData.adas, aes(x=class, 
  y=red_blood_cell_count, fill=class)) + stat_boxplot(geom = 'errorbar') + geom_boxplot(show.legend=FALSE) + 
  ggtitle("\nADAS Over-sampling") + 
  labs(x="Classification", y="Millions of Cells per Cubic Millimeter\n (million cells/cmm)") +
  theme(plot.title = element_text(face="bold", size=14)) +
  scale_fill_manual(values = c("orange3", "turquoise3")) + 
  stat_summary(geom = "text", fun = quantile, 
  aes(label=sprintf("%1.1f", ..y..)), position=position_nudge(x=0.5), size=3.5) +
  stat_summary(fun.data = mean.n, geom = "text", fun = mean, colour = "black", size=3) +
  stat_summary(fun.data = sd.n, geom = "text", fun = mean, colour = "black", size=3)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", 
          common.legend=FALSE,
          legend="bottom")


#--------------------Categorical Barplot Comparisons--------------------#
#convert the categorical variables to factors
trainingData$specific_gravity <- as.factor(trainingData$specific_gravity)
trainingData$albumin <- as.factor(trainingData$albumin)
trainingData$hypertension <- as.factor(trainingData$hypertension)

trainingData.os$specific_gravity <- as.factor(trainingData.os$specific_gravity)
trainingData.os$albumin <- as.factor(trainingData.os$albumin)
trainingData.os$hypertension <- as.factor(trainingData.os$hypertension)

trainingData.us$specific_gravity <- as.factor(trainingData.us$specific_gravity)
trainingData.us$albumin <- as.factor(trainingData.us$albumin)
trainingData.us$hypertension <- as.factor(trainingData.us$hypertension)

#list the levels for all categorical variables that were fed into SMOTE and ADAS
levels(factor(trainingData.smote$specific_gravity))
levels(factor(trainingData.smote$albumin))
levels(factor(trainingData.smote$hypertension))

levels(factor(trainingData.adas$specific_gravity))
levels(factor(trainingData.adas$albumin))
levels(factor(trainingData.adas$hypertension))

#store the new values for specific gravity in 1.02 or 1.025 based on the median 1.0225

trainingData.smote$specific_gravity[trainingData.smote$specific_gravity > 1.02 & 
                                      trainingData.smote$specific_gravity < 1.0225] <- 1.02
trainingData.smote$specific_gravity[trainingData.smote$specific_gravity > 1.0225] <- 1.025

trainingData.adas$specific_gravity[trainingData.adas$specific_gravity > 1.02 & 
                                      trainingData.adas$specific_gravity < 1.0225] <- 1.02
trainingData.adas$specific_gravity[trainingData.adas$specific_gravity > 1.0225] <- 1.025

trainingData.smote$specific_gravity <- as.factor(trainingData.smote$specific_gravity)
trainingData.smote$albumin <- as.factor(trainingData.smote$albumin)
trainingData.smote$hypertension <- as.factor(trainingData.smote$hypertension)

trainingData.adas$specific_gravity <- as.factor(trainingData.adas$specific_gravity)
trainingData.adas$albumin <- as.factor(trainingData.adas$albumin)
trainingData.adas$hypertension <- as.factor(trainingData.adas$hypertension)

#specific gravity
og <- ggplot(data=trainingData, aes(x=specific_gravity, 
  fill=classification)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.4) + 
  ggtitle("Specific Gravity\nOriginal Training Sample") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14),
        axis.text.x=element_text(angle=90)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,110)

os <- ggplot(data=trainingData.os, aes(x=specific_gravity, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.4) + 
  ggtitle("\nOver-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14),
        axis.text.x=element_text(angle=90)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4"))+ 
  guides(color = guide_legend(reverse = TRUE)) + ylim(0,110)

us <- ggplot(data=trainingData.us, aes(x=specific_gravity, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.4) + 
  ggtitle("\nUnder-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14),
        axis.text.x=element_text(angle=90)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4"))+ 
  guides(color = guide_legend(reverse = TRUE)) + ylim(0,110)

smote <- ggplot(data=trainingData.smote, aes(x=specific_gravity, 
  fill=class)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.4) + 
  ggtitle("\nSMOTE Over-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14),
        axis.text.x=element_text(angle=90)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,110)

adas <- ggplot(data=trainingData.adas, aes(x=specific_gravity, 
  fill=class)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.4) + 
  ggtitle("\nADAS Over-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14),
        axis.text.x=element_text(angle=90)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,110)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", common.legend=FALSE,
          legend="bottom")

#albumin
og <- ggplot(data=trainingData, aes(x=albumin, 
  fill=classification)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.7) + 
  ggtitle("Albumin\nOriginal Training Sample") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

os <- ggplot(data=trainingData.os, aes(x=albumin, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.7) + 
  ggtitle("\nOver-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4")) + ylim(0,250) 

us <- ggplot(data=trainingData.us, aes(x=albumin, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.7) + 
  ggtitle("\nUnder-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4"))+ ylim(0,250)

smote <- ggplot(data=trainingData.smote, aes(x=albumin, 
  fill=class, label=..count..)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.7) + 
  ggtitle("\nSMOTE Over-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

adas <- ggplot(data=trainingData.adas, aes(x=albumin, 
  fill=class, label=..count..)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.7) + 
  ggtitle("ADAS Over-sampling") + labs(y="Count",
  x="") + theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", common.legend=FALSE,
          legend="bottom")

#hypertension
og <- ggplot(data=trainingData, aes(x=hypertension, 
  fill=classification)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.9) + 
  ggtitle("Hypertension\nOriginal Training Sample") + labs(y="Count",
  x="") + 
  theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

os <- ggplot(data=trainingData.os, aes(x=hypertension, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.9) + 
  ggtitle("\nOver-sampling") + labs(y="Count",
  x="") + 
  theme(plot.title = element_text(face="bold", size=14)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4")) + ylim(0,250) 

us <- ggplot(data=trainingData.us, aes(x=hypertension, 
  fill=classification)) + geom_bar(position=position_dodge2(reverse=TRUE)) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.9) + 
  ggtitle("\nUnder-sampling") + labs(y="Count",
  x="") + 
  theme(plot.title = element_text(face="bold", size=14)) + 
  guides(fill = guide_legend(reverse = TRUE)) + 
  scale_fill_manual(values = c("seagreen4", "chocolate4")) + ylim(0,250)

smote <- ggplot(data=trainingData.smote, aes(x=hypertension, 
  fill=class)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.9) + 
  ggtitle("\nADAS Over-sampling") + labs(y="Count",
  x="") +
  theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

adas <- ggplot(data=trainingData.adas, aes(x=hypertension, 
  fill=class)) + geom_bar(position=position_dodge2()) + 
  geom_text(stat="count", size=4, aes(label=..count..), vjust=.9) + 
  ggtitle("\nADAS Over-sampling") + labs(y="Count",
  x="") +
  theme(plot.title = element_text(face="bold", size=14)) + 
  scale_fill_manual(values = c("chocolate4", "seagreen4")) + ylim(0,250)

ggarrange(og, os, us, smote, adas, ncol=3, nrow=2, align = "v", common.legend=FALSE,
          legend="bottom")

#minor cleaning

levels(trainingData$albumin)
levels(factor(testData$albumin))

testData <- testData[!(testData$albumin==5),]
levels(factor(testData$albumin))

#convert training data classification labels
levels(trainingData$classification) <- c("NotCKD","CKD")
levels(trainingData.os$classification) <- c("CKD","NotCKD")
levels(trainingData.us$classification) <- c("CKD","NotCKD")
levels(trainingData.smote$class) <- c("NotCKD","CKD")
levels(trainingData.adas$class) <- c("NotCKD","CKD")

#convert the categorical variables to factors for test data
testData$specific_gravity <- as.factor(testData$specific_gravity)
testData$albumin <- as.factor(testData$albumin)
testData$hypertension <- as.factor(testData$hypertension)

#convert test data classification labels
levels(testData$classification) <- c("NotCKD","CKD")

str(trainingData)
str(testData)

################################################################################
#-------------Classification Model Training: Original Training Data------------#
################################################################################
#A. Global options that we will use across all of our trained models

ctrl <- trainControl(method = 'cv',
                     number = 10,
                     savePredictions = TRUE,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)
?trainControl
#B. Decision Tree: original training data
dt_orig <- train(classification ~ .,
                 data = trainingData,
                 method = "rpart",
                 trControl = ctrl,
                 metric = "ROC")

#C. Random Forest: original training data
rf_orig <- train(classification ~ .,
                 data = trainingData,
                 method = "rf",
                 trControl = ctrl,
                 metric = "ROC")

#-------------Classification Model Testing: Original Training Data------------#
#- - - - -Decision Tree Model- - - - -#
#A. Decision Tree Model predictions
dt_orig_pred <- predict(dt_orig,testData,type="prob")

#B. Decision Tree Class Probabilities
dt_orig_test <- factor(ifelse(dt_orig_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Decision Tree Precision/Recall/F1 Measure
accuracy_dtOrig <- mean(testData$classification == dt_orig_test)
precision_dtOrig <- posPredValue(dt_orig_test, testData$classification, positive="CKD")
recall_dtOrig <- sensitivity(dt_orig_test, testData$classification, positive="CKD")
F1_dtOrig <- (2*precision_dtOrig*recall_dtOrig)/(recall_dtOrig + precision_dtOrig)

#- - - - -Random Forest Model- - - - -#
#A. Random Forest Model predictions
rf_orig_pred <- predict(rf_orig,testData,type="prob")

#B. Random Forest Class Probabilities
rf_orig_test <- factor(ifelse(rf_orig_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Random Forest Precision/Recall/F1 Measure
accuracy_rfOrig <- mean(testData$classification == rf_orig_test)
precision_rfOrig <- posPredValue(rf_orig_test, testData$class, positive="CKD")
recall_rfOrig <- sensitivity(rf_orig_test, testData$class, positive="CKD")
F1_rfOrig <- (2*precision_rfOrig*recall_rfOrig)/(recall_rfOrig + precision_rfOrig)

################################################################################
#-------------Classification Model Training: Over-sampling Data------------#
################################################################################
#A. Decision Tree: over-sampling data

dt_os <- train(classification ~ .,
               data = trainingData.os,
                 method = "rpart",
                 trControl = ctrl,
                 metric = "ROC")

#B. Random Forest: over-sampling data

rf_os <- train(classification ~ .,
               data = trainingData.os,
                 method = "rf",
                 trControl = ctrl,
                 metric = "ROC")

#-------------Classification Model Testing: Over-sampling Data------------#
#- - - - -Decision Tree Model- - - - -#
#A. Decision Tree Model predictions
dt_os_pred <- predict(dt_os,testData,type="prob")

#B. Decision Tree Class Probabilities
dt_os_test <- factor(ifelse(dt_os_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Decision Tree Precision/Recall/F1 Measure
accuracy_dtOs <- mean(testData$classification == dt_os_test)
precision_dtOs <- posPredValue(dt_os_test, testData$classification, positive="CKD")
recall_dtOs <- sensitivity(dt_os_test, testData$classification, positive="CKD")
F1_dtOs <- (2*precision_dtOs*recall_dtOs)/(recall_dtOs + precision_dtOs)

#- - - - -Random Forest Model- - - - -#
#A. Random Forest Model predictions
rf_os_pred <- predict(rf_os,testData,type="prob")

#B. Random Forest Class Probabilities
rf_os_test <- factor(ifelse(rf_os_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Random Forest Precision/Recall/F1 Measure
accuracy_rfOs <- mean(testData$classification == rf_os_test)
precision_rfOs <- posPredValue(rf_os_test, testData$classification, positive="CKD")
recall_rfOs <- sensitivity(rf_os_test, testData$classification, positive="CKD")
F1_rfOs <- (2*precision_rfOs*recall_rfOs)/(recall_rfOs + precision_rfOs)

################################################################################
#-------------Classification Model Training: Under-sampling Data------------#
################################################################################
#A. Decision Tree: under-sampling data

dt_us <- train(classification ~ .,
               data = trainingData.us,
               method = "rpart",
               trControl = ctrl,
               metric = "ROC")

#B. Random Forest: under-sampling data

rf_us <- train(classification ~ .,
               data = trainingData.us,
               method = "rf",
               trControl = ctrl,
               metric = "ROC")

#-------------Classification Model Testing: Under-sampling Data------------#
#- - - - -Decision Tree Model- - - - -#
#A. Decision Tree Model predictions
dt_us_pred <- predict(dt_us,testData,type="prob")

#B. Decision Tree Class Probabilities
dt_us_test <- factor(ifelse(dt_us_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Decision Tree Precision/Recall/F1 Measure
accuracy_dtUs <- mean(testData$classification == dt_us_test)
precision_dtUs <- posPredValue(dt_us_test, testData$classification, positive="CKD")
recall_dtUs <- sensitivity(dt_us_test, testData$classification, positive="CKD")
F1_dtUs <- (2*precision_dtUs*recall_dtUs)/(recall_dtUs + precision_dtUs)

#- - - - -Random Forest Model- - - - -#
#A. Random Forest Model predictions
rf_us_pred <- predict(rf_us,testData,type="prob")

#B. Random Forest Class Probabilities
rf_us_test <- factor(ifelse(rf_us_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Random Forest Precision/Recall/F1 Measure
accuracy_rfUs <- mean(testData$classification == rf_us_test)
precision_rfUs <- posPredValue(rf_us_test, testData$classification, positive="CKD")
recall_rfUs <- sensitivity(rf_us_test, testData$classification, positive="CKD")
F1_rfUs <- (2*precision_rfUs*recall_rfUs)/(recall_rfUs + precision_rfUs)

################################################################################
#-------------Classification Model Training: SMOTE Data------------#
################################################################################
#A. Decision Tree: smote data

dt_smote <- train(class ~ .,
                  data = trainingData.smote,
               method = "rpart",
               trControl = ctrl,
               metric = "ROC")

#D. Random Forest: smote data

rf_smote <- train(class ~ .,
                  data = trainingData.smote,
               method = "rf",
               trControl = ctrl,
               metric = "ROC")

#-------------Classification Model Testing: SMOTE Data------------#
#- - - - -Decision Tree Model- - - - -#
#A. Decision Tree Model predictions
dt_smote_pred <- predict(dt_smote,testData,type="prob")

#B. Decision Tree Class Probabilities
dt_smote_test <- factor(ifelse(dt_smote_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Decision Tree Precision/Recall/F1 Measure
accuracy_dtsmote <- mean(testData$classification == dt_smote_test)
precision_dtsmote <- posPredValue(dt_smote_test, testData$classification, positive="CKD")
recall_dtsmote <- sensitivity(dt_smote_test, testData$classification, positive="CKD")
F1_dtsmote <- (2*precision_dtsmote*recall_dtsmote)/(recall_dtsmote + precision_dtsmote)

#- - - - -Random Forest Model- - - - -#
#A. Random Forest Model predictions
rf_smote_pred <- predict(rf_smote,testData,type="prob")

#B. Random Forest Class Probabilities
rf_smote_test <- factor(ifelse(rf_smote_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Random Forest Precision/Recall/F1 Measure
accuracy_rfsmote <- mean(testData$classification == rf_smote_test)
precision_rfsmote <- posPredValue(rf_smote_test, testData$classification, positive="CKD")
recall_rfsmote <- sensitivity(rf_smote_test, testData$classification, positive="CKD")
F1_rfsmote <- (2*precision_rfsmote*recall_rfsmote)/(recall_rfsmote + precision_rfsmote)

################################################################################
#-------------Classification Model Training: ADAS Data------------#
################################################################################
#A. Decision Tree: adas data

dt_adas <- train(class ~ .,
                 data = trainingData.adas,
                  method = "rpart",
                  trControl = ctrl,
                  metric = "ROC")

#B. Random Forest: adas data

rf_adas <- train(class ~ .,
                 data = trainingData.adas,
                  method = "rf",
                  trControl = ctrl,
                  metric = "ROC")

#-------------Classification Model Testing: ADAS Data------------#
#- - - - -Decision Tree Model- - - - -#
#A. Decision Tree Model predictions
dt_adas_pred <- predict(dt_adas,testData,type="prob")

#B. Decision Tree Class Probabilities
dt_adas_test <- factor(ifelse(dt_adas_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Decision Tree Precision/Recall/F1 Measure
accuracy_dtadas <- mean(testData$classification == dt_adas_test)
precision_dtadas <- posPredValue(dt_adas_test, testData$classification, positive="CKD")
recall_dtadas <- sensitivity(dt_adas_test, testData$classification, positive="CKD")
F1_dtadas <- (2*precision_dtadas*recall_dtadas)/(recall_dtadas + precision_dtadas)

#- - - - -Random Forest Model- - - - -#
#A. Random Forest Model predictions
rf_adas_pred <- predict(rf_adas,testData,type="prob")

#B. Random Forest Class Probabilities
rf_adas_test <- factor(ifelse(rf_adas_pred$CKD > 0.5,"CKD","NotCKD"))

#C. Random Forest Precision/Recall/F1 Measure
accuracy_rfadas <- mean(testData$classification == rf_adas_test)
precision_rfadas <- posPredValue(rf_adas_test, testData$classification, positive="CKD")
recall_rfadas <- sensitivity(rf_adas_test, testData$classification, positive="CKD")
F1_rfadas <- (2*precision_rfadas*recall_rfadas)/(recall_rfadas + precision_rfadas)

################################################################################
#-------------Cross Validation Models: Decision Tree------------#
################################################################################
#D. Unfold the k-Folds for each model
#--------------Decision Tree Folds--------------#
#orig
dt_kfold_pred <- dt_orig$pred
dt_kfold_pred$equal <- ifelse(dt_kfold_pred$pred == dt_kfold_pred$obs, 1,0)

#over-sampling
dtos_kfold_pred <- dt_os$pred
dtos_kfold_pred$equal <- ifelse(dtos_kfold_pred$pred == dtos_kfold_pred$obs, 1,0)

#under-sampling
dtus_kfold_pred <- dt_us$pred
dtus_kfold_pred$equal <- ifelse(dtus_kfold_pred$pred == dtus_kfold_pred$obs, 1,0)

#smote
dtsmote_kfold_pred <- dt_smote$pred
dtsmote_kfold_pred$equal <- ifelse(dtsmote_kfold_pred$pred == dtsmote_kfold_pred$obs, 1,0)

#adas
dtadas_kfold_pred <- dt_adas$pred
dtadas_kfold_pred$equal <- ifelse(dtadas_kfold_pred$pred == dtadas_kfold_pred$obs, 1,0)

#--------------Decision Tree Fold Tables--------------#
#orig
dt_eachfold <- dt_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
dt_eachfold
#over-sampling
dtos_eachfold <- dtos_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
dtos_eachfold
#under-sampling
dtus_eachfold <- dtus_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
dtus_eachfold
#smote
dtsmote_eachfold <- dtsmote_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
dtsmote_eachfold
#adas
dtadas_eachfold <- dtadas_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
dtadas_eachfold

#--------------Decision Tree Fold Graphs--------------#
dtcv_og <- ggplot(data=dt_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Decision Tree k-Fold CV:\
  Original Training Sample") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.75, 0.95)

dtcv_os <- ggplot(data=dtos_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Decision Tree k-Fold CV:\
  Over-sampling") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.75, 0.95)

dtcv_us <- ggplot(data=dtus_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Decision Tree k-Fold CV:\
  Under-sampling") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.75, 0.95)

dtcv_smote <- ggplot(data=dtsmote_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Decision Tree k-Fold CV:\
  SMOTE") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.75, 0.95)

dtcv_adas <- ggplot(data=dtadas_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Decision Tree k-Fold CV:\
  ADAS") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.75, .95) 

#--------------Decision Tree Graphs Combined--------------#
ggarrange(dtcv_og, dtcv_smote, dtcv_adas, ncol=3, nrow=1, 
          align = "v", widths = c(1,1,1,1,1),
          common.legend=FALSE, legend="bottom")

################################################################################
#-------------Cross Validation Models: Random Forest------------#
################################################################################
#D. Unfold the k-Folds for each model
#--------------Random Forest Folds--------------#
#orig
rf_kfold_pred <- rf_orig$pred
rf_kfold_pred$equal <- ifelse(rf_kfold_pred$pred == rf_kfold_pred$obs, 1,0)

#over-sampling
rfos_kfold_pred <- rf_os$pred
rfos_kfold_pred$equal <- ifelse(rfos_kfold_pred$pred == rfos_kfold_pred$obs, 1,0)

#under-sampling
rfus_kfold_pred <- rf_us$pred
rfus_kfold_pred$equal <- ifelse(rfus_kfold_pred$pred == rfus_kfold_pred$obs, 1,0)

#smote
rfsmote_kfold_pred <- rf_smote$pred
rfsmote_kfold_pred$equal <- ifelse(rfsmote_kfold_pred$pred == rfsmote_kfold_pred$obs, 1,0)

#adas
rfadas_kfold_pred <- rf_adas$pred
rfadas_kfold_pred$equal <- ifelse(rfadas_kfold_pred$pred == rfadas_kfold_pred$obs, 1,0)

#--------------Random Forest Fold Tables--------------#
#orig
rf_eachfold <- rf_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
rf_eachfold
#over-sampling
rfos_eachfold <- rfos_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
rfos_eachfold
#under-sampling
rfus_eachfold <- rfus_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
rfus_eachfold
#smote
rfsmote_eachfold <- rfsmote_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
rfsmote_eachfold
#adas
rfadas_eachfold <- rfadas_kfold_pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
rfadas_eachfold

#--------------Random Forest Fold Graphs--------------#
rfcv_og <- ggplot(data=rf_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Random Forest k-Fold CV:\
  Original Training Sample") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.95, 1.05)

rfcv_os <- ggplot(data=rfos_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Random Forest k-Fold CV:\
  Over-sampling") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.95, 1.05)

rfcv_us <- ggplot(data=rfus_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Random Forest k-Fold CV:\
  Under-sampling") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.95, 1.05)

rfcv_smote <- ggplot(data=rfsmote_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Random Forest k-Fold CV:\
  SMOTE") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.95, 1.05)

rfcv_adas <- ggplot(data=rfadas_eachfold, aes(x=Resample, y=Accuracy, group=1)) +
  geom_boxplot(color="springgreen4") + ggtitle("Random Forest k-Fold CV:\
  ADAS") + xlab("") +
  geom_point() + theme_bw() + 
  geom_text(size=3, aes(label=round(Accuracy,3)), vjust=-.5) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylim(0.95, 1.05)

#--------------Random Forest Graphs Combined--------------#
ggarrange(rfcv_og, rfcv_smote, rfcv_adas, ncol=3, nrow=1, 
          align = "v", widths = c(1,1,1,1,1),
          common.legend=FALSE, legend="bottom")

################################################################################
#-------------Confusion Matrices: Decision Tree------------#
################################################################################
dt_og_cm <- as.data.frame(table(predict(dt_orig,testData), testData$classification))
dt_og_cm1 <- ggplot(data = dt_og_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("Decision Tree Confusion Matrix:\ 
  Original Training Sample") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

dt_os_cm <- as.data.frame(table(predict(dt_os,testData), testData$classification))
dt_os_cm1 <- ggplot(data = dt_os_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + 
  ggtitle("\ 
  Over-sampling") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class") + 
  xlim(rev(levels(dt_os_cm$Var1)))

dt_us_cm <- as.data.frame(table(predict(dt_us,testData), testData$classification))
dt_us_cm1 <- ggplot(data = dt_us_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + 
  ggtitle("\ 
  Under-sampling") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class") + 
  xlim(rev(levels(dt_os_cm$Var1)))

dt_smote_cm <- as.data.frame(table(predict(dt_smote,testData), testData$classification))
dt_smote_cm1 <- ggplot(data = dt_smote_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("\ 
  SMOTE") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

dt_adas_cm <- as.data.frame(table(predict(dt_adas,testData), testData$classification))
dt_adas_cm1 <- ggplot(data = dt_adas_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("\ 
  ADAS") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

#--------------Decision Tree Confusion Matrices Combined--------------#
ggarrange(dt_og_cm1, dt_smote_cm1, dt_adas_cm1, ncol=3, nrow=1, 
          align = "v", widths = c(1,1,1,1,1),
          common.legend=FALSE, legend="bottom")

################################################################################
#-------------Confusion Matrices: Random Forest------------#
################################################################################
rf_og_cm <- as.data.frame(table(predict(rf_orig,testData), testData$classification))
rf_og_cm1 <- ggplot(data = rf_og_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("Random Forest Confusion Matrix:\ 
  Original Training Sample") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

rf_os_cm <- as.data.frame(table(predict(rf_os,testData), testData$classification))
rf_os_cm1 <- ggplot(data = rf_os_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + 
  ggtitle("Random Forest Confusion Matrix:\ 
  Over-sampling") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class") + 
  xlim(rev(levels(rf_os_cm$Var1)))

rf_us_cm <- as.data.frame(table(predict(rf_us,testData), testData$classification))
rf_us_cm1 <- ggplot(data = rf_us_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + 
  ggtitle("Random Forest Confusion Matrix:\ 
  Under-sampling") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class") + 
  xlim(rev(levels(rf_os_cm$Var1)))

rf_smote_cm <- as.data.frame(table(predict(rf_smote,testData), testData$classification))
rf_smote_cm1 <- ggplot(data = rf_smote_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("\ 
  SMOTE") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

rf_adas_cm <- as.data.frame(table(predict(rf_adas,testData), testData$classification))
rf_adas_cm1 <- ggplot(data = rf_adas_cm, mapping = aes(x = Var1, y = Var2)) + 
  geom_tile(aes(fill = Freq), show.legend=FALSE) + ggtitle("\ 
  ADAS") + geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "slategray1", high = "mediumpurple", trans = "log") +
  xlab("Predicted Class") + ylab("Actual Class")

#--------------Random Forest Confusion Matrices Combined--------------#
ggarrange(rf_og_cm1, rf_smote_cm1, rf_adas_cm1, ncol=3, nrow=1, 
          align = "v", widths = c(1.01,1,1),
          common.legend=FALSE, legend="bottom")

################################################################################
#-------------Classification Models: Decision Tree------------#
################################################################################
rpart.plot(dt_orig$finalModel,box.palette="GnRd", shadow.col="gray", nn=TRUE,
                       main="Decision Tree:\ Original Training Sample", 
                       type = 4)

rpart.plot(dt_os$finalModel,box.palette="RdGn", shadow.col="gray", nn=TRUE,
                       main="Decision Tree:\ Over-sampling", xflip = TRUE,
                       type = 4, extra = 101)
rpart.plot(dt_us$finalModel,box.palette="RdGn", shadow.col="gray", nn=TRUE,
                       main="Decision Tree:\ Under-sampling", xflip = TRUE,
                       type = 4, extra = 101)
rpart.plot(dt_smote$finalModel,box.palette="GnRd", shadow.col="gray", nn=TRUE,
                       main="Decision Tree:\ SMOTE",
                       type = 4)
rpart.plot(dt_adas$finalModel,box.palette="GnRd", shadow.col="gray", nn=TRUE,
                       main="Decision Tree:\ ADAS",
                       type = 4)

################################################################################
#-------------Classification Models: Random Forest------------#
################################################################################
#original
rfprep <- rf_prep(trainingData[,-9], trainingData[,9])
varImpPlot(rfprep$rf, main = "Random Forest Importance Plot:\ Original Training Data")

#over-sampling
rfprep.os <- rf_prep(trainingData.os[,-9], trainingData.os[,9])
varImpPlot(rfprep.os$rf, main = "Random Forest Importance Plot:\ Over-sampling")

#under-sampling
rfprep.us <- rf_prep(trainingData.us[,-9], trainingData.us[,9])
varImpPlot(rfprep.us$rf, main = "Random Forest Importance Plot:\ Under-sampling")

#smote
rfprep.smote <- rf_prep(trainingData.smote[,-9], trainingData.smote[,9])
varImpPlot(rfprep.smote$rf, main = "Random Forest Importance Plot:\ SMOTE")

#adas
rfprep.adas <- rf_prep(trainingData.adas[,-9], trainingData.adas[,9])
varImpPlot(rfprep.adas$rf, main = "Random Forest Importance Plot:\ ADAS")

################################################################################
#-------------Classification Models Performance: Accuracy------------#
################################################################################
#Lets reset the chart settings so we see one chart at a time
par(mfrow = c(1,1))

#Compare the Recall of the models: TP / TP + FN. 
#To do that, we'll need to combine our results into a dataframe

model_compare_accuracy <- data.frame(Model = c('DT-Orig',
                                             'RF-Orig',
                                             'DT-SMOTE',
                                             'RF-SMOTE',
                                             'DT-ADASYN',
                                             'RF-ADASYN'),
                                   Accuracy = c(accuracy_dtOrig,
                                              accuracy_rfOrig,
                                              accuracy_dtsmote,
                                              accuracy_rfsmote,
                                              accuracy_dtadas,
                                              accuracy_rfadas))

ggplot(aes(x = reorder(Model,-Accuracy),y = Accuracy),data = model_compare_accuracy) +
  geom_bar(stat = 'identity', fill = 'lightgoldenrod1') +
  ggtitle('Comparative Accuracy of Models on Test Data') +
  xlab('Models')  +
  ylab('Accuracy Measure')+
  geom_text(aes(label = round(Accuracy,2))) + theme_bw()



################################################################################
#-------------Classification Models Performance: Recall------------#
################################################################################
#Lets reset the chart settings so we see one chart at a time
par(mfrow = c(1,1))

#Compare the Accuracy of the models: (TP + TN) / (TP + FP + TN + FN). 
#To do that, we'll need to combine our results into a dataframe

model_compare_recall <- data.frame(Model = c('DT-Orig',
                                             'RF-Orig',
                                             'DT-SMOTE',
                                             'RF-SMOTE',
                                             'DT-ADASYN',
                                             'RF-ADASYN'),
                                   Recall = c(recall_dtOrig,
                                              recall_rfOrig,
                                              recall_dtsmote,
                                              recall_rfsmote,
                                              recall_dtadas,
                                              recall_rfadas))

ggplot(aes(x = reorder(Model,-Recall),y = Recall),data = model_compare_recall) +
  geom_bar(stat = 'identity', fill = 'light blue') +
  ggtitle('Comparative Recall of Models on Test Data') +
  xlab('Models')  +
  ylab('Recall Measure')+
  geom_text(aes(label = round(Recall,2))) + theme_bw()

################################################################################
#-------------Classification Models Performance: Precision------------#
################################################################################
#Compare the Precision of the models: TP/TP+FP 
model_compare_precision <- data.frame(Model = c('DT-Orig',
                                             'RF-Orig',
                                             'DT-SMOTE',
                                             'RF-SMOTE',
                                             'DT-ADASYN',
                                             'RF-ADASYN'),
                                   Precision = c(precision_dtOrig,
                                              precision_rfOrig,
                                              precision_dtsmote,
                                              precision_rfsmote,
                                              precision_dtadas,
                                              precision_rfadas))

ggplot(aes(x = reorder(Model,-Precision),y = Precision),data = model_compare_precision) +
  geom_bar(stat = 'identity',fill = 'light green') +
  ggtitle('Comparative Precision of Models on Test Data') +
  xlab('Models')  +
  ylab('Precision Measure')+
  geom_text(aes(label = round(Precision,2))) + theme_bw()

################################################################################
#-------------Classification Models Performance: F1 Measure------------#
################################################################################
#Compare the Precision of the models: TP/TP+FP 
model_compare_f1 <- data.frame(Model = c('DT-Orig',
                                                'RF-Orig',
                                                'DT-SMOTE',
                                                'RF-SMOTE',
                                                'DT-ADASYN',
                                                'RF-ADASYN'),
                                      F1 = c(F1_dtOrig,
                                                    F1_rfOrig,
                                                    F1_dtsmote,
                                                    F1_rfsmote,
                                                    F1_dtadas,
                                                    F1_rfadas))

ggplot(aes(x=reorder(Model,-F1),y = F1),data = model_compare_f1) +
  geom_bar(stat = 'identity',fill = 'thistle2') +
  ggtitle('Comparative F1 of Models on Test Data') +
  xlab('Models')  +
  ylab('F1 Measure')+
  geom_text(aes(label = round(F1,2))) + theme_bw()
