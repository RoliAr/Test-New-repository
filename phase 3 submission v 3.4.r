#############################################################
## Step 1: Load and install packages and libraries
#############################################################

#This package used to calculate the benchmark error rate
source("C:/Users/ROEYE/OneDrive/Documents/Roli New/Documents/BabsonAnalytics.R") 

#Install and call mice library to clean the data set

install.packages("mice")
install.packages("tidyverse")
install.packages("caret")
install.packages("readr")
install.packages("nnet")
install.packages("NeuralNetTools")
install.packages("rpart")
install.packages("rpart.plot")
library(mice)

#Install and call tidyverse library to check data duplication

library(tidyverse)


#Install and call caret library to to streamline the model training

library(caret)

#Install and call NeuralNetTools library to to visualize the nueral net
library(readr)
library(nnet)
library(NeuralNetTools)

#Install and call to building classification and regression trees
library(rpart)
library(rpart.plot)


#############################################################
## Step 2: Load the data set
#############################################################

df = read.csv("C:/Users/ROEYE/OneDrive/Desktop/cs-training.csv")
summary(df) #Show data set summary
set.seed(1234) #Ensure that the same random values are generated each time the code is executed.

#############################################################
## Step 3: Manage the dataset
#############################################################
df$SeriousDlqin2yrs=as.factor(df$SeriousDlqin2yrs)# convert the target to logical because the target is True/False
df$X = NULL #delete the ID field because it has no impact on prediction 

df$RevolvingUtilizationOfUnsecuredLines[df$RevolvingUtilizationOfUnsecuredLines>1.10]=NA
df$DebtRatio[df$DebtRatio>5000]=NA
df$MonthlyIncome[df$MonthlyIncome>25000]=NA

#############################################################
## Step 4: Clean the dataset
#############################################################

#see imputation index for NumberOfDependents
idx_NumberOfDependents= is.na(df$NumberOfDependents)
df$NumberOfDependents[idx_NumberOfDependents]

#see imputation index for MonthlyIncome
idx_MonthlyIncome= is.na(df$MonthlyIncome)
df$MonthlyIncome[idx_MonthlyIncome]

#fill the missing values using bootstrap

#imputation
imp_sample = mice(df, m = 1, method = "sample") # Impute missing values
df= complete(imp_sample) # Store imputed data

#standarize the data set
standeralization = preProcess(df, c("center", "scale")) 
df = predict(standeralization, df)



#############################################################
## Step 5: partitioning the data set to training and testing
#############################################################

N = nrow(df) #The number of rows in the data set
training_cases = sample(nrow(df),round(0.6*nrow(df)))
training_data = df[training_cases,]
test_data = df[-training_cases,]

#observations from the testing data
observations= test_data$SeriousDlqin2yrs 

#############################################################
## Step 6: Build Neural Network Model
#############################################################
#build the model with 4 hidden layers
nn_model = nnet(SeriousDlqin2yrs ~ ., data=training_data, size = 5)
#piloting the net
par(mar = numeric(4))
plotnet(nn_model, pad_x = 0.3)

#############################################################
## Step 7: Predict
#############################################################
#make predictions using the test_data
nn_predections = predict(nn_model, test_data, type="class") 
#create classification table to compare predictions to observations
table(nn_predections, test_data$SeriousDlqin2yrs)

#############################################################
## Step 8: Evaluate
#############################################################
error_rate_nn = sum(nn_predections != test_data$SeriousDlqin2yrs)/nrow(test_data)#calculate the error rate of the model
error_bench_nn = benchmarkErrorRate(training_data$SeriousDlqin2yrs, test_data$SeriousDlqin2yrs)

predictionTF_cutoff = nn_predections>0.5 #same cut off for all models
nn_observations = ifelse(test_data$SeriousDlqin2yrs==1,TRUE,FALSE)#same data type of predictions and observations
sensitivity_nn = sum(predictionTF_cutoff == TRUE & nn_observations == TRUE)/sum(nn_observations == TRUE)#how many times we predict trues
specificity_nn = sum(predictionTF_cutoff == FALSE & nn_observations == FALSE)/sum(nn_observations == FALSE)#how many times we predict falses



#############################################################
## Step 8: Stacking
#############################################################

#################################################
##Step 8.1: Models for stacking - Helper Models
#################################################

##Step 8.1.1: Classification Tree

#build model
stoppingRules = rpart.control(minsplit=50, minbucket = 10, cp = 0.01) #use to control the number of splits
model_tree = rpart(SeriousDlqin2yrs ~ ., data=training_data, control = stoppingRules )#build the model

#Cross validation
model_tree = easyPrune(model_tree) #control over fitting with prune
rpart.plot(model_tree)#show the pruned tree

#predictions
predictions_pruned_tree = predict(model_tree, test_data, type="class")
table(predictions_pruned_tree,observations )#Classification Tree Prediction Table

#Evaluate
error_pruned_tree = sum(predictions_pruned_tree != observations)/nrow(test_data)
sensitivity_tree = sum(predictions_pruned_tree == 1 & test_data$SeriousDlqin2yrs == 1)/sum(test_data$SeriousDlqin2yrs == 1)
specificity_tree = sum(predictions_pruned_tree == 0 & test_data$SeriousDlqin2yrs == 0)/sum(test_data$SeriousDlqin2yrs == 0)


#################################
##Step 8.1.2: Logistic Regression
#################################

#build model
model_LR= glm(SeriousDlqin2yrs~.,data=training_data, family=binomial)

#Cross validation
model_LR=step(model_LR)

#predictions
LLR_Prediction= predict(model_LR, test_data, type="response") #type of this prediciton
predictions_tf_LR= LLR_Prediction>0.5 #getting true/false instead of probability using 0.4 cut-off
predictions_tf_LR = ifelse(predictions_tf_LR=="TRUE",1,0)
table(predictions_tf_LR,observations) #Logistic Regression Prediction Table

#Evaluate
error_rate_LR= sum(predictions_tf_LR != observations)/nrow(test_data)# error rate using the model
senseitivity_LR = sum(predictions_tf_LR == 1 & observations == 1)/ sum(observations==1)#how many time we correctly predict True 
specifity_LR = sum(predictions_tf_LR == 0 & observations == 0)/ sum(observations==0)#how many time we correctly predict False



#################################################
##Step 8.1.3: Stacking
#################################################


pred_LR_full = predict(model_LR, df, type= "response")
pred_rf_full = predict(model_tree, df)
df_stacked = cbind(df,pred_LR_full, pred_rf_full)


#data partitioning
train_stacked = df_stacked[training_cases, ]
test_stacked = df_stacked[-training_cases, ]

#build manager model
manager_stacked = nnet(SeriousDlqin2yrs ~ ., data = train_stacked, size = 4)

#predictions
pred_stacked = predict(manager_stacked, test_stacked, type="class")
pred_stacked_TF = pred_stacked > 0.5
observations_stacked = ifelse(observations==1,TRUE,FALSE)
table(pred_stacked_TF,observations_stacked)

#Evaluate
error_stacked = sum(pred_stacked_TF != observations_stacked)/nrow(test_stacked)
error_bench_stack = benchmarkErrorRate(train_stacked$SeriousDlqin2yrs, test_stacked$SeriousDlqin2yrs)

senseitivity_stacked = sum(pred_stacked_TF == TRUE & observations_stacked == TRUE)/ sum(observations_stacked==TRUE)#how many time we correctly predict True 
specifity_stacked = sum(pred_stacked_TF == FALSE & observations_stacked == FALSE)/ sum(observations_stacked==FALSE)#how many time we correctly predict False


