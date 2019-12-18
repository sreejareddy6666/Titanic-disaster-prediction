#Logistic regression (binary classification)

# dataset is available at https://www.kaggle.com/c/titanic/data

## Load Titanic library to get the dataset
library(titanic)

## Load the datasets
data("titanic_train")
data("titanic_test")

###############Data details and missing values imputation
## Setting Survived column for test data to NA
titanic_test$Survived <- NA

## Combining Training and Testing dataset
complete_data <- rbind(titanic_train, titanic_test)

## Check data structure
str(complete_data)


## Let's check for any missing values in the data
colSums(is.na(complete_data))

## Checking for empty values
colSums(complete_data=='')

## Check number of uniques values for each of the column 
## to find out columns which we can convert to factors
sapply(complete_data, function(x) length(unique(x)))

## Missing values imputation
complete_data$Embarked[complete_data$Embarked==""] <- "S"
complete_data$Age[is.na(complete_data$Age)] <- median(complete_data$Age,na.rm=T)

## Removing Cabin as it has very high missing values, passengerId, Ticket and Name are not required
library(dplyr)
titanic_data <- complete_data %>% select(-c(Cabin, PassengerId, Ticket, Name))

#For logistic regression, it is important to create dummy variables 
#for all the categorical variables so that all the factors are taken into account while creating the model.
## Converting "Survived","Pclass","Sex","Embarked" to factors
for (i in c("Survived","Pclass","Sex","Embarked")){
  titanic_data[,i]=as.factor(titanic_data[,i])
}

## Create dummy variables for categorical variables
library(dummies)
titanic_data <- dummy.data.frame(titanic_data, names=c("Pclass","Sex","Embarked"), sep="_")

train <- titanic_data[1:667,]
test <- titanic_data[668:891,]

model <- glm(Survived ~.,
             family=binomial(link='logit'),
             data=train) 
summary(model)

## Predicting Test Data
result <- predict(model,newdata=test,type='response')
result <- ifelse(result > 0.5,1,0)
result <- factor(result)
result

## Confusion matrix and statistics
library(caret)
confusionMatrix(table(result,as.factor(test$Survived)))


## ROC Curve and calculating the area under the curve(AUC)
library(ROCR)
predictions <- predict(model, newdata=test, type="response")
ROCRpred <- prediction(predictions, test$Survived)
ROCRperf <- performance(ROCRpred, measure = "tpr", x.measure = "fpr")
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7), print.cutoffs.at = seq(0,1,0.1))

#As a rule of thumb, a model with good predictive ability 
#should have an AUC closer to 1 (1 is ideal) 
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
auc

## Predicting Test Data
result <- predict(model,newdata=titanic_data[892:nrow(titanic_data),],type='response')
result <- ifelse(result > 0.5,1,0)
result <- factor(result)
result

titanic_data$Survived[892:nrow(titanic_data)]<-result
