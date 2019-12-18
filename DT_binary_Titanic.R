# decision tree (binary classification-breast cancer dataset)
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
dim(titanic_data)
## Converting "Survived","Pclass","Sex","Embarked" to factors
for (i in c("Survived","Pclass","Sex","Embarked")){
  titanic_data[,i]=as.factor(titanic_data[,i])
}

train <- titanic_data[1:667,]
test <- titanic_data[668:891,]


library("rpart")
library("rpart.plot")
tree<-rpart(Survived~., data = train, method = "class")
rpart.plot(tree)
plot(tree)
text(tree, digits = 3)
print(tree)
prp(tree,box.palette = "Reds")
rpart.rules(tree, roundint=FALSE, clip.facs=TRUE)
row.names(tree$frame)[tree$where]
library(rattle)
rattle::asRules(tree, TRUE)
#You can get the number of rules (leaves) in this way:
nrules <- as.integer(rownames(tree$frame[tree$frame$var == "<leaf>",]))
nrules
#You can also iterate for the rules like this:
#rules <- lapply(nrules, path.rpart, tree=fit, pretty=0, print.it=FALSE)
#Another alternative is using the package rpart.plot
#rules <- rpart.plot::rpart.rules(model, cover=T, nn=T)

predict_unseen <-predict(tree, test, type = 'class')
table_mat <- table(test$Survived, predict_unseen)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
paste("Accuracy for Test set=",accuracy_Test)

