library(dplyr) # for data manipulation
library(caret) # for model-building
library(DMwR) # for smote implementation 
library(smotefamily) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) 



#####Data processing#####



#R packages
library(dplyr) # for data manipulation
library(caret) # for model-building
library(DMwR) # for smote implementation 
library(smotefamily) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC)
library(ROSE)

##data loading###
data <- read.csv("pesticides.csv",header = T)

###Datasets pre-processing####
nzv <- nearZeroVar(data, saveMetrics= TRUE)
dim(data)
nzv <- nearZeroVar(data)
filteredDescr <- data[,-nzv]
dim(filteredDescr)
head(filteredDescr)
write.csv(filteredDescr,"pesticides-nzv.csv")
#相关变量处理
newdata <- read.csv("pesticides-nzv.csv",header = T)
newdata=filteredDescr[,-662]#delete group
descrCor <-  cor(newdata)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
newdata <- newdata[,-highlyCorDescr]
group <- as.matrix(data$group)#merge frame +group
finaldata <- as.data.frame(cbind(newdata,group))
write.csv(finaldata,'pesticides-nzvcorr.csv')


#“over","down","both"
imbal_train <- read.csv("pesticides-nzvcorr.csv",header = T)
imbal_train$group = ifelse(imbal_train$group == "control", "control", "treat")
data_over <-ovun.sample(group ~ ., data=imbal_train,method="over",N=450)$data
data_under <-ovun.sample(group ~ ., data=imbal_train,method="under",N=376)$data
data_both <-ovun.sample(group ~ ., data=imbal_train,method="both")$data

#SMOTE
library(DMwR)
data <- read.csv("pesticides-nzvcorr.csv",header = T)
data$group <- as.factor(data$group)
table(data$group)
DATA <- SMOTE(group ~ ., data, perc.over=500, perc.under =120)
table(DATA$group)
prop.table(table(DATA$group))


#####Datasets saving######
write.csv(DATA, "pesticides-smote.csv")

#####Datasets partitioning######
data <- read.csv("pesticides-under.csv",header = T)

index<- createDataPartition(data$group, p=.75,list = F)
data_train <- data[index,]
data_test <- data[-index,]
table(data_train$group)
prop.table(table(data_train$group))
table(data_test$group)
prop.table(table(data_test$group))

write.csv(data_train,'pesticides-under-train.csv',row.names = F)
write.csv(data_test,'pesticides-under-test.csv',row.names = F)
