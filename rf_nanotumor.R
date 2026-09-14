# ==============================================================================
# Project: Nano-Tumor Data Analysis (MBA Thesis)
# Script: Machine Learning Pipeline (Random Forest Model)
# Created: 2023
# Description: Preprocessing, Random Forest training and evaluation
# ==============================================================================

# Load packages
library(caret)
library(tidyverse)
library(doParallel)
library(Boruta)
library(glmnet)
library(randomForest)
library(gbm)
library(e1071)
library(kernlab)
library(LiblineaR)
library("readxl")
library(h2o)

# Initialize H2O ----
h2o.init(nthreads=-1, max_mem_size="2G")
h2o.removeAll() 

# Data preprocessing function ----
Dat_preprocessing <- function(Variable = "DE_Max") {
  ## Load the data
  data <- read.csv(file = 'data.csv')
  data <- data%>%select(Type:ZP, Variable)
  data <- data%>%filter(!is.na(ZP))
  
  ## Encode categorical variables
  tofactor<- function (x) {
    for (i in 1:ncol(x)){
      if (is.character(x[,i]) == TRUE){
        x[,i] = factor(x[,i],
                       levels = levels(factor(x[,i])),
                       labels = seq(1:length(levels(factor(x[,i])))))
      }
    }
    return(x)
  }
  
  ## Encoding categorical variables
  data_r <- data %>% tofactor 
  #str(data_r)
  
  ## Scale data: normalized value = (value - mean)/sd
  scaled      <-caret::preProcess(data_r[,-dim(data_r)[2]], method = c("center"))
  transformed <-predict(scaled, newdata = data_r)
  colnames(transformed)[9]<-"Y"
  
  ## One-hot encoding
  dummies     <- dummyVars(Y~., data = transformed)
  mm          <- predict(dummies, newdata = transformed)
  
  ## Create H2OFrame for use in the h2o package
  MM<-as.h2o(mm %>% as.data.frame %>% mutate(Y=transformed[,"Y"]))
  
  ## Feature selection: Zero and near-zero variance features
  feature_variance <- caret::nearZeroVar(MM, saveMetrics = TRUE)
  shortlistedVars  <- colnames(MM[, feature_variance$nzv == 'FALSE'])
  #print(shortlistedVars)
  
  # Create training (80%) and testing (20%) sets
  splits <- h2o.splitFrame(MM, c(0.8), seed=123)
  splits_lable <- h2o.splitFrame(as.h2o(data_r), c(0.8), seed=3496)
  train  <- h2o.assign(splits[[1]], "train.hex") # 80%
  valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
  
  ## Define target and predictor variables
  response <- "Y"
  predictors <- setdiff(shortlistedVars, response)
  
  ## Convert to data frame
  Dat_train <- as.data.frame(train[,c(predictors,response)])
  X_test    <- as.data.frame(valid[,c(predictors,response)])%>%select(-Y)
  Y_test    <- as.data.frame(valid[,c(predictors,response)])%>%select(Y)
  
  
  return (list(Dat_train, X_test,Y_test))
}

# Create the ML function ----
ML <- function (data, cv = 5, rep = 5, method) {
  set.seed(3496)
  
  ## Create control function for training
  ctrl <- trainControl(method  = 'repeatedcv', 
                       number  = cv, # 5-fold cv  
                       repeats = rep,
                       search  ='grid') # Grid search method
  grid <- base::expand.grid(.mtry=c(1:10))
  model <- train(Y ~ .,
                 data = as.data.frame(data), 
                 method = method,
                 num.trees = 700,
                 
                 trControl = ctrl,
                 metric = "RMSE",
                 tuneGrid = grid)        
  return(model)
} 

# Data preprocessing ----
Dat_DE_Max  <- Dat_preprocessing(Variable = "DE_Max")
Dat_DE_24   <- Dat_preprocessing(Variable = "DE_24")
Dat_DE_168  <- Dat_preprocessing(Variable = "DE_168")

set.seed(3496)

## Training the RF model ----
#DE Max
rfFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_Max, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

#DE 24
rfFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_24, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

#DE 168
rfFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_168, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

## Testing the model ----
## DE MAX
rfFit_DE_Max_test  <- rfFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])

## DE 24
rfFit_DE_24_test  <- rfFit_DE_24%>%predict(newdata = Dat_DE_24[2])

## DE 168
rfFit_DE_168_test  <- rfFit_DE_168%>%predict(newdata = Dat_DE_168[2])

## Predicting ----
## DE MAX
pred_DE_Max_test<-cbind.data.frame(rfFit_DE_Max_test)

## DE 24
pred_DE_24_test<-cbind.data.frame(rfFit_DE_24_test)

## DE 168
pred_DE_168_test<-cbind.data.frame(rfFit_DE_168_test)


post_results_DE_Max <-apply(pred_DE_Max_test, 2, postResample, obs = as.matrix(Dat_DE_Max[[3]]))
post_results_DE_24 <-apply(pred_DE_24_test, 2, postResample, obs = as.matrix(Dat_DE_24[[3]]))
post_results_DE_168 <-apply(pred_DE_168_test, 2, postResample, obs = as.matrix(Dat_DE_168[[3]]))


# Feature importance ----
#DE Max
rfImp <- varImp(rfFit_DE_Max, scale = TRUE)
## Plot top 5 variables in a bar chart
#plot(rfImp,top = 10)
imp <- as.data.frame(rfImp[1])
imp <- imp %>% mutate(nomes=row.names(.))
head(imp)
imp <- top_n(imp, n=5, Overall) 
ggplot(imp, aes(x=reorder(nomes, Overall), weight= Overall, fill = "magenta")) + 
  geom_bar() +
  theme_light(base_size = 15) +
  scale_fill_discrete(name="Variable Group") +
  ylab("Importance (%)") +
  xlab("Variable Name")

#DE 24
rfImp <- varImp(rfFit_DE_24, scale = TRUE)
## Plot top 5 variables in a bar chart
imp <- as.data.frame(rfImp[1])
imp <- imp %>% mutate(nomes=row.names(.))
head(imp)
imp <- top_n(imp, n=5, Overall) 
ggplot(imp, aes(x=reorder(nomes, Overall), weight= Overall, fill = "magenta")) + 
  geom_bar() +
  theme_light(base_size = 15) +
  scale_fill_discrete(name="Variable Group") +
  ylab("Importance (%)") +
  xlab("Variable Name")

#DE 168
rfImp <- varImp(rfFit_DE_168, scale = TRUE)
## Plot top 5 variables in a bar chart
imp <- as.data.frame(rfImp[1])
imp <- imp %>% mutate(nomes=row.names(.))
head(imp)
imp <- top_n(imp, n=5, Overall) 
ggplot(imp, aes(x=reorder(nomes, Overall), weight= Overall, fill = "magenta")) + 
  geom_bar() +
  theme_light(base_size = 15) +
  scale_fill_discrete(name="Variable Group") +
  ylab("Importance (%)") +
  xlab("Variable Name")

# Predicted vs. Observed plot ----
#DE 24
pred  <- cbind.data.frame(rfFit_DE_24_test)
obs   <- rbind.data.frame(as.data.frame(Dat_DE_24[3]))%>%rename(Obs = Y)
PlotDat <- cbind.data.frame(obs,pred)
p <- ggplot(PlotDat, aes(x=Obs, y = rfFit_DE_24_test)) + 
  geom_point(size = 3, colour = "coral2") +
  theme_light(base_size = 15) +
  scale_shape_manual(values=13:(13+9))+
  geom_abline (intercept = 0,
               slope     = 1,
               color     ="black",size = 1)+
  scale_x_continuous(limits = c(0, 12))+
  scale_y_continuous(limits = c(0, 12))+
  ylab("DE predita (%ID)") +
  xlab("DE observada (%ID)")
p

#DE Max
pred  <- cbind.data.frame(rfFit_DE_Max_test)
obs   <- rbind.data.frame(as.data.frame(Dat_DE_Max[3]))%>%rename(Obs = Y)
PlotDat <- cbind.data.frame(obs,pred)
p <- ggplot(PlotDat, aes(x=Obs, y = rfFit_DE_Max_test)) + 
  geom_point(size = 3, colour = "coral2") +
  theme_light(base_size = 15) +
  scale_shape_manual(values=13:(13+9))+
  geom_abline (intercept = 0,
               slope     = 1,
               color     ="black",size = 1)+
  scale_x_continuous(limits = c(0, 12))+
  scale_y_continuous(limits = c(0, 12))+
  ylab("DE predita (%ID)") +
  xlab("DE observada (%ID)")
p

#DE 168
pred  <- cbind.data.frame(rfFit_DE_168_test)
obs   <- rbind.data.frame(as.data.frame(Dat_DE_168[3]))%>%rename(Obs = Y)
PlotDat <- cbind.data.frame(obs,pred)
p <- ggplot(PlotDat, aes(x=Obs, y = rfFit_DE_168_test)) + 
  geom_point(size = 3, colour = "coral2") +
  theme_light(base_size = 15) +
  scale_shape_manual(values=13:(13+9))+
  geom_abline (intercept = 0,
               slope     = 1,
               color     ="black",size = 1)+
  scale_x_continuous(limits = c(0, 12))+
  scale_y_continuous(limits = c(0, 12))+
  ylab("DE predita (%ID)") +
  xlab("DE observada (%ID)")
p
