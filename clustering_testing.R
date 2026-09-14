# ==============================================================================
# Project: Nano-Tumor Data Analysis (MBA Thesis)
# Script: Machine Learning Pipeline – Clustering Analysis
# Created: 2023
# Description: Preprocessing, cluster analysis and RF modeling
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
library(readxl)
library(h2o)
library(FactoMineR)
library(factoextra)

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

# Data preprocessing ----
Dat_DE_Max  <- Dat_preprocessing(Variable = "DE_Max")


# K-means clustering ----
# Fiding k
fviz_nbclust(Dat_DE_Max[[1]], kmeans, method = "wss")+
  geom_vline(xintercept = 3, linetype = 2)

set.seed(123)
km.res=kmeans(Dat_DE_Max[[1]], 3, nstart=25)
#print(km.res)
aggregate(Dat_DE_Max[[1]], by=list(cluster=km.res$cluster), mean)
Dat_DE_Max2=cbind(Dat_DE_Max[[1]], cluster=km.res$cluster)
#head(Dat_DE_Max2)
#km.res$centers

# Visualize clusters
fviz_cluster(km.res, data=Dat_DE_Max2,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
             ellipse.type="euclid",
             star.plot=TRUE,
             #repel=TRUE,
             ggtheme=theme_minimal()
)


fviz_cluster(km.res, data = Dat_DE_Max2,
             palette = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"),
             ggtheme = theme_minimal(),
             main = "Partitioning Clustering Plot"
)

cluster1_teste <- filter(Dat_DE_Max2, cluster == 1)
cluster2_teste <- filter(Dat_DE_Max2, cluster == 2)
cluster3_teste <- filter(Dat_DE_Max2, cluster == 3)

# Train RF per cluster ----
# Create the ML function ----
ML <- function (data, cv = 5, rep = 5, method) {
  set.seed(3496)
  
  ## Create control function for training
  ctrl <- trainControl(method  = 'repeatedcv', 
                       number  = cv, # 5-fold cv  
                       repeats = rep,
                       search  ='grid')
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


## Training the RF model ----
#Cluster1
rfFit_DE_Max <-ML (data=cluster1_teste, method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_Max, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

#Cluster2
rfFit_DE_Max <-ML (data=cluster2_teste, method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_Max, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

#Cluster3
rfFit_DE_Max <-ML (data=cluster3_teste, method = 'rf') 
trellis.par.set(caretTheme())
plot(rfFit_DE_Max, col = 'coral2',lty=par("lty"),lwd=1, 
     xlab= 'Número de preditores selecionados aleatóriamente',ylab= 'R²' )

# Compute hierarchical clustering and cut into 4 clusters ----
fviz_nbclust(Dat_DE_Max[[1]], FUN = hcut, method = "wss")+
  geom_vline(xintercept = 3, linetype = 2)

res <- hcut(Dat_DE_Max[[1]], k = 3, stand = TRUE)

# Visualize
fviz_dend(res, rect = TRUE, cex = 0.5,
          k_colors = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"))

fviz_cluster(res, data=df_carros_final,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
             ellipse.type="euclid",
             star.plot=TRUE,
             #repel=TRUE,
             ggtheme=theme_minimal()
)

fviz_cluster(res, data = df_carros_final,
             palette = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"),
             ggtheme = theme_minimal(),
             main = "Partitioning Clustering Plot"
)

df_grupos <- cutree(res, k = 3)
table(df_grupos)
df_resultado_cluster <- data.frame(df_grupos) 

#df_resultado_cluster
df_final <- cbind(Dat_DE_Max[[1]], df_resultado_cluster)

#df_final
# Original clusters as columns
conf.seeds <- table(df_final$df_grupos,Dat_DE_Max[[1]]$Y)
#conf.seeds

# Calculate prediction accuracy function
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100} 
accuracy(conf.seeds)

cluster1 <- filter(df_final, df_grupos == 1)
cluster2 <- filter(df_final, df_grupos == 2)
cluster3 <- filter(df_final, df_grupos == 3)

