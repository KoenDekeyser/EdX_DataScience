
# Libraries ---------------------------------------------------------------
#These will install the required libraries if necessary
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(forcats)) install.packages("forcats")
if(!require(corrplot)) install.packages("corrplot")
if(!require(randomForest)) install.packages("randomForest")
if(!require(GGally)) install.packages("GGally")

#Loading needed libraries
library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(ggplot2)
library(forcats)
library(corrplot)
library(randomForest)
library(GGally)

# Pulling and wrangling dataset -------------------------------------------
#URL is cut up to improve the aesthetics of the report
url1 <- "https://raw.githubusercontent.com/KoenDekeyser/EdX_DataScience/main/"
url2 <- "WDI_database.csv"
url <- paste0(url1,url2)

#initial wrangling to pivot it from tidy to long data
db_pull <- read.csv(url) %>%
  select(area, value, date, name) %>%
  pivot_wider(names_from = name, values_from = value,
              names_sep = "_") %>%
  filter(!is.na(Undernourishment))

#Replace the column headers with no-space names
names(db_pull) <-  str_replace_all(names(db_pull), c(" " = "_"))
summary(db_pull)

#indicators deselected because of too many missing data
db <- db_pull %>%
  select(!c(Poverty, Inequality, Water_use_agriculture,
            Life_expectancy)) %>%
  #Remaining NA's are filled up by taking the area mean of that indicator
  group_by(area) %>%
  mutate(Agricultural_land = ifelse(is.na(Agricultural_land), 
                                    mean(Agricultural_land, na.rm = T), 
                                    Agricultural_land),
         Forest_land  = ifelse(is.na(Forest_land), 
                               mean(Forest_land, na.rm = T), 
                               Forest_land),
         Cereal_yield  = ifelse(is.na(Cereal_yield), 
                                mean(Cereal_yield, na.rm = T), 
                                Cereal_yield),
         GDP_growth = ifelse(is.na(GDP_growth), 
                             mean(GDP_growth, na.rm = T), 
                             GDP_growth),
         Food_imports = ifelse(is.na(Food_imports), 
                               mean(Food_imports, na.rm = T),  
                               Food_imports),
         Population_growth = ifelse(is.na(Population_growth ), 
                                    mean(Population_growth , na.rm = T),  
                                    Population_growth),
         GDP_per_capita = ifelse(is.na(GDP_per_capita ), 
                                 mean(GDP_per_capita , na.rm = T),  
                                 GDP_per_capita),
         Agriculture_GDP = ifelse(is.na(Agriculture_GDP ), 
                                  mean(Agriculture_GDP , na.rm = T),  
                                  Agriculture_GDP),
         Agricultural_jobs = ifelse(is.na(Agricultural_jobs ), 
                                    mean(Agricultural_jobs , na.rm = T),  
                                    Agricultural_jobs),
         Food_production_index = ifelse(is.na(Food_production_index ), 
                                        mean(Food_production_index , na.rm = T),  
                                        Food_production_index),
         Urban_population = ifelse(is.na(Urban_population ), 
                                   mean(Urban_population , na.rm = T),  
                                   Urban_population)) %>%
  ungroup() %>%
  #two indicators are converted to their logarithm
  mutate(GDP_per_capita_log = log(GDP_per_capita),
         Cereal_yield_log = log(Cereal_yield)) %>%
  select(!c(GDP_per_capita, Cereal_yield)) %>%
  na.omit()
rm(db_pull)


# Splitting into training and validation set ------------------------------
# Validation set will be 10% of dataset
set.seed(1, sample.kind="Rounding") #if using R 3.5 or earlier, use `set.seed(1)
test_index <- createDataPartition(y = db$Undernourishment, 
                                  times = 1, p = 0.1, list = FALSE)
edx <- db[-test_index,]
validation <- db[test_index,]
rm(test_index)


# Modelling approach ------------------------------------------------------
#RMSE function
RMSE <- function(true_undernourishment, predicted_undernourishment){
  sqrt(mean((true_undernourishment - predicted_undernourishment)^2))
}


# Results -----------------------------------------------------------------

#Additional evaluation dataset will be 10% of training dataset
set.seed(1, sample.kind="Rounding")#if using R 3.5 or earlier, use `set.seed(1)
test_index <- createDataPartition(y = edx$Undernourishment, 
                                  times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]

rm(test_index)

#A general GLm model with all numeric indicators as predictors
glm_train <- train(Undernourishment ~ .,
                   method = "glm",
                   metric = "RMSE",
                   data = edx_train[,-1])
summary(glm_train)
glm_hat <- predict(glm_train, edx_test)
RMSE_glm <- RMSE(edx_test$Undernourishment, glm_hat)

#A randomforest model primed for RMSE evaluation
set.seed(1, sample.kind="Rounding")#if using R 3.5 or earlier, use `set.seed(1)
rf_train <- train(Undernourishment ~ ., 
                  method = "rf",
                  metric = "RMSE",
                  data = edx_train[,-1])
rf_hat <- predict(rf_train, edx_test)
RMSE_rf <- RMSE(edx_test$Undernourishment, rf_hat)

data.frame(
  Model = c("Genearlized linear model",
            "Random forest"),
  Performance = c(RMSE_glm,
                  RMSE_rf)) %>%
  kbl("latex",
      booktabs = T,
      caption = "Performance of the models") %>%
  kable_classic(full_width = F) %>%
  row_spec(0, bold=T) %>%
  kable_styling(latex_options = "hold_position")

#finetuning
set.seed(1, sample.kind="Rounding") 
#Five-fold cross validation with zero iterations
control <- trainControl(method="cv", number = 5)
#tuning parameters between 4 and 10
grid <- data.frame(mtry = seq(4,10,1))

train_rf<- train(Undernourishment ~ ., 
                 method = "rf",
                 metric = "RMSE",
                 trControl = control,
                 tuneGrid = grid,
                 data = edx_train[,-1])

#Final model
set.seed(1, sample.kind="Rounding")#if using R 3.5 or earlier, use `set.seed(1)
rf_fit <- randomForest(Undernourishment ~ .,
                       minNode  = train_rf$bestTune$mtry,
                       data = edx_train[,-1])
rf_hat <- predict(rf_fit, validation)
RMSE_final <- RMSE(validation$Undernourishment, rf_hat)

#Show the importance 
stack(importance(rf_fit)[order(importance(rf_fit), decreasing = T),])


