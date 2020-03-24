install.packages(c('randomForest', 'cowplot', 'ggplot2', 'pROC', 'ROSE'))

library(ggplot2)
library(cowplot) # Improve some of ggplot2's default settings
theme_set(theme_cowplot()) # Need this line bcz: as of version 1.0.0, cowplot does not change the default ggplot2 theme anymore. To recover the previous behavior.
library(randomForest)
library(pROC) # Model performance evaluation and ROC curve
library(ROSE) # Randomly over or under sample sizes


## Load data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data <- read.csv(url, header = FALSE, sep="")

## Feature Engineering
# First, have a look at the dataset
head(data)
# Find the dataset in unlabeled

# Second, Rename the columns according to the detailed information listed on UCI website: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
colnames(data) <- c("chk_ac_status_1",
                    "duration_month_2", "credit_history_3", "purpose_4",
                    "credit_amount_5","savings_ac_bond_6","p_employment_since_7", 
                    "instalment_pct_8", "personal_status_9","other_debtors_or_grantors_10", 
                    "present_residence_since_11","property_type_12","age_in_yrs_13",
                    "other_instalment_type_14", "housing_type_15", 
                    "number_cards_this_bank_16","job_17","no_people_liable_for_mntnance_18",
                    "telephone_19", "foreign_worker_20", 
                    "good_bad_21")

head(data)

# Third, check the data structure
str(data)
# Good to see the dataset is well structured, and no missing values.
# Convert all int variables to numeric.
# good_bad_21 is supposed to be a factor, where 1 represents "Good"(not default) and 2 represents "Bad"(default).
data$duration_month_2 <- as.numeric(data$duration_month_2)
data$credit_amount_5 <- as.numeric(data$credit_amount_5 )
data$instalment_pct_8 <- as.numeric(data$instalment_pct_8)
data$present_residence_since_11 <- as.numeric(data$present_residence_since_11)
data$age_in_yrs_13 <- as.numeric(data$age_in_yrs_13)
data$number_cards_this_bank_16 <- as.numeric(data$number_cards_this_bank_16)
data$no_people_liable_for_mntnance_18 <- as.numeric(data$no_people_liable_for_mntnance_18)

data$good_bad_21 <- as.factor(ifelse(test = data$good_bad_21==1, yes = "Good", no = "Bad"))

# Check the data structure again
str(data)
# It looks good now

# Visualize the dataset
barplot(table(data$good_bad_21), col="light blue", main="Class Distribution")
# We find the dataset is highly unbalanced, which may affect the model building. We will consider this later.

set.seed(42)

# Divide the dataset into train and test set by 70% and 30%.
div_part <- sort(sample(nrow(data), nrow(data)*.7))
train <- data[div_part,]
test <- data[-div_part,]

## Build Logistic Regression Model
glm_model <- glm(good_bad_21 ~ ., data=train, family = binomial)

# Look how the model performs
glmpredict <- predict(glm_model, newdata = test, type = "response")
par(pty="s") # To get rid of the unnecessary blank parts of the graph
roc(test$good_bad_21, glmpredict, plot=TRUE, legacy.axes=TRUE, percent = TRUE,
    xlab="False Positive Percentage", ylab="True Positive Percentage", print.auc=TRUE)

# Analyze the TPP, the FPP and the thresholds used when the False Positive Rate is between 20 and 40
glm.roc.info <- roc(test$good_bad_21, glmpredict, legacy.axes=TRUE)
glm.roc.df <- data.frame(
  tpp=glm.roc.info$sensitivities*100,
  fpp=(1-glm.roc.info$specificities)*100,
  thresholds=glm.roc.info$thresholds
)
glm.roc.df[glm.roc.df$fpp > 20 & glm.roc.df$fpp < 40,]

# Draw the partial AUC
roc(test$good_bad_21, glmpredict, plot=TRUE, legacy.axes=TRUE, percent = TRUE,
    xlab="False Positive Percentage", ylab="True Positive Percentage", main="Logistic Regression ROC Graph",
    col="#377eb8", lwd=4, print.auc=TRUE, print.auc.x=45, partial.auc=c(80,60),
    auc.polygon=TRUE, auc.polygon.col="#377eb822")

## Build Random Forest Model
# Since we don't have NAs in the dataset, we could skip the step to deal with the NAs
# If we need to deal with the missing values in the train set, we could use rfImpute()

# Build the RF model with default parameters
rf_model <- randomForest(good_bad_21 ~ ., data = train, proximity = TRUE)

# Get a summary of the RF model and see how well it fits the train dataset
rf_model

# Understand the model summary:
# Our RF model is for classification
# The number of trees is 500, and we will check if 500 trees is enough for optimal classification later on
# 4 variables were considered in our RF model at each internal node, we will figure out if 4 is the best value or not later, too
# OOB error estimate is 26.86%. Means that 73.14% of the OOB samples were correctly classified by the RF model.
# From the Confusion Matrix, we notice that the error between classes is highly unbalanced. This is because the Bad class is much larger than the Good class as we see before.
# Random Forests, trying to minimize overall error rate, will keep the error rate low on the large class while letting the smaller classes have a larger error rate.
# We could balance the error by setting different weights to each class.

## Deal with the unbalanced dataset
# Weighted Random Forests - Weighted sample size

# Oversampling
table(train$good_bad_21)
data_over <- ovun.sample(good_bad_21 ~ ., data = train, method = "over", N = 980)$data # The number is two times of the larger one(Good)
table(data_over$good_bad_21) # The sample of Bad was successfully amplified, and the train data was balanced now 
rf_model_over <- randomForest(good_bad_21 ~ ., data = data_over, proximity = TRUE)
rf_model_over

# Undersampling
data_under <- ovun.sample(good_bad_21 ~ ., data = train, method = "under", N = 420)$data # The number is two times of the smaller one(Bad)
table(data_under$good_bad_21) # The sample of Good was successfully shrinked, and the train data was balanced again
rf_model_under <- randomForest(good_bad_21 ~ ., data = data_under, proximity = TRUE)
rf_model_under

# Both, ie increase the number of smaller side at the same time decrease the number of larger side, RANDOMLY
data_both <- ovun.sample(good_bad_21 ~ ., data = train, method = "both", N = 700)$data # The numer is the total observations of Good and Bad
table(data_both$good_bad_21) # The number of Good and Bad seems close, and the train data was balanced now
rf_model_both <- randomForest(good_bad_21 ~ ., data = data_both, proximity = TRUE)
rf_model_both

# Oversampling has the lowest OOB error rate, and it doesn't lose any information of our dataset.
# So we choose to over the sample and keep going with the rf_model_over to optimize the model.

## Optimize the RF model(after balancing the dataset)
# First: To see if 500 trees is enough for optimal classification, we can plot the error rates.
# Create a dataframe that formats the error rate
oob.error.data <- data.frame(
  Trees=rep(1:nrow(rf_model_over$err.rate), times=3),
  Type=rep(c("OOB", "Bad", "Good"), each=nrow(rf_model_over$err.rate)),
  Error=c(rf_model_over$err.rate[,"OOB"],
          rf_model_over$err.rate[,"Bad"],
          rf_model_over$err.rate[,"Good"])
)

# Plot the error rate
ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

# In general, we see the error rates decrease when our RF has more trees, if we add more trees, would the error rate go down further?

# Second, to test the hypothesis, make another RF model with 1,000 trees
rf_model_over_1000 <- randomForest(good_bad_21 ~ ., data = data_over, ntree=1000, proximity=TRUE)
rf_model_over_1000
# The OOB is almost the same as before.
# The confusion matrix also shows that we didn't do a much better job correctly classifying Bad.

# Plot the error rates just like before.
oob.error.data <- data.frame(
  Trees=rep(1:nrow(rf_model_over_1000$err.rate), times=3),
  Type=rep(c("OOB", "Bad", "Good"), each=nrow(rf_model_over_1000$err.rate)),
  Error=c(rf_model_over_1000$err.rate[,"OOB"],
          rf_model_over_1000$err.rate[,"Bad"],
          rf_model_over_1000$err.rate[,"Good"])
)

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
# We could see that the error rates stabilize right after 500 trees.
# So adding more trees doesn't help. We still go with 500 trees.

# Second: To make sure we are considering the optimal number of variables at each internal node in the tree.
# Start by making an empty vector that can hold 20 values.(already the maximum variables we could consider)
oob.values <- vector(length = 20)
# Create a loop to test different numbers of variables at each step
for(i in 1:20){
  temp.model <- randomForest(good_bad_21 ~ ., data = data_over, mtry=i, ntree=500)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}

oob.values
# The third value has the lowest OOB error rate, so we would take 3 as the optimal values.

## Check the performance improvement after optimizing the model
# Build the optimal model using the parameters we have tuned
rf_model_opt <- randomForest(good_bad_21 ~ ., data = data_over, mtry=3, ntree=500, proximity = TRUE)
rf_model_opt
# The OOB error rate decreased a little bit from 9.39% to 9.18%, with the class error both decreased too.
# We did improve the model although it's a little bit. The possible reason is the data size is relatively small.

# Draw an MDS plot with samples to show us how they are related to each other.
distance.matrix <- dist(1-rf_model_opt$proximity)
mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

# Format the data for ggplot
mds.values <- mds.stuff$points
mds.data <- data.frame(
  Sample = rownames(mds.values),
  X=mds.values[,1],
  Y=mds.values[,2],
  Status=data_over$good_bad_21
)

# Draw the graph
ggplot(data = mds.data, aes(x=X, y=Y, label=Sample))+
  geom_text(aes(color=Status))+
  theme_bw()+
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep=""))+
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep = ""))+
  ggtitle("MDS plot using (1 - Random Forest Proximities)")

## Compare the prediction performance of Logistic Regression and Random Forest in our dataset
# Draw the ROC graph of Logistic Regression again
roc(test$good_bad_21, glmpredict, plot=TRUE, legacy.axes=TRUE, percent = TRUE,
    xlab="False Positive Percentage", ylab="True Positive Percentage",
    col="#377eb8", lwd=4, print.auc=TRUE)

# Using the optimal RF model to make a prediction, and draw the ROC graph
rf_predict <- predict(rf_model_opt, newdata = test, type="prob")
plot.roc(test$good_bad_21, rf_predict[,2], percent=TRUE,
         col="#4daf4a", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=40)

par(pty="m")
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col=c("#377eb8", "#4daf4a"), lwd=4)
