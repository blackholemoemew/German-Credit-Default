# GermanCreditDefault

## 1.Introduction
This is a backup of my coursework for the Financial Modeling course. The main topic is to build and optimize a machine learning model to predict whether an individual would default or not on credit card, and to learn how to handle imbalanced dataset.

Below is a brief summary of the project.

| Category | Description | Source |
|:--------:|:-----------:|:-------------------:|
|Dataset|German credit data from UCI database|[Link](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)|
|Language|R|[Code](/GermanCredit.R)|
|Packages|ggpolt2, cowplot, randomForest, pROC, ROSE||
|Algorithms|Logistic Regression, Random Forest||
|Reference 1|Video: Random Forests in R|[Link](https://www.youtube.com/watch?v=6EXPYzbfLCE&t=801s)|
|Reference 2|Article: Using Random Forest to Learn Imbalanced Data|[Link](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)|
|Reference 3|Video: Handling Class Imbalance Problem in R: Improving Predictive Model Performance|[Link](https://www.youtube.com/watch?v=Ho2Klvzjegg&t=984s)|

## 2.Load Data and Feature Engeering
The good news was that there was no NAs in the dataset, so I didn't struggle with handling the missing values. But I wanted to keep in mind that Random Forests has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing. In R, we could use rfImpute() function from randomForest package.

```R
> url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
> data <- read.csv(url, header = FALSE, sep="")
> 
> colnames(data) <- c("chk_ac_status_1",
+                     "duration_month_2", "credit_history_3", "purpose_4",
+                     "credit_amount_5","savings_ac_bond_6","p_employment_since_7", 
+                     "instalment_pct_8", "personal_status_9","other_debtors_or_grantors_10", 
+                     "present_residence_since_11","property_type_12","age_in_yrs_13",
+                     "other_instalment_type_14", "housing_type_15", 
+                     "number_cards_this_bank_16","job_17","no_people_liable_for_mntnance_18",
+                     "telephone_19", "foreign_worker_20", 
+                     "good_bad_21")
> 
> data$duration_month_2 <- as.numeric(data$duration_month_2)
> data$credit_amount_5 <- as.numeric(data$credit_amount_5 )
> data$instalment_pct_8 <- as.numeric(data$instalment_pct_8)
> data$present_residence_since_11 <- as.numeric(data$present_residence_since_11)
> data$age_in_yrs_13 <- as.numeric(data$age_in_yrs_13)
> data$number_cards_this_bank_16 <- as.numeric(data$number_cards_this_bank_16)
> data$no_people_liable_for_mntnance_18 <- as.numeric(data$no_people_liable_for_mntnance_18)
> 
> data$good_bad_21 <- as.factor(ifelse(test = data$good_bad_21==1, yes = "Good", no = "Bad"))
```

## 3.Get Familiar with the Dataset
There are 1,000 observations of 20 attributes (7 numerical, 13 categorical), and a class column where 1 represents "Good" (no events) and 2 represents "Bad" (events) in this dataset. The detailed information of each attributes could be accessed [here](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/). I divided the dataset by 70/30 into a train set and test set for my model builing and evaluation. From the distribution graph, I noticed that the dataset was imbalanced because the number of "Good" samples was twice as much as "Bad" samples. The imbalanced dataset was a problem that needed to consider. I tried several ways to deal with it.

![Distribution.jpeg](https://i.loli.net/2020/03/24/mPeaCwb72YvpFSI.jpg)

## 4.Using Logisic Regression Model
Because this is a binomial classification problem, the first algorithm I thought about is logistic regression, which was easy to implement, interpret and train. I built a logistic regressio model first and looked how it performed. The logistic regression achieved a 80.47 AUC score. It was not bad.

```R
> glm_model <- glm(good_bad_21 ~ ., data=train, family = binomial)
> 
> glmpredict <- predict(glm_model, newdata = test, type = "response")
> par(pty="s") # To get rid of the unnecessary blank parts of the graph
> roc(test$good_bad_21, glmpredict, plot=TRUE, legacy.axes=TRUE, percent = TRUE,
+     xlab="False Positive Percentage", ylab="True Positive Percentage", print.auc=TRUE)

Call:
roc.default(response = test$good_bad_21, predictor = glmpredict,     percent = TRUE, plot = TRUE, legacy.axes = TRUE, xlab = "False Positive Percentage",     ylab = "True Positive Percentage", print.auc = TRUE)

Data: glmpredict in 90 controls (test$good_bad_21 Bad) < 210 cases (test$good_bad_21 Good).
Area under the curve: 80.47%
```

![GLM-ROC.jpeg](https://i.loli.net/2020/03/23/rYcPd5KoC46Qk9s.jpg)

In this case, we wanted to achieve a high TPP percentage with a relatively low FPP, i.e. correctly classified as much "Bad" samples as possible at the same time incorrectly classified as little "Good" samples as possible. We could trade-off the TPP and FPP by choosing different threshold. So I printed out all the TPP, the FPP and the threshold in the range of FPP from 20 to 40. Supposed we could accept a FPP of 30%, then we could use 0.6485 as a threshold to make a prediction and got a TPP of 76.19%. In practical, 30% of "Good" individual would be unnecessary paid more attention because our model incorrectly classified them as "Bad". The trade-off should take more things into consideration.

```R
> glm.roc.df[glm.roc.df$fpp > 20 & glm.roc.df$fpp < 40,]
         tpp      fpp thresholds
98  80.00000 38.88889  0.6043567
99  80.00000 37.77778  0.6071486
100 80.00000 36.66667  0.6080981
101 79.52381 36.66667  0.6105500
102 79.04762 36.66667  0.6132030
103 79.04762 35.55556  0.6138499
104 79.04762 34.44444  0.6200281
105 78.57143 34.44444  0.6278430
106 78.57143 33.33333  0.6301111
107 78.09524 33.33333  0.6330924
108 77.61905 33.33333  0.6361043
109 77.14286 33.33333  0.6383487
110 77.14286 32.22222  0.6406966
111 76.66667 32.22222  0.6411083
112 76.66667 31.11111  0.6435608
113 76.19048 31.11111  0.6465538
114 76.19048 30.00000  0.6485132
115 75.71429 30.00000  0.6502164
116 75.23810 30.00000  0.6513013
117 74.76190 30.00000  0.6545168
118 74.76190 28.88889  0.6595778
119 74.76190 27.77778  0.6635910
120 74.76190 26.66667  0.6650678
121 74.28571 26.66667  0.6672349
122 73.80952 26.66667  0.6771005
123 73.33333 26.66667  0.6854107
124 73.33333 25.55556  0.6876607
125 73.33333 24.44444  0.6933296
126 72.85714 24.44444  0.6980166
127 72.38095 24.44444  0.7008047
128 71.90476 24.44444  0.7043748
129 71.42857 24.44444  0.7084716
130 71.42857 23.33333  0.7118293
131 71.42857 22.22222  0.7148019
132 70.95238 22.22222  0.7179930
133 70.47619 22.22222  0.7199887
134 70.47619 21.11111  0.7218029
135 70.00000 21.11111  0.7244272
136 69.52381 21.11111  0.7275383
137 69.04762 21.11111  0.7316884
138 68.57143 21.11111  0.7399027
139 68.09524 21.11111  0.7484806
```

I didn't dive in the Logistic Regression here, and only used it as a comparision for the Random Forests model.

## 5.Build a Random Forests Model
Another popular discrimination tool is Random Forests. Random Forests own many excellent features:

> Some Features of Random Forests
>  - It is unexcelled in accuracy among current algorithms.
>  - It can handle thousands of input variables without variable deletion.
>  - It gives estimates of what variables are important in the classification.
>  - It generates an internal unbiased estimate of the generalization error as the forest building progresses.
>  - It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
>  - It has methods for balancing error in class population unbalanced data sets.
>  - It computes proximities between pairs of cases that can be used in clustering, locating outliers, or (by scaling) give interesting views of the data.
>
> By Leo Breiman and Adele Cutler. Check full information here: [Link](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#features)

First, built the original Random Forests model using default parameters. The output told us:
1. Our RF model is for classification.
2. The number of trees is 500, and we will check if 500 trees is enough for optimal classification later.
3. 4 variables were considered in our RF model at each internal node, and we will figure out if 4 is the best value or not later.
4. OOB error estimate is 26.86%, which means 73.14% of the OOB samples were correctly classified by the RF model.
5. From the Confusion Matrix, we could see the effect of a highly imbalanced dataset.

```R
> rf_model <- randomForest(good_bad_21 ~ ., data = train, proximity = TRUE)
> rf_model

Call:
 randomForest(formula = good_bad_21 ~ ., data = train, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 26.86%
Confusion matrix:
     Bad Good class.error
Bad   68  142  0.67619048
Good  46  444  0.09387755
```

## 6.Handle the Imbalanced Dataset
Random Forests, trying to minimize overall error rate, will keep the error rate low on the large class while letting the smaller classes have a larger error rate. I balanced the error by setting different weights to each class. I used the R package "ROSE" to randomly increase or decrease the sample size.

### 6.1 Oversampling
After amplifying the "Bad" sample size to the same with "Good" sample, I trained the RF model again. The OOB error rate decreased from 26.86% to 9.39% with each class error rate more balanced.

```R
> data_over <- ovun.sample(good_bad_21 ~ ., data = train, method = "over", N = 980)$data # The number is two times of the larger one(Good)
> rf_model_over <- randomForest(good_bad_21 ~ ., data = data_over, proximity = TRUE)
> rf_model_over

Call:
 randomForest(formula = good_bad_21 ~ ., data = data_over, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 9.39%
Confusion matrix:
     Good Bad class.error
Good  434  56  0.11428571
Bad    36 454  0.07346939
```

### 6.2 Undersampling
Then, I tried undersampling the "Good" examples and trained the RF model. The OOB error rate increased from 26.86% to 30.48%, though the two class error rates were balanced. Many people said that usually downsizing performed better than oversizing. But from my experiment, the latter one did a better job.

```R
> data_under <- ovun.sample(good_bad_21 ~ ., data = train, method = "under", N = 420)$data # The number is two times of the smaller one(Bad)
> rf_model_under <- randomForest(good_bad_21 ~ ., data = data_under, proximity = TRUE)
> rf_model_under

Call:
 randomForest(formula = good_bad_21 ~ ., data = data_under, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 30.48%
Confusion matrix:
     Good Bad class.error
Good  139  71   0.3380952
Bad    57 153   0.2714286
```

### 6.3 Both
I also tried to increase the "Good" examples and decrease the "Bad" examples at the same time. The OOB error rate decreased to 10.14%, and the class error rates seemed pretty good.

```R
> data_both <- ovun.sample(good_bad_21 ~ ., data = train, method = "both", N = 700)$data # The numer is the total observations of Good and Bad
> rf_model_both <- randomForest(good_bad_21 ~ ., data = data_both, proximity = TRUE)
> rf_model_both

Call:
 randomForest(formula = good_bad_21 ~ ., data = data_both, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 10.14%
Confusion matrix:
     Good Bad class.error
Good  330  40  0.10810811
Bad    31 299  0.09393939
```

### 6.4 Trade-off between the Three Ways
Intuitively speaking, both might be the best way and it did perform much better than original dataset. However, since oversampling had the lowest OOB error rate, and it didn't lose any information of our dataset("both" and "under" both lost information by randomly deleting some samples). I chose to oversize the sample and used the amplified dataset to optimize the model.

## 7.Optimize the Random Forests Model
I tuned two parameters to help optimize the RF model we just built: the number of trees and the number of variables considered at each internal node.

### 7.1 Number of Trees
First, I draw a graph to show how the error rates changed as the trees grew in our original model (ntree=500, used the train set that had been amplified). In general, the error rates decreased when our RF had more trees, if more trees added, would the error rates go down further?

```R
> oob.error.data <- data.frame(
+   Trees=rep(1:nrow(rf_model_over$err.rate), times=3),
+   Type=rep(c("OOB", "Bad", "Good"), each=nrow(rf_model_over$err.rate)),
+   Error=c(rf_model_over$err.rate[,"OOB"],
+           rf_model_over$err.rate[,"Bad"],
+           rf_model_over$err.rate[,"Good"])
+ )
> 
> # Plot the error rate
> ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
+   geom_line(aes(color=Type))
```
![RF-500.jpeg](https://i.loli.net/2020/03/23/Cfle5a8JXwpgS9t.jpg)

Then, I built a new RF model by setting the ntree=1000, means the model would generate 1,000 trees at this time. And draw the error rates graph. The OOB error rate was almost the same as before(9.8% vs 9.39%). And the confusion matrix also showed that we didn't do a much better job correctly classifying "Bad" examples. From the error rates graph, I noticed that the error rates stabilized right after 500 trees. So adding more trees didn't help. I still went with 500 trees.

```R
> rf_model_over_1000 <- randomForest(good_bad_21 ~ ., data = data_over, ntree=1000, proximity=TRUE)
> rf_model_over_1000

Call:
 randomForest(formula = good_bad_21 ~ ., data = data_over, ntree = 1000,      proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 1000
No. of variables tried at each split: 4

        OOB estimate of  error rate: 9.8%
Confusion matrix:
     Good Bad class.error
Good  428  62  0.12653061
Bad    34 456  0.06938776

> oob.error.data <- data.frame(
+   Trees=rep(1:nrow(rf_model_over_1000$err.rate), times=3),
+   Type=rep(c("OOB", "Bad", "Good"), each=nrow(rf_model_over_1000$err.rate)),
+   Error=c(rf_model_over_1000$err.rate[,"OOB"],
+           rf_model_over_1000$err.rate[,"Bad"],
+           rf_model_over_1000$err.rate[,"Good"])
+ )
> 
> ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
+   geom_line(aes(color=Type))
```
![RF-1000.jpeg](https://i.loli.net/2020/03/23/xHUw53azqMyZSW7.jpg)

### 7.2 Number of Variables at Each Internal Node
I built different RF models by setting mtry from 1 to 20 (already the maximum variables we could consider) and compared the OOB error rates of those models. The third value had the lowest OOB error rate, 0.09081633, so I would take 3 as the optimal values.

```R
> for(i in 1:20){
+   temp.model <- randomForest(good_bad_21 ~ ., data = data_over, mtry=i, ntree=500)
+   oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
+ }
> 
> oob.values
 [1] 0.12448980 0.10306122 0.09081633 0.09489796 0.10510204 0.10306122 0.10918367 0.10408163 0.11836735 0.11326531
[11] 0.11326531 0.12142857 0.11530612 0.11326531 0.11734694 0.11020408 0.11224490 0.11428571 0.11326531 0.11836735
```

### 7.3 Build the Optimal RF Model
From the analysis above, I could build the optimal RF model now. The OOB error rate decreased a little bit from 9.39% to 9.18%, with the class error both decreased too. The model was indeed improved although it was a small progress. The possible reason may be the data size was small.

```R
> rf_model_opt <- randomForest(good_bad_21 ~ ., data = data_over, mtry=3, ntree=500, proximity = TRUE)
> rf_model_opt

Call:
 randomForest(formula = good_bad_21 ~ ., data = data_over, mtry = 3,      ntree = 500, proximity = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 3

        OOB estimate of  error rate: 9.18%
Confusion matrix:
     Good Bad class.error
Good  434  56  0.11428571
Bad    40 450  0.08163265
```

## 8.Clustering the Samples by MDS Graph
I drew a MDS graph for the optimal I just built using the internally calculated proximities to find the ralation between the variables. From the graph, almost all the "Good" samples were on the left side and "Bad" samples were on the right side. The sample 757 might be misclassified as "Bad" and actully a good credit individual. The x-axis accounted for 11.2% of the variance in the distance matrix while the y-axis only accounted for 3.8% of the variance in the distance matrix. If we had a new example and didn't know it was good or bad, I could be confident to classify it as good if it clustered down left-hand tail and classify it as bad if it clustered up right-hand tail.

```R
> distance.matrix <- dist(1-rf_model_opt$proximity)
> mds.stuff <- cmdscale(distance.matrix, eig = TRUE, x.ret = TRUE)
> mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)
> 
> # Format the data for ggplot
> mds.values <- mds.stuff$points
> mds.data <- data.frame(
+   Sample = rownames(mds.values),
+   X=mds.values[,1],
+   Y=mds.values[,2],
+   Status=data_over$good_bad_21
+ )
> ggplot(data = mds.data, aes(x=X, y=Y, label=Sample))+
+   geom_text(aes(color=Status))+
+   theme_bw()+
+   xlab(paste("MDS1 - ", mds.var.per[1], "%", sep=""))+
+   ylab(paste("MDS2 - ", mds.var.per[2], "%", sep = ""))+
+   ggtitle("MDS plot using (1 - Random Forest Proximities)")
```
![RF-MDS.jpeg](https://i.loli.net/2020/03/23/C8pYrNicqymT4kX.jpg)

## 9.Compare the Two Models Using ROC
Drew the ROC curve of the Logistic Regression model and the optimal Random Forests model, and put the curve together to make a comparision.

```R
> # Draw the ROC graph of Logistic Regression again
> roc(test$good_bad_21, glmpredict, plot=TRUE, legacy.axes=TRUE, percent = TRUE,
+     xlab="False Positive Percentage", ylab="True Positive Percentage",
+     col="#377eb8", lwd=4, print.auc=TRUE)
Setting levels: control = Bad, case = Good
Setting direction: controls < cases

Call:
roc.default(response = test$good_bad_21, predictor = glmpredict,     percent = TRUE, plot = TRUE, legacy.axes = TRUE, xlab = "False Positive Percentage",     ylab = "True Positive Percentage", col = "#377eb8", lwd = 4,     print.auc = TRUE)

Data: glmpredict in 90 controls (test$good_bad_21 Bad) < 210 cases (test$good_bad_21 Good).
Area under the curve: 80.47%
> 
> # Using the optimal RF model to make a prediction, and draw the ROC graph
> rf_predict <- predict(rf_model_opt, newdata = test, type="prob")
> plot.roc(test$good_bad_21, rf_predict[,2], percent=TRUE,
+          col="#4daf4a", lwd=4, print.auc=TRUE, add=TRUE, print.auc.y=40)
Setting levels: control = Bad, case = Good
Setting direction: controls > cases
> 
> par(pty="m")
> legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col=c("#377eb8", "#4daf4a"), lwd=4)
```
![GLMvsRF.jpeg](https://i.loli.net/2020/03/23/ezrwnLFQx7Saiyv.jpg)

## 10.Conclusion
In terms of ROC score, the two models performed nearly the same. But I would go with the Random Forests algorithm, because:

1. The logistic regression may have the problem of overfit while random forests doesn't.
2. From the ROC graph, when the TPP was at a high level, the random forests model brought a lower FPP than logistic regression model, which was what we wanted to achieve.
3. Random Forests provided information to cluster the samples, which was also a way for us to make a classification.
