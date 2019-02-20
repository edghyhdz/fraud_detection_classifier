# Fraud detection classifier

In this notebook it is presented how the data was firstly delved with, how it was processed and prepared in order to build a binary classification model, which eventually indicate whether a transaction was fraudulent or not.

The data set was obtained from `kaggle` and can be downloaded [here](https://www.kaggle.com/ntnu-testimon/paysim1/kernels). It is a syntetically generated data set `"[using] aggregated data from [a] private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods"` [<sup>1</sup>](https://www.kaggle.com/ntnu-testimon/paysim1/home). It consists of 6 million data rows, containing financial information of transactions and current account status. 


The classification model was built using tensorflow's estimator API.

>**NOTE:** The whole procedure is probably better explained on the jupyter notebook, if you want to see all the code and how everything was done, look at the noteebok file rendered via nbviewer clicking [here](https://nbviewer.jupyter.org/github/edghyhdz/fraud_detection_classifier/blob/master/Fraud_detection_classifier.ipynb), otherwise just click on the `Fraud_detection_classifier.ipynb` file on this repository. **The table of contents does not work when rendered via GITHUB!** 

Otherwise you can continue reading here, where just a summary of the whole analysis is presented.

# 1 Data overview

Column data overview (as taken from database [description](https://www.kaggle.com/ntnu-testimon/paysim1/kernels)):

**`step`** - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

**`type`** - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

**`amount`** - amount of the transaction in local currency.

**`nameOrig`** - customer who started the transaction

**`oldbalanceOrg`** - initial balance before the transaction

**`newbalanceOrig`** - new balance after the transaction

**`nameDest`** - customer who is the recipient of the transaction

**`oldbalanceDest`** - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

**`newbalanceDest`** - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

**`isFraud`** - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

**`isFlaggedFraud`** - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.


## 1.1 Checking the data

Seeing how many fraud events were labeled in the whole data period. There were on average 11 events per hour.

<img align="center" width="50%" height="50%" src="https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/fraud_hist.jpeg">

<img align="center" width="120%" height="120%" src="https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/data_exp.jpeg">

## 1.2 Looking for data correlation

As seen below in the graphs, we can see that in the upper left graph there is a clear log<sub>10</sub> correlation between `oldbalanceOrg` vs `amount` where `isFraud == 1`. We may probably use this to our advantage when defining which type of model we will be using, as well as what the validation and training sets will be.

Meaning that most likely we will not be using the entire data set to train the model
<img align="left" width="120%" height="120%" src="https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/data_exp_2.jpeg">

# 2 Proposed problem solution

The graph on the left, shows a sample fraccion (`5%`) of `amount` vs `oldbalanceOrg` for the case in which `isFraud == 0`. The shaded line indicates when an overlapp between `isFraud == 1` and `isFraud == 0` is happening. Meaning that we can keep the data which lies inside the shaded area. And thus, use it in order to build our model.

Once the data outside the shaded area has been removed, we obtain what is displayed on the right graph. 

Proposed data separation   |  Separated data
:-------------------------:|:-------------------------:
![](https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/data_sep.jpeg)  |  ![](https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/data_sep_3.jpeg)


Since only 0.38 % of `filtered_data` is marked as fraud (`isFraud == 1`), and in order to maintain data homogeneity wrt to labeled and unlabeled data, the training data will consist of 0.76 % of the whole `sub_df`. 

This means that we will be taking randomly 0.38% from data labeled as `isFraud == 0`, in order to build the data set which will be used to train our classification model.

# 3 Model validation

In order to validate the model, it was decided to use a `4fold Cross-validation` framework. Meaning that `4` different and independent models will be trained and validated using each of the cross-validation data sets.

The way in which the data was split, was by following a `stratified` splitting, which returns stratified folds of training and validation data. An example of the stratified folds is shown in the image below. 

<img align="center" width="70%" height="70%" src="https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/cross_val.jpeg">


These files are also included under the `cross_val_folds` [folder](https://github.com/edghyhdz/fraud_detection_classifier/tree/master/cross_val_folds)


# 4 Results

The results are presented below, were for the case of the model accuracy, the shaded area is the average and ± standard deviations of the `test_data` for all 4 cross-validation folds. The confusion matrix, was built by using only one of the trained models.

Model accuracy             |  Confusion matrix
:-------------------------:|:-------------------------:
![](https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/accuracy.jpeg)  |  ![](https://github.com/edghyhdz/fraud_detection_classifier/blob/master/images/confusion_matrix_.jpeg)

Out of `4078` predictions, `4055` were correctly predicted, `23` were false positives, and `0` false negatives



# 5 Acknowledgements

Thanks to [E. A. Lopez-Rojas](https://www.kaggle.com/ealaxi) for providing this excelent data set.

