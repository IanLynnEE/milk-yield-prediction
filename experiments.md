# Topic

## Abstract


## Introduction



## Framework

The data were initially verified before further processing. We have more than 20 features for a record, and some features were poorly managed. We thus suspect that some records might contain errors, and those should not be included in the training data. Therefore, we compared features of each record and dropped the records with confliction before preprocessing.

In data modeling, features were encoded and transformed such that the output features contain only numbers, and options were provided to drop features or modify encoding methods for different models.

The data will be sent to three models, ANN, SVR and RF. We would verify the usefulness of features by using options provided by data modeling. After that, the models were refined by using different parameters.



## Methods

### Data Verifying

The following things were checked:

1. The age of a cow should agree with the sampling date and the birthday.
2. The sampling date should be no later than the recording date.
3. The number of days of lactation should agree with the sampling date and the last delivery date.
4. The birthday recorded in the "report" should agree with the one in the "birth".
5. A cow should have exactly one birthday.
6. The serial number of a cow should not appear in the "report" after it died.
7. Cows with ages less than 21 months should not be on the "report".
8. The number of deliveries should be less or equal to the number of breedings.

### Data Modeling

<!-- Define how to calculate mean and std here. -->

<!-- explain options provided in the framework for later jobs -->

### Training Model

The models were trained by different features and parameters.

To verify the models, two methods were used. One was using 10 random validation sets, and we denoted the average RMSE as Average RMSE. The other method used data sampling between 2018/01/01 to 2018/06/30 as the validation set, and we denoted the corresponding RMSE as Special RMSE.

#### Select Features

We first used different features and encoding methods to verify the usefulness of features. To verify the usefulness of features, sets of parameters for models were used. 

For ANN, we used 3 hidden layers with hidden neurons 150, 100 and 50. Rectified linear unit (Relu) was chosen as the activation function for the network. Adam optimizer was selected with 50 max iterations. 

For RF, the number of estimators was 1000, with the criterion set to be squared error.

For SVR, epsilon and cost were set to be 0.1, 1 respectively.

We also picked the min-max scaler as the features scaler, as we found not all the features were normally distributed.

#### Select Parameters

Given the selected features and encoding methods, we do some experiments on different settings of models. 

For SVR, we mainly focus on different epsilon and cost for the Radial basis function kernel. Different kernels, such as the sigmoid kernel, were tested as well.

For ANN, we focus on different numbers of hidden layers, as we were curious about the underfitting/overfitting problem.

#### Handling Test Data with Missing Features

<!-- Define how does "add time interval" works-->



## Result

### Data Verifying

After examining the verifying result, the following data were dropped:
- ID = 6960: sampling date > recoding date.
- ID = 16714: missing important features.
- Data with cows' ages less than 21 months.
- Data with milk yield equal to 0.

The cows with the following serial numbers were recorded to be dead in the "spec", yet they appear in the "report" later:
- 98051976
- 94051730
- 97051412

After a quick check, we believed that the serial number would be reused on some occasions. As a result, we would not adopt data in the "spec" before further review.

In the process, we also found that the data in columns 16 to 21 in the "report" doesn't match the official meaning. In column 18, the official description is "number of breeding", yet we found lots of records have 0 in this column. In column 16, the official description is "last breeding date", yet it might be later than the recording date of the record. However, the time intervals calculated by these features proved to be valuable later. As a result, we can only suggest that the descriptions are misleading.

### Data Modeling

<!-- Define features' name and meaning here -->

<!-- Define how does label encoding work on serial. It used birthday. -->

### Training Models

#### Select Features

##### ANN

We first used ANN to verify different encoding methods and features. Starting from the following features:

| feature   | preprocessing     |
|-----------|-------------------|
| year      |                   |
| month     |                   |
| ranch     | label encoded     |
| serial    | based on birthday |
| father    | frequency encoded |
| mother    | frequency encoded |
| delivery  |                   |
| lactation |                   |
| age       |                   |

We tried to change the encoding methods of the serial and the ranch, and we tried to provide more features to ANN.

|                          | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | Test 6 |
|--------------------------|--------|--------|--------|--------|--------|--------|
| Frequency encoded serial | X      | O      | O      | O      | O      | O      |
| One-hot encoded ranch    | X      | X      | O      | O      | O      | O      |
| Add breeding             | X      | X      | X      | O      | O      | O      |
| Add delivery season      | X      | X      | X      | X      | O      | O      |
| Add time intervals       | X      | X      | X      | X      | X      | O      |
|                          |        |        |        |        |        |        |
| Average RMSE             | 6.092  | 6.014  | 5.988  | 5.931  | 5.976  |        |
| Special RMSE             | 5.837  | 5.810  | 5.837  | 5.871  | 5.825  |        |
| Real RMSE                |        |        |        |        | 6.274  | 6.257  |

In the process, we found out that max_iter is a key factor for the result. For different sets of features, the ideal max_iter varies, which makes it hard to determine the usefulness of features.

Moreover, for the special (fixed) validation set, the result varied with time. The above table took the mean of 10 runs. Therefore, we hoped the Random Forest Regressor can give us clear views.

##### Random Forest Regressor

The Random Forest Regressor takes much longer to compute, 

We try to find the usefulness of features first and do experiments on different encoding later.

We will only use the special validation set in the second part. Random Forest Regressor will give us the same result as long as `random_state` is fixed. We found out that the special validation set can better descript the real RMSE. 

上面那句好難寫。反正就是，因為固定 random state，我們能夠確定，特別挑的那組 valid set 更貼近上傳後的 RMSE。那我們就不用跑 random 的 valid 了，這樣可以省下很多時間。



features scaler: `MinMaxScaler`

| feature   | preprocessing     |
|-----------|-------------------|
| year      |                   |
| month     |                   |
| ranch     | one-hot encoded   |
| serial    | frequency encoded |
| father    | frequency encoded |
| mother    | frequency encoded |
| delivery  |                   |
| lactation |                   |
| age       |                   |



|                     | Test 1 | Test 2 | Test 3 | Test 4 |
|---------------------|--------|--------|--------|--------|
| Add breeding        | X      | O      | O      | O      |
| Add delivery season | X      | X      | O      | O      |
| Add time intervals  | X      | X      | X      | O      |
|                     |        |        |        |        |
| Average RMSE        | 5.505  | 5.392  | 5.346  |        |
| Special RMSE        | 5.764  | 5.709  | 5.689  |        |
| Real RMSE           | 6.303  | 6.260  | 6.242  | ??     |

看起來 breeding 跟 deliveery season 都有點用處。

<!-- X doesn't mean dropping this feature, it means going back to the label encoded. -->

|                          | Test 1 | Test 2 | Test 3 |
|--------------------------|--------|--------|--------|
| Frequency encoded serial | O      | O      | X      |
| One-hot encoded ranch    | O      | X      | X      |
| Add breeding             | O      | O      | O      |
| Add delivery season      | O      | O      | O      |
|                          |        |        |        |
| Special RMSE             | 5.689  | 5.689  | 5.980  |


### SVR

    # regr = SVR(
    #     epsilon=0.1,
    #     C=1,
    #     cache_size=8000
    # )


SVR 也是一樣每次都一樣的結果（固定 valid 時），不固定的 std 大概 0.06~0.08

|                          | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | Test 6 | Test 7 | Test 8 | Test 9 |
|--------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Frequency encoded serial | X      | O      | O      | O      | O      | O      | O      | O      | O      |
| One-hot encoded ranch    | X      | X      | O      | O      | O      | X      | X      | X      | X      |
| Add breeding             | X      | X      | X      | O      | O      | O      | O      | X      | O      |
| Add delivery season      | X      | X      | X      | X      | O      | O      | X      | O      | O      |
| Add time intervals       | X      | X      | X      | X      | X      | X      | X      | X      | O      |
|                          |        |        |        |        |        |        |        |        |        |
| Average RMSE             | 6.435  | 6.331  | 6.394  | 6.405  | 6.330  | 6.209  | 6.282  | 6.256  |        |
| STD RMSE                 | 0.079  | 0.089  | 0.082  | 0.064  | 0.063  | 0.063  | 0.073  | 0.073  |        |
| Real RMSE                |        |        |        |        |        | 6.524  |        |        | 6.389  |


Use test 6's result:
<!-- Picture here -->

e = 0.4, c = 0.9 doesn't do well on random sets.

0.4, 0.9, 6.275063, 27409
0.4, 0.9, 6.286322, 27346
0.4, 0.9, 6.264244, 27361
0.4, 0.9, 6.327693, 27360
0.4, 0.9, 6.326100, 27332
0.4, 0.9, 6.253120, 27362
0.4, 0.9, 6.122793, 27346
0.4, 0.9, 6.221437, 27334
0.4, 0.9, 6.249654, 27381
0.4, 0.9, 6.190822, 27376

