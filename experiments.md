# Topic

## Abstract


## Introduction



## Methods

### Data Modeling

<!-- Def features' name and meaning here or at result? -->


### Select features

Def:
- Average RMSE: Average RMSE on 10 random validation sets.
- Special RMSE: Use data after 2018/06/30 as validation set.
- Real RMSE: Set `max_iter=50` instead of 300.
<!-- Define how does "add time interval" work here?-->

### Select models' parameters




## Result

### Data Modeling

#### Dropping Data


#### Encoding




### Select features

#### ANN

model:
```python
MLPRegressor(
    hidden_layer_sizes=(150,100,50),
    max_iter = 300,
    activation = 'relu',
    solver = 'adam'
)
```

features scaler: `MinMaxScaler`

<!-- Should this table be defined ealier? -->
| feature   | preprocessing     |
|-----------|-------------------|
| year      |                   |
| month     |                   |
| ranch     | lable encoded     |
| serial    | based on birthday |
| father    | frequency encoded |
| mother    | frequency encoded |
| delivery  |                   |
| lactation |                   |
| age       |                   |
| breeding  |                   |



|                          | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | Test 6 |
|--------------------------|--------|--------|--------|--------|--------|--------|
| Ascending serial         | O      | X      | X      | X      | X      | X      |
| Frequency encoded serial | X      | O      | O      | O      | O      | O      |
| One-hot encoded ranch    | X      | X      | O      | O      | O      | O      |
| Add breeding             | O      | O      | O      | X      | O      | O      |
| Add delivery season      | X      | X      | X      | X      | O      | O      |
| Add time interval        | X      | X      | X      | X      | X      | O      |
|                          |        |        |        |        |        |        |
| Average RMSE             | 5.879  | 5.666  | 5.650  | 5.774  | 5.632  |        |
| Special RMSE             | 6.138  | 5.913  | 5.869  | 5.853  | 6.187  |        |
| Real RMSE                |        |        |        |        | 6.274  | 6.257  |



#### Random Forest Regressor

model:
```python
RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    random_state=1,
    n_jobs=-1
)
```

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
| Add time interval   | X      | X      | X      | O      |
|                     |        |        |        |        |
| Average RMSE        | 5.505  | 5.392  | 5.346  |        |
| Special RMSE        | 5.764  | 5.709  | 5.689  |        |
| Real RMSE           | 6.303  | 6.260  | 6.242  | ??     |

