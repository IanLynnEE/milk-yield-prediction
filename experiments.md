
## Different features

### ANN

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

#### Original features

| feature   | preprocess        |
|-----------|-------------------|
| year      |                   |
| month     |                   |
| ranch     | lable encode      |
| serial    | based on birthday |
| father    | frequency encode  |
| mother    | frequency encode  |
| delivery  |                   |
| lactation |                   |
| age       |                   |
| breeding  |                   |

Average RMSE on random validation sets: 5.879
RMSE on a special validation set: 6.138

#### Use frequency encoded serial

Average RMSE on random validation sets: 5.666
RMSE on a special validation set: 5.913

Looks good, we will use frequency encoded serial for following test.

#### Use one-hot encoded ranch as well

Average RMSE on random validation sets: 5.650
RMSE on a special validation set: 5.869

Looks good, we will use one-hot encoded ranch for following test.

#### Without breeding (col 18):

Average RMSE on random validation sets: 5.774
RMSE on a special validation set: 5.853

#### Add deliverySeason

Average RMSE on random validation sets: 5.632
RMSE on a special validation set: 6.187
Real RMSE on the test set: 6.2737843 with max_iters=50

Looks good, we will Add deliverySeason for following test.

We found out max_iters is too high only now. We will still use max_iters = 300 for next test, for better comparison.

#### Add some time intervals

Average RMSE on random validation sets: 5.373
RMSE on a special validation set: 6.037

However, some test data cannot compute time intervals.

We merge the result from `Add deliverySeason` with this, and get: 

Real RMSE on the test set: 6.2570669 with max_iters=50
