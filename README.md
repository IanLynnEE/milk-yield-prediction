# Milk Yield Prediction

## Preprocessing

There are 33252 training data with 1821 cows, and 4263 test data with 846 cows.

For the 846 cows in the test dataset, 679 of them can be found in training data, while 170 of them cannot.

### Verify Data

The function `verify_data()` gives warnings if any of the following things are violated:

1. The age of a cow should equal to (sampling date - birthday).
2. The sampling date should be no later than the recording date.
3. The number of days of lactation should equal to (sampling date - last delivery date).
4. The birthday recorded in report.csv should agree with the one in birth.csv.
5. A cow should have exactly one birthday.
6. The serial number of a cow should not appear in report.csv after it died.

We recommend to drop following training data:
- ID = 6960: sampling date > recoding date.
- ID = 16714: missing too many features.

The cows with the following serial numbers are recorded to be dead, yet they appear in report.csv as well:
- 98051976
- 94051730
- 97051412

After a quick check, it seems that the serial number will be reused on some occasions. As a result, we will not adopt data in spec.csv before further review.

### Get Features

The function `get_all_features()` returns the training and test set with available features.

Based on the result of `verify_data()`, we adopt the following features:

| Original Column Name | Column Name in Program | Meaning                        | type    |
| -------------------- | ---------------------- | ------------------------------ | ------- |
| 1                    | ID                     | ID                             | int64   |
| 2                    | year                   | year of sampling               | int64   |
| 3                    | month                  | month of sampling              | int64   |
| 4                    | ranch                  | ranch of the cow               | int32   |
| 5                    | serial                 | encoded serial number          | int64   |
| 5                    | father                 | encoded father's serial number | int32   |
| 5                    | mother                 | encoded mother's serial number | int32   |
| 9                    | delivery               | #deliveries                    | int64   |
| 10                   | lactation              | #days of lactation             | int64   |
| 11                   | volume                 | milk yield                     | float64 |
| 14                   | age                    | age in month                   | int64   |
|                      |                        |                                |         |

We only use data in report.csv, as we cannot get the mother's ID (col 6) from birth.csv (no corresponding calf serial can be found). Consequently, we cannot get the father's ID (col 7) from breed.csv.
