# Project: Binary Classification of Machine Failures


## Data Dictionary: Binary Classification of Machine Failures

This data dictionary describes the columns in the dataset used for binary classification of machine failures. The goal is to predict whether a machine will fail based on various operational parameters and failure flags.

| **Column Name**          | **Data Type** | **Description**                                                                                                | **Units**      | **Possible Values/Range**                                  | **Relevance to Machine Failure**                                                                    | **Potential Issues & Considerations**                                                                                                                      |
|--------------------------|---------------|--------------------------------------------------------------------------------------------------------------------|----------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **id**                   | `int64`       | Unique identifier for each data point (observation/row).                                                      | -              | Unique integer values.                                    | **Not directly predictive**, primarily for indexing and record identification.                    | Should be unique for all rows. Not to be used as a feature for model training directly unless indexing or time-series context is important (which is not indicated). |
| **Product ID**           | `object`      | Identifier for the product/machine being monitored. Could represent different machine types or serial numbers. | -              | Alphanumeric or string values (e.g., "L47181", "M29487", etc.). | May indicate **different machine models or batches** which could have varying failure characteristics. | Needs to be checked for cardinality. High cardinality might require dimensionality reduction or feature engineering techniques (like grouping or encoding).  Consider if it's ordinal or nominal.  Check for consistency in naming conventions. |
| **Type**                 | `object`      | Type of the machine or process.  Might represent different categories of machines or operational modes.       | -              | Categorical string values (e.g., "Type H", "Type M", "Type L").                                | Could indicate **inherent differences in reliability or operational stresses** based on machine type. | Check the distinct categories and their distribution. Consider encoding methods (One-Hot Encoding, Label Encoding) for machine learning models. Check for class imbalance across different types. |
| **Air temperature [K]**  | `float64`     | Air temperature surrounding the machine during operation.                                                     | Kelvin (K)     | Continuous positive values (typically in a realistic range for industrial environments, e.g., 290K - 310K). | **Environmental factor** that can influence machine component temperatures and performance. Higher temperatures can lead to increased stress and wear. | Check for realistic temperature ranges and potential outliers (sensor errors). Consider conversion to Celsius if more interpretable for some audiences.  May have seasonal or daily patterns. |
| **Process temperature [K]**| `float64`     | Process temperature of a specific part or component within the machine.                                        | Kelvin (K)     | Continuous positive values, typically higher than air temperature as it reflects internal machine operation. | **Direct indicator of machine operating condition.** Elevated process temperature can be a sign of overheating or inefficiency, potentially leading to failure. | Compare to air temperature; a significant difference might be important. Check for outliers and realistic operating ranges for the process. Potential correlation with 'Air temperature [K]'. |
| **Rotational speed [rpm]** | `int64`       | Rotational speed of a key machine component (e.g., spindle, motor shaft).                                  | Revolutions per minute (rpm) | Integer values, likely within a defined operational range for the machine type.                  | **Operational parameter directly related to machine stress and wear.**  Excessive or inconsistent speed can contribute to mechanical failures and fatigue. | Check for typical operating speed ranges and any unexpected values. Consider potential non-linear relationships with failure.  May be interactions with 'Torque [Nm]'. |
| **Torque [Nm]**          | `float64`     | Torque applied by the machine or on a specific component. Represents the rotational force.                      | Newton-meters (Nm) | Continuous positive values, typically within a machine's operational capacity.                      | **Indicates load and stress on the machine.** High torque can lead to strain and potential failure of mechanical parts. | Check for typical torque ranges and potential overload situations (unusually high values).  Consider relationships with 'Rotational speed [rpm]' and 'Tool wear [min]'. |
| **Tool wear [min]**        | `int64`       | Cumulative tool wear measured in minutes.  Represents the amount of wear and tear on a cutting or working tool. | Minutes (min)  | Non-negative integer values, increasing over time or usage.                                  | **Direct measure of tool degradation.**  Excessive tool wear can lead to decreased performance, quality issues, and eventually machine failure if not addressed. | Tool wear should generally be monotonically increasing. Look for resets or anomalies. Consider the relationship between tool wear and maintenance schedules. |
| **Machine failure**      | `int64`       | **Target variable**. Binary indicator of machine failure.                                                              | -              | Binary values: `0` (No failure), `1` (Failure).                                       | **The variable to be predicted.** Indicates whether a machine has experienced a failure event during the observation period. | Understand the definition of "failure" in this context (e.g., critical breakdown, performance degradation beyond a threshold). Check for class imbalance (ratio of failures to non-failures). |
| **TWF**                  | `int64`       | Tool Wear Failure flag. Indicates if a failure occurred specifically due to tool wear.                               | -              | Binary values: `0` (No TWF), `1` (TWF).                                              | **Specific type of failure indicator.** May provide insights into root causes of machine failures.  Could be correlated with 'Tool wear [min]' and 'Machine failure'. |  Check correlation with 'Tool wear [min]' and 'Machine failure'. Understand if TWF is a *cause* of 'Machine failure' or a *subtype*. |
| **HDF**                  | `int64`       | Heat Dissipation Failure flag. Indicates if a failure occurred due to issues related to heat dissipation (e.g., overheating). | -              | Binary values: `0` (No HDF), `1` (HDF).                                              | **Specific type of failure indicator.**  Related to temperature parameters.  Could be correlated with 'Air temperature [K]' and 'Process temperature [K]'. | Check correlation with temperature columns and 'Machine failure'. Similar to TWF, understand its causal relationship. |
| **PWF**                  | `int64`       | Power Failure flag. Indicates if a failure occurred due to power related issues (e.g., power surge, power loss).        | -              | Binary values: `0` (No PWF), `1` (PWF).                                              | **Specific type of failure indicator.**  May point to electrical system problems.                                 | Consider if there are other power-related features not explicitly listed. Understand its correlation with 'Machine failure'. |
| **OSF**                  | `int64`       | Overstrain Failure flag. Indicates if a failure occurred due to excessive strain or stress on machine components. | -              | Binary values: `0` (No OSF), `1` (OSF).                                              | **Specific type of failure indicator**, Related to mechanical stress and load. May be linked to 'Torque [Nm]' and 'Rotational speed [rpm]'. | Check correlation with 'Torque [Nm]', 'Rotational speed [rpm]', and 'Machine failure'.  Understand the definition of "overstrain" in this context. |
| **RNF**                  | `int64`       | Random Failure flag. Indicates if a failure occurred due to random or unexplained reasons, or if it couldn't be attributed to TWF, HDF, PWF, or OSF. | -              | Binary values: `0` (No RNF), `1` (RNF).                                              | **Specific type of failure indicator**, capturing failures not explained by other specific failure types. May represent unexpected or complex failure modes. |  Understand the criteria for classifying a failure as "Random".  Check if RNF is mutually exclusive with other failure flags or if it can co-occur. |

**Key Considerations for Data Analysis and Machine Learning:**

* **Feature Scaling:** Numerical features like temperature, rotational speed, torque, and tool wear are on different scales. Feature scaling (e.g., Standardization or Min-Max Scaling) will be important for many machine learning algorithms.
* **Categorical Feature Encoding:**  'Product ID' and 'Type' are categorical features and need to be encoded into numerical representations for most machine learning models (e.g., One-Hot Encoding, Label Encoding, or potentially more advanced embeddings depending on cardinality and information content).
* **Correlation and Multicollinearity:** Check for correlations between features, especially between 'Air temperature [K]' and 'Process temperature [K]', and between 'Rotational speed [rpm]' and 'Torque [Nm]'. High multicollinearity might affect model stability and interpretability.
* **Class Imbalance:**  Machine failure datasets often exhibit class imbalance (fewer failures than non-failures). Techniques to handle class imbalance (e.g., oversampling, undersampling, SMOTE, class weights in models) might be necessary.
* **Feature Interactions:** Consider exploring interactions between features. For example, the combination of high 'Process temperature [K]' and high 'Torque [Nm]' might be a stronger predictor of failure than either feature alone.
* **Time Dependency (If Applicable):** While not explicitly stated, if the data is collected over time, consider if there is any temporal dependency or trends that might be important (though 'id' being just an integer might suggest rows are independent snapshots).
* **Failure Type Flags:** The flags TWF, HDF, PWF, OSF, and RNF provide valuable insights into failure modes. Analyze their distributions, correlations with other features, and their relationship to the target variable 'Machine failure'.  Determine if they are mutually exclusive or can co-occur.
* **Domain Knowledge:** Leveraging domain knowledge about machine operation and failure modes will be crucial for feature engineering, model selection, and interpreting results.

This data dictionary provides a solid foundation for understanding and working with the dataset "Binary Classification of Machine Failures" dataset.

## Data Analysis Results

## Descriptive Statistics

The following table shows descriptive statistics of the features presented in the dataset.

![logo](images/1_statistics.png)

## Class imbalance

The next figure shows the proportion of each class, 0 being the *normal functioning state* and 1  the *failure state*, for all the instances dataset, evidencing a highly imbalanced dataset, where the majority class is more than 60 times prevalent than the minority one.

![logo](images/2_classimbalance.png)

## Class imbalance as a function of type...

The following figure depicts the class imbalance, displaying the number of instances, for each kind of...

![logo](images/3_classimbalance_types.png)

## Machine failure proportion by each kind of type...

The following figure shows the machine failure proportion according to each kind of type...

![logo](images/4_machinefailure_type.png)

## Boxplot of features according to Machine Failure

The following figure shows a boxplot of selected features according to Machine Failure.


![logo](images/5_processtemperature_failure.png)

## Violinplot of features according to Machine Failure

The following figure shows a violinplot of selected features according to Machine Failure.

![logo](images/6_airtemperature_failure.png)

## Boxplot of features according to the kind of...

The next figure shows a boxplot of three selected features according to the kind of...


![logo](images/7_process_temperature_type.png)

## Violinplot of features according to kind of...

The following figure shows a violinplot of features according to kind of...

![logo](images/8_airtemperature_violin.png)

## Violin plot for numeric features according to Machine Failure

The next figure shows a violin plot for numeric features according to Machine Failure attribute.

![logo](images/9_violin_features_failure.png)

## Violin plot for numeric features according to kind of

The following figure shows a violin plot for numeric features according to kind of

![logo](images/10_violin_features_type.png)

## Correlation Matrix

The next figure shows the correlation matrix comprising all the features present in the dataset, except the target variable (failure or not), ID, product ID, and type... The highly correlated features were dropped from this dataset.

![logo](images/11_corrmatrix.png)

## Top 20 Product's ID with the largest number of instances of failures 

The next figure shows the top 20 Product's ID with the largest number of instances of failures.

![logo](images/12_productID_count.png)

## Product's IDs with the Highest Average Failure Rate

The next figure shows the top 20 Product's ID with the highest average failure rates. There are only two product's IDS with 100% average failure rate, and only two with 75% average failure rate.


![logo](images/13_productID_highestaverage_failure.png)

## Product's IDs with the least number of failuers

The next figure shows the Product's IDs with the least number of failuers, but different than zero.

![logo](images/14_productID_leastfailures_differentthanzero.png)

## Product's IDs with the Highest Count of Failure's Rate

The following figure shows the Product's IDs with the Highest Count of Failure's Rate, showing 14 Product's IDs have a total number of failure counts totaling more than 100.

![logo](images/15_productID_highestcountfailure.png)

## Machine Learning Modeling Results

The next figure shows the results on the testing set by using a XGBoost Classifier, using stratification of the classes and a shuffle split(**StratifiedShuffleSplit**), by considering the number of splits equal to 2. The f1-score of the majority is perfect (1.0), and for the minority class there is a downgrade in the performance regarding the recall metrics (around 0.78), showing that in some situations there could be a failure instance being classified as a normal instance. However, **the AUC metrics lays around 90%, which evidences a good overall performance for this classification problem**.

![logo](images/16_modelresults.png)

## Summary of the Findings


## Acknowledgements


## MIT License

Copyright (c) [2025] [Dr. Vagner Zeizer Carvalho Paes]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
