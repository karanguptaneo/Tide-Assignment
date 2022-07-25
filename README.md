
# Report For the Receipt Matcher

This Project Aims to build a model based pipeline which takes a receipt data and order them with the most likely to be correct at the top of
the list.

### Below is the Approach
* Firstly loaded the Model into a Pandas Dataframe(Tabular Structure)
* Then checked if there is any Missing Values or not (no missing value found).
* Creating a target variable which tells us is -matched_transaction_id == 'feature_transaction_id or not 
* Then splitting the data representing Train , CV , Test data for their Respective purposes.
* Then Checking the Visualization of the with PCA
* Checking for Imbalancing of data (founded Imbalancing Balanced with Oversampling less dominating class).
* Applied Diferrent Models with the following Metric's

 #### Model Reference

| Model Name          |Test Loss|Test mAP|Cost| Interpretability|
|---------------------|---------|--------|----|------|
| Logistic Regression |0.223    |0.73    |less|more  |
| Decision Tree       |0.242    |0.64    |less|more  |
| Random Forest       |0.216    |0.73    |more|less  |

* If We only care about performance we can go with Random Forest , but considering Other aspects let's go forward with Logistic Regression for now.

* Comming to feature importance DateMappingMatch , DifferentPredictedDate Seems to be the top 2 features.

![Screenshot](feature_importance.png)

* Two files are included in the Repo train.py and predict.py


## Usage
### jupyter notebook
##### for the complete jupyter notebook please look at 'matching_notebook.ipynb'

###

### Using train.py (for training and saving the pipeline(model) )
```python

 Please give the input in the following order - 
    -------------------------------------------------------------------------------------------------------
    receipt_id ,company_id ,feature_transaction_id
    DateMappingMatch, AmountMappingMatch, DescriptionMatch,DifferentPredictedTime, TimeMappingMatch, 
    PredictedNameMatch,ShortNameMatch, DifferentPredictedDate, PredictedAmountMatch,PredictedTimeCloseMatch 
    -------------------------------------------------------------------------------------------------------
from predict import predict_and_rank

sample = [
    ['40' , 'company_1' , 'xyz' , 0.1,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.1] ,
    ['40' , 'company_1' , 'abc' , 0.1,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.2] , 
    ['40' , 'company_1' , 'qwe' , 0.2,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.2]
    ]
# load the saved pipeline(model)
with open('log_regress.sav' , 'rb') as model:
        pipeline = pickle.load(model)

output_ = predict_and_rank(pipeline , sample)
```