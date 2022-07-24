# predict.py
import pickle

def predict_and_rank(model , input_):
    data = input_.copy()
    length = len(data)
    
    # appending probablities of having a match at the end of each row.
    for iter_ in range(length):
        probability =  { 'prob':model.predict_proba( [data[iter_][3:]] )[0][1] }
        data[iter_].append( probability )
    
    # sorting on the Basis of Probability which is appended in each row just previously.
    data.sort(key=lambda a: a[-1]['prob'] ,reverse = True)
    return data

if __name__ == '__main__':
    # please give the input to the above function only in this order
    """" 
    Please give the input in the following order - 
    -------------------------------------------------------------------------------------------------------
    receipt_id ,company_id ,feature_transaction_id
    DateMappingMatch, AmountMappingMatch, DescriptionMatch,DifferentPredictedTime, TimeMappingMatch, 
    PredictedNameMatch,ShortNameMatch, DifferentPredictedDate, PredictedAmountMatch,PredictedTimeCloseMatch 
    -------------------------------------------------------------------------------------------------------
    """
    # just some sample testing here
    sample = [
    ['40' , 'company_1' , 'xyz' , 0.1,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.1] ,
    ['40' , 'company_1' , 'abc' , 0.1,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.2] , 
    ['40' , 'company_1' , 'qwe' , 0.2,0.2,0.1,0.4,0.5,0.1,0.4,0.1 , 0.2 ,0.2]
    ]
    
    with open('log_regress.sav' , 'rb') as model:
        pipeline = pickle.load(model)
    
    print(*predict_and_rank(pipeline , sample) , sep  ='\n')
