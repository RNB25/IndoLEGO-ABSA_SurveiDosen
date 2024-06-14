from typing import List, Dict

"""
The metrics in this research follows the following rule:
1. The definition of true positive is said that an ABSA tuple that exist in the prediction list, also exist in the target list
2. The definition of the false positive is said that an ABSA tuple that exist in the prediction list do not exist in the target list
3. The definition of the false negative is said that an ABSA tuple that exist in the target list do not exist in the prediction list
"""
def peqi():
    return (print("berhasil"))

def lower(preds_or_targets):
    result = str(preds_or_targets)
    result = result.lower()
    return eval(result)

def recall(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        Recall metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * Recall value.
    """
    true_positive = 0
    false_negative = 0
    for prediction,target in zip(lower(predictions),lower(targets)):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
    result = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return result

def precision(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        Precision metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * Precision value.
    """
    true_positive = 0
    false_positive = 0
    for prediction,target in zip(lower(predictions),lower(targets)):
        for prediction_tuple in prediction:
            if prediction_tuple in target:
                true_positive += 1
            else:
                false_positive += 1
    result = true_positive/(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    return result

def f1_score(predictions:List[List[Dict]],targets:List[List[Dict]]) -> float:
    """
    ### DESC
        F1 score metric function for ABSA.
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * F1 score value.
    """
    recall_value = recall(predictions,targets)
    precision_value = precision(predictions,targets)
    result = (2 * recall_value * precision_value)/(recall_value + precision_value) if (recall_value + precision_value) > 0 else 0
    return result

def summary_score(predictions:List[List[Dict]],targets:List[List[Dict]]) -> Dict:
    """
    ### DESC
        Score summary (recall, precision, f1 score).
    ### PARAMS
    * predictions: List of list of prediction dictionary.
    * targets: List of list of target dictionary.
    ### RETURN
    * Score summary in a dictionary form.
    """
    return {
        "recall" : recall(predictions,targets),
        "precision" : precision(predictions,targets),
        "f1_score" : f1_score(predictions,targets)
    }
