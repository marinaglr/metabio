import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, roc_auc_score


def calc_scores(y_test, y_test_pred, y_pred_prob=[], silent=False):
        
    ### Calculate scores
    recall = recall_score(y_test, y_test_pred, average='macro', pos_label=1)
    precision = precision_score(y_test, y_test_pred, average='macro', pos_label=1)
    f1 = f1_score(y_test, y_test_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)

    if y_pred_prob != []:
        auc = roc_auc_score(y_test, y_pred_prob)
        results = pd.DataFrame({"F1 score": [round(f1,3)], "Balanced acc.": [round(bal_acc,3)], "MCC": [round(mcc,3)], 
                            "Precision": [round(precision,3)], "Recall": [round(recall,3)], "AUC": [round(auc,3)]})
    else:
        auc = ""
        results = pd.DataFrame({"F1 score": [round(f1,3)], "Balanced acc.": [round(bal_acc,3)], "MCC": [round(mcc,3)], 
                                "Precision": [round(precision,3)], "Recall": [round(recall,3)], "AUC": [auc]})
    if silent == False:
        print(f"Performance on test set:")
        print(results)
    return  results
    
def calc_mean_results(all_results):
    
    mean_results = pd.DataFrame()
    
    mean_df = pd.DataFrame([all_results.mean(axis=0)], columns=all_results.columns)
    std_df = pd.DataFrame([all_results.std(axis=0)], columns=all_results.columns)

    for i in all_results.columns:
        mean_results[i] = mean_df[i]
        mean_results[f"{i}_std"] = std_df[i]
        
    return mean_results

def evaluate_model_CV(endpoint, desc_type, method, crossvalidation_output_path, feat_sel=False, metab_labels=False):
    """
    Calculate mean and std from the p-values calculated for the test set and stored in the crossvalidation output file
    
    Parameters:
    ----------
    endpoint: str - name of the endpoint
    desc_type: str - chem or metab
    method: str - RF, KNN, GB or SVM
    crossvalidation_output_path: str - path where the predictions on the crossvalidation sets are stored
    feat_sel: bool - use results from models including feature selection prior to model training
    metab_labels: bool - use results from models including labeled metabolites in the training set
    
    Output:
    -------
    return: dataframe containing the mean evaluation results
    """
    
    # Load output dataframe with p-values
    if metab_labels == True:
        cv_filename = f'{crossvalidation_output_path}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}_metab.csv'
        
    else:
        cv_filename = f'{crossvalidation_output_path}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv'
    cv_results = pd.read_csv(cv_filename)
    cv_numbers = cv_results.cv.unique()
    
    # Print performance on test set
    all_results = pd.DataFrame()
    for i in cv_numbers:
        results_df_cv = cv_results[cv_results["cv"] == i]
        result = calc_scores(results_df_cv[endpoint].values, results_df_cv["prediction"].values, 
                                y_pred_prob=results_df_cv["probability"].values, silent=True)
        all_results = pd.concat([all_results, result])
        
    ### Calculate mean results on test set
    mean_results = calc_mean_results(all_results)
    
    return mean_results


def evaluate_test_set(model, X_test, y_test, smiles_test, all_results, cv_count, write_prediction=True, outfile=''):
    '''
    Make predictions for test set
    Input:
        model: trained model to be evaluated
        X_test: array - input features from test set
        y_test: array - class labels from training set
        smiles_test: array - smiles from compounds in test set
        outfile: file (in writting mode) to write out the predictions of the test set
        all_results: df - df to append the evaluation results of the current fold
        cv_count: int - fold number
    
    Output:
        all_results: df - concatenated evaluation results over all folds
        outfile with predictions made on test set
    '''
    pred_class = model.predict(X_test)
    temp = pd.DataFrame(model.predict_proba(X_test), columns=['0', '1'])
    pred_prob = temp['1'].values #column with the probability of class 1            
    
    # Print performance on test set
    result = calc_scores(y_test, pred_class, y_pred_prob=pred_prob, silent=True)
    all_results = pd.concat([all_results, result])
    
    if write_prediction:
        # Write prediction and further compound information for each split into output file
        long_line = ''
        for n, i in enumerate(X_test): 
            newline = f'{n}, {y_test[n]}, {pred_prob[n]}, {pred_class[n]}, {smiles_test[n][0]}, {cv_count}\n'
            long_line += newline
        outfile.write(long_line)
    return all_results

