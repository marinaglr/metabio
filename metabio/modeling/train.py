import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC

import pickle
from imblearn.over_sampling import SMOTENC

from metabio.modeling.evaluate import calc_mean_results, evaluate_test_set
from metabio.data_prep.split_data import create_CV_splits

MAIN_PATH = Path(__file__).parent.parent.parent
MODEL_PATH = f'{MAIN_PATH}/models/'
CROSSVALIDATION_PATH = f'{MAIN_PATH}/results/crossvalidation/'
assert os.path.isdir(MODEL_PATH), f'Model path does not exist.'
assert os.path.isdir(CROSSVALIDATION_PATH), f'Results/crossvalidation path does not exist.'


def train_fold(X_train_df, X_test_df, y_train, method, splitter, feat_sel, oversampling, endpoint, cv_count, grid_search=True, num_trees=500):
    '''
    Fit model using the defined method optimizing the hyperparameters with gridsearch.
    Input:
    ------
    method: str - RF, KNN, GB or SVM
    X_train_df: df - input features from training set
    X_test_df: df - input features from test set
    y_train: array - class labels from training set
    splitter: splitter object or int - splitter for the inner CV; if int -> number of folds in a (Stratified)KFold
    feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
    oversampling: bool - perform oversampling with SMOTENC
    endpoint: str - toxicological endpoint (used to annotate file with selected columns from feature selection)
    cv_count: int - fold number
    grid_search: bool - perform a grid search on the RF model to optimize the hyperparameters (for the rest of methods a grid search is always included)
    num_trees: int - number of trees of RF (only if grid_search==False)
    
    Output:
    -------
    trained model with optimized hyperparameters
    '''
    X_train = X_train_df.values
    ### Feature selection with lasso
    if feat_sel == True:                    
        X_train_df, X_test_df, lasso_coefficients = feat_selection(X_train_df, y_train, X_test_df, 
                                                splitter=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2020), endpoint=endpoint, count=cv_count)
        X_train = X_train_df.values
    
    ### Oversampling with SMOTEC
    if oversampling == True:
        X_train_df, y_train = oversample(X_train_df, y_train)
        X_train = X_train_df.values

    if method == 'RF':
        if grid_search == True:
            param_grid = {'n_estimators': [400,700,1000], 'class_weight': ['balanced'], 'min_samples_leaf': [1,2,3]}
            grid = GridSearchCV(RandomForestClassifier(random_state=2020), param_grid=param_grid, refit=True, cv=splitter, scoring='f1_macro', n_jobs=4)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = RandomForestClassifier(n_estimators=num_trees, class_weight='balanced', min_samples_leaf=3, random_state=2020, n_jobs=4)

    elif method == 'KNN':
        param_grid = {'n_neighbors': [3,5,8], 'weights': ['uniform', 'distance']}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, refit=True, cv=splitter, scoring='f1_macro', n_jobs=4)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
    elif method == 'GB':
        param_grid = {'n_estimators': [200,400,600], 'min_samples_leaf': [1,2,3], 'learning_rate': [0.1, 0.01]}
        grid = GridSearchCV(GradientBoostingClassifier(random_state=2020), param_grid=param_grid, refit=True, cv=splitter, scoring='f1_macro', n_jobs=8)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
    elif method == 'SVM':
        param_grid = {'C': [0.1, 1, 10], 'class_weight': ['balanced'], 'gamma': ['scale', 'auto', 1, 0.01, 0.1]}
        grid = GridSearchCV(SVC(probability=True, random_state=2020), param_grid=param_grid, refit=True, cv=splitter, scoring='f1_macro', n_jobs=4)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

    model.fit(X_train, y_train)
    return model, X_test_df


def feat_selection(X_train_df, y_train, X_test_df, splitter, endpoint, count, feature_set='biotransf_fingerprint'):
    '''
    Perform feature selection with a lasso model.
    Input:
    ------
    X_train_df: df - input features from training set
    y_train: array - class labels from training set
    X_test_df: df - input features from test set
    splitter: splitter object or int - splitter for the inner CV; if int -> number of folds in a (Stratified)KFold
    endpoint: str - toxicological endpoint (used to annotate file with selected columns from feature selection)
    cv_count: int - fold number

    Output:
    -------
    X_train_df and X_test_df: df with selected features
    lasso_coefficients - df containing the column names and the assigned lasso coefficients
    '''
    model_sel = LassoCV(cv=splitter, random_state=2020, n_jobs=4, max_iter=1000000, normalize=False)
    model_sel.fit(X_train_df.values, y_train)

    selected_feat = []
    for i in range(len(model_sel.coef_)):
        if model_sel.coef_[i] != 0:
            selected_feat.append(X_train_df.columns[i])

    # Store selected columns to filter them when applying the model on new compounds
    pd.DataFrame(selected_feat, columns=['Feature']).to_csv(f'{MODEL_PATH}_{endpoint}_{feature_set}_{count}_columns.csv', index=False)

    #selected_feat = X_train_df.columns[(model_sel.coef_ > 0)]
    print('total features: {}'.format((X_train_df.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print('features with coefficients shrank to zero: {}'.format(np.sum(model_sel.coef_ == 0)))

    # Save coefficients
    coeff = pd.DataFrame({'Coefficients': abs(model_sel.coef_)})
    lasso_coefficients = pd.DataFrame({'Feature': X_train_df.columns, 'Coefficient': coeff['Coefficients'].values})
    
    X_train_df = X_train_df[selected_feat]
    X_test_df = X_test_df[selected_feat]
    
    return X_train_df, X_test_df, lasso_coefficients

def oversample(X_train_df, y_train, sampling_strategy=0.8):
    '''
    Apply oversampling with SMOTENC on a set of categorical and numerical features.
    Input:
        X_train_df: df - input features from training set
        y_train: array - class labels from training set
    
    Output:
        X_train_df and y_train with oversampled data.
    '''
    X_train = X_train_df.values
    # Get location of fingerprint columns (not just name of columns)
    cat_columns=[]
    col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower() or 'bit_' in c.lower() or 'transf' in c.lower() or 'Num' in c or 'Count' in c or '_fr_' in c]]
    for i in col:
        cat_columns.append(X_train_df.columns.get_loc(i))

    sm = SMOTENC(categorical_features=cat_columns, sampling_strategy=sampling_strategy)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_train_df = pd.DataFrame(X_train)
    
    return X_train_df, y_train


def assert_paths():
    if os.path.isdir(MODEL_PATH) == False:
        os.mkdir(MODEL_PATH)    
    if os.path.isdir(MODEL_PATH) == False:
        os.mkdir(MODEL_PATH)
    if os.path.isdir(CROSSVALIDATION_PATH) == False:
        os.mkdir(CROSSVALIDATION_PATH)  


def train_models(data_X_df, class_y, smiles, cv_folds, endpoint, desc_type, method='RF',
                feat_sel=False, oversampling=False, grid_search=False, num_trees=500):
    '''
    Train models within a crossvalidation and save internal predictions
    Input:
        data_X_df: dataframe - input dataframe with descriptors
        class_y: list - with the class values
        smiles: list - with the smiles matching the class_y array
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        desc_type: str - chem or metab
        num_trees: int - number of trees of RF (only if grid_search==False)
        method: str - RF, KNN, GB or SVM
        feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
        oversampling: bool - perform oversampling with SMOTENC
        grid_search: bool - perform a grid search on the RF model to optimize the hyperparameters (for the rest of methods a grid search is always included)
    
    Output:
        return: models
        print: file '{CROSSVALIDATION_PATH}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv' with cross-validation p-values and prediction
    '''
    ### Initialization
    data_X = data_X_df.values
    all_results = pd.DataFrame()
    assert_paths()
    
    ### Split data for crossvalidation
    sss = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=2020)
    models_cv = [] # Save all models to use for further prediction
    count = 1

    ### Prepare output file
    cv_outfile = open(f'{CROSSVALIDATION_PATH}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv', 'w')
    header = f'name,{endpoint},probability,prediction,smiles,cv\n'
    cv_outfile.write(header)

    ### Variance filter
    n_cols_1 = len(data_X_df.columns)
    selector = VarianceThreshold(0.001)
    selector.fit_transform(data_X_df)
    data_X_df = data_X_df[data_X_df.columns[selector.get_support(indices=True)]]
    data_X = data_X_df.values
    n_cols_2 = len(data_X_df.columns)
    print(f'Variance filter removed {n_cols_1-n_cols_2} columns')
    pd.DataFrame(data_X_df.columns, 
                    columns=['Feature']).to_csv(f'{MODEL_PATH}/model_{endpoint}_{desc_type}_parent_variance_filter_columns.csv', index=False)

    ### Normalize all descriptors
    scaler = StandardScaler()
    data_X = scaler.fit_transform(data_X_df)
    data_X_df = pd.DataFrame(data_X, columns=data_X_df.columns)
    data_X_df.reset_index(drop=True, inplace=True)
    print(data_X_df.shape)
    # Save the scaler
    if os.path.isdir(f'{MODEL_PATH}/normalizer/') == False:
        os.mkdir(f'{MODEL_PATH}/normalizer/')
    scalerfile = f'{MODEL_PATH}/normalizer/model_{endpoint}_{desc_type}_parent.pkl'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    
    ### Fit model within crossvalidation and make prediction for respective test set
    for train_index, test_index in sss.split(data_X, class_y):
        print(f'Training CV {count}...')
        ### Prepare the data splits
        X_train, X_test = data_X[train_index], data_X[test_index]
        y_train, y_test = class_y[train_index], class_y[test_index]
        smiles_train, smiles_test = smiles[train_index], smiles[test_index]

        X_train_df = pd.DataFrame(X_train, columns=data_X_df.columns)
        X_test_df = pd.DataFrame(X_test, columns=data_X_df.columns)

        # Train model with selected method. Feature selection and oversampling on training data may be included prior to model training.
        model, X_test_df = train_fold(X_train_df, X_test_df, y_train, method, sss, feat_sel, oversampling, endpoint, cv_count=count, grid_search=grid_search, num_trees=num_trees)
        X_test = X_test_df.values
        
        # Make predictions for test set, append the results to df (returned) and write out the predictions in cv_outfile
        all_results = evaluate_test_set(model, X_test, y_test, smiles_test, all_results, count, write_prediction=True, outfile=cv_outfile)

        # Collect models for later predictions of external test data
        models_cv.append(model)

        ### Save the model to disk
        filename = f'{MODEL_PATH}/model_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_parent_{count}.sav'
        pickle.dump(model, open(filename, 'wb'))

        count += 1
        
    ### Calculate mean results on test set
    mean_results = calc_mean_results(all_results)
    print(mean_results)         
    
    cv_outfile.close()

    # return trained models
    return models_cv, mean_results
    
def train_models_from_CV_files(cv_folds, endpoint, desc_type, method='RF', feat_sel=False, oversampling=False, 
                                grid_search=False, num_trees=500, splits_path=f'{MAIN_PATH}/data/splits/'):
        
    '''
    Train models within a crossvalidation -> CV files are loaded (and not directly calculated)
    Internal predictions are saved
    Input:
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        desc_type: str - chem or metab
        num_trees: int - number of trees of RF (if grid_search==False)
        method: str - RF, KNN, GB or SVM
        feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
        oversampling: bool - perform oversampling with SMOTENC
        grid_search: bool - performa a grid search on the RF model to optimize the hyperparameters
        splits_path: str - path to the folder where the training and test set splits are saved
    
    Output:
        return: models
        print: file '{MAIN_PATH}/results/crossvalidation/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv' with cross-validation p-values and prediction
    '''
    ### Initialization
    all_results = pd.DataFrame()
    models_cv = [] # Save all models to use for further prediction
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2020)
    assert_paths()

    ### Prepare output file
    cv_outfile = open(f'{CROSSVALIDATION_PATH}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv', 'w')
    header = f'name,{endpoint},probability,prediction,smiles,cv\n'
    cv_outfile.write(header)
    
    ### Train CV models
    for count in range(1, cv_folds+1):
        print(f'Training CV {count}...')
        X_train_df = pd.read_csv(f'{splits_path}/{endpoint}_trainingset_parent_{count}.csv')
        y_train = X_train_df[endpoint].values
        smiles_train = X_train_df['SMILES (Canonical)'].values
        X_train_df = X_train_df.drop(['SMILES (Canonical)', endpoint], axis=1)
        if 'parent_SMILES' in X_train_df:
            X_train_df.drop(['parent_SMILES'], axis=1, inplace=True)

        X_test_df = pd.read_csv(f'{splits_path}/{endpoint}_testset_parent_{count}.csv')
        y_test = X_test_df[endpoint].values
        smiles_test = X_test_df['SMILES (Canonical)'].values
        X_test_df = X_test_df.drop(['SMILES (Canonical)', endpoint], axis=1)
        
       # Train model with selected method. Feature selection and oversampling on training data may be included prior to model training.
        model, X_test_df = train_fold(X_train_df, X_test_df, y_train, method, sss, feat_sel, oversampling, endpoint, cv_count=count, grid_search=grid_search, num_trees=num_trees)
        X_test = X_test_df.values
        
        # Make predictions for test set, append the results to df (returned) and write out the predictions in cv_outfile
        all_results = evaluate_test_set(model, X_test, y_test, smiles_test, all_results, count, write_prediction=True, outfile=cv_outfile)

        # Collect models for later predictions of external test data
        models_cv.append(model)

        ### Save the model to disk
        filename = f'{MODEL_PATH}/model_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_parent_{count}.sav'
        pickle.dump(model, open(filename, 'wb'))

        count += 1
        
    ### Calculate mean results on test set
    mean_results = calc_mean_results(all_results)    
    print(mean_results)

    cv_outfile.close()

    # Return trained models
    return models_cv, mean_results

def train_models_metabolite_label(parents_df, metabolites_df, cv_folds, endpoint, endpoint_col, desc_type, create_splits=False, method='RF', 
                                  feat_sel=False, oversampling=False, grid_search=False, num_trees=500):
    
    '''
    Train models including labeled metabolites in the training set.
    Input:
        parents_df: df - input dataframe of parent compounds with descriptors
        metabolites_df: df - input dataframe of metabolites with descriptors (and labels for training)
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        endpoint_col: str - name of the column containing the class label
        desc_type: str - chem or metab
        method: str - RF, KNN, GB or SVM
        feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
        oversampling: bool - perform oversampling with SMOTENC
        grid_search: bool - performa a grid search on the RF model to optimize the hyperparameters
        num_trees: int - number of trees of RF (if grid_search==False)
    
    Output:
        return: models
        print: file '{MAIN_PATH}/results/crossvalidation/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}.csv' with cross-validation p-values and prediction
    '''
    ### Initialization
    all_results = pd.DataFrame()
    models_cv = [] # Save all models to use for further prediction
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2020)
    assert_paths()

    ### Prepare output file
    cv_outfile = open(f'{CROSSVALIDATION_PATH}/crossvalidation_{endpoint}_{method}_{desc_type}_featSel={feat_sel}_metab.csv', 'w')
    header = f'name,{endpoint},probability,prediction,smiles,cv\n'
    cv_outfile.write(header)
    
    ### Create CV splits if the files don't exist already
    if create_splits == True:
        create_CV_splits(parents_df, metabolites_df, cv_folds, endpoint, endpoint_col)
    
    ### Train CV models
    for count in range(1, cv_folds+1):
        print(f'Training CV {count}...')
        X_train_df = pd.read_csv(f'{MAIN_PATH}/data/splits/{endpoint}_trainingset_metab_{count}.csv')
        y_train = X_train_df[endpoint].values
        smiles_train = X_train_df['SMILES (Canonical)'].values
        X_train_df = X_train_df.drop(['SMILES (Canonical)', endpoint], axis=1)

        X_test_df = pd.read_csv(f'{MAIN_PATH}/data/splits/{endpoint}_testset_metab_{count}.csv')
        y_test = X_test_df[endpoint].values
        smiles_test = X_test_df['SMILES (Canonical)'].values
        X_test_df = X_test_df.drop(['SMILES (Canonical)', endpoint], axis=1)
        
        # Train model with selected method. Feature selection and oversampling on training data may be included prior to model training.
        model, X_test_df = train_fold(X_train_df, X_test_df, y_train, method, sss, feat_sel, oversampling, endpoint, cv_count=count, grid_search=grid_search, num_trees=num_trees)
        X_test = X_test_df.values
        
        # Make predictions for test set, append the results to df (returned) and write out the predictions in cv_outfile
        all_results = evaluate_test_set(model, X_test, y_test, smiles_test, all_results, count, write_prediction=True, outfile=cv_outfile)

        # Collect models for later predictions of external test data
        models_cv.append(model)

        ### Save the model to disk
        filename = f'{MODEL_PATH}/model_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_metab_{count}.sav'
        pickle.dump(model, open(filename, 'wb'))

        count += 1
        
    ### Calculate mean results on test set
    mean_results = calc_mean_results(all_results)       
    print(mean_results)  

    cv_outfile.close()
    
    return models_cv, mean_results


if __name__ == '__main__':
    
    data_X_df = pd.read_csv('../data/splits/MNT_trainingset_metab_1.csv')
    class_y = data_X_df['MNT'].values
    smiles = data_X_df['SMILES (Canonical)'].values
    X_df = data_X_df.drop(['MNT','SMILES (Canonical)'], axis=1)

    train_models(X_df, class_y, smiles, cv_folds=5, endpoint='MNT', desc_type='chem', num_trees=500, method='RF',
               feat_sel=False, oversampling=True, grid_search=True)

    #train_models_from_CV_files(cv_folds=5, endpoint='MNT', desc_type='chem', num_trees=500, method='RF',
    #                            feat_sel=False, oversampling=False, grid_search=True, splits_path='../data/splits')