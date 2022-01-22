import pandas as pd
import re
import os

from sklearn.model_selection import StratifiedShuffleSplit
from metabio.data_prep.prepare_data import prepare_descriptors


def create_CV_splits(parents_df, metabolites_df, output_path, cv_folds, endpoint, endpoint_col, model_path, only_parents=False, split_num=-1):
    """
    Create and save the training and test sets for CV. 
    The training set contains both parent compounds and metabolites which are not among the test compounds.
    The test set is separated in parent and metabolite compounds for separate evaluation.
    Input:
        parents_df: dataframe - input dataframe of parent compounds with descriptors
        metabolites_df: dataframe - input dataframe of metabolites with descriptors (and labels for training)
        output_path: str - path to the folder to store the splits
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        endpoint_col: str - name of the column containing the class label
        model_path: str - path to save the feature scaler model and the remaining columns after variance filter 
        only_parents: bool - True to write out only the files containing parent compounds
        split_num: int - split number to create the splits for. To create all, set value to -1
    
    Output files:
        testset parents: f"{output_path}/{endpoint}_testset_metabLabels_{count}.csv"
        training set (parents and metabolites): "{output_path}/{endpoint}_trainingset_metabLabels_{count}.csv"
    """
    assert os.path.isdir(output_path), 'Output path to store does not exist.'

    parents_df.reset_index(drop=True, inplace=True) # make parent index from 0 to X
    info_cols = [c for c in metabolites_df.columns if "info_" in c] # additional columns with information about the metabolites (e.g. phaseII, biotransformation, etc.) to remove
    metabolites_df = metabolites_df[metabolites_df['parent_SMILES'].notna()] # remove parent compounds (in case they are found here) -> parent compounds have a NaN value in the parent_SMILES column
    metabolites_df.drop(info_cols, axis=1, inplace=True)

    # Save class and smiles information
    class_y_parents = parents_df[endpoint_col].values
    data_X_df_smiles = pd.concat([parents_df, metabolites_df], axis=0)  
    data_X_df_smiles.reset_index(drop=True, inplace=True) # make df index from 0 to X
    
    data_X_df = data_X_df_smiles.drop(["SMILES (Canonical)", "parent_SMILES", endpoint_col], axis=1)
    
    ### Split data for crossvalidation
    sss = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=2020)
    count = 1
    
    ### Apply variance filter and normalizer for parent and metab data sets
    # Parent data set
    parents_X_df = parents_df.drop(["SMILES (Canonical)", endpoint_col], axis=1)
    parents_X_df, selected_cols_parent, scaler_parent = prepare_descriptors(X_df=parents_X_df, parent_or_metab="parent", endpoint=endpoint, model_path=model_path)
    # Metab data set
    data_X_df, selected_cols_metab, scaler_metab = prepare_descriptors(X_df=data_X_df, parent_or_metab="metab", endpoint=endpoint, model_path=model_path)
    
    
    ### Fit model within crossvalidation and make prediction for respective test set
    for train_index, test_index in sss.split(parents_df, class_y_parents): # consider only the index of the parent compounds
        print(f"CV number {count}")
        
        if split_num == -1 or split_num == count: # only create file for the needed splits
            ### Prepare the data splits
            X_train_df, X_test_df = data_X_df_smiles.loc[train_index], data_X_df_smiles.loc[test_index] # first take the training parents and then collect their metabolites
            # save parents
            endp = X_train_df[endpoint_col].values
            smi = X_train_df["SMILES (Canonical)"].values
            X_train_formatted = X_train_df.drop([endpoint_col, "SMILES (Canonical)", "parent_SMILES"], axis=1)
            X_train_formatted = X_train_formatted[selected_cols_parent]
            X_train_formatted_norm = scaler_parent.transform(X_train_formatted.values)
            X_train_formatted = pd.DataFrame(X_train_formatted_norm, columns=X_train_formatted.columns)
            X_train_formatted = pd.concat([pd.DataFrame(smi, columns=["SMILES (Canonical)"]), pd.DataFrame(endp, columns=[endpoint]), X_train_formatted], axis=1)
            X_train_formatted.to_csv(f"{output_path}/{endpoint}_trainingset_parent_{count}.csv", index=False)

            if only_parents == False:
                ### Get the metabolites of the parents in the test set (to remove them from training set)
                test_compounds = pd.DataFrame()
                for parent_smiles in X_test_df["SMILES (Canonical)"].values:
                    parent_smiles_re = re.escape(parent_smiles)
                    parent_re = f"(^| ){parent_smiles_re}(\,|$)" # ensure that is the parent compound (and not a substructure)
                    current_metab = data_X_df_smiles[data_X_df_smiles["parent_SMILES"].str.contains(parent_re)==True]
                    test_compounds = pd.concat([test_compounds, current_metab], axis=0)
                test_compounds = pd.concat([test_compounds, X_test_df], axis=0)

                ### Get the metabolites of the parents in the training set
                train_metab = pd.DataFrame()
                for parent_smiles in X_train_df["SMILES (Canonical)"].values:
                    parent_smiles_re = re.escape(parent_smiles)
                    parent_re = f"(^| ){parent_smiles_re}(\,|$)" # ensure that is the parent compound (and not a substructure)
                    current_metab = data_X_df_smiles[data_X_df_smiles["parent_SMILES"].str.contains(parent_re)==True]
                    for metab_idx in current_metab.index:
                        if current_metab["SMILES (Canonical)"].loc[metab_idx] in test_compounds["SMILES (Canonical)"].values: #remove metabolites in test set (as metabolites or parents)
                            pass
                        else:
                            one_train_metab = pd.DataFrame([current_metab.loc[metab_idx]])
                            train_metab = pd.concat([train_metab, one_train_metab], axis=0)
                            
                X_train_df = pd.concat([X_train_df, train_metab], axis=0)
                X_train_df.drop_duplicates(subset=["SMILES (Canonical)"], keep="first", inplace=True) # remove duplicate metabolites -> if duplicate from a parent, parent is kept

                y_train = X_train_df[endpoint_col].values
                smiles_train = X_train_df["SMILES (Canonical)"].values           
                X_train_df.drop([endpoint_col, "SMILES (Canonical)", "parent_SMILES"], axis=1, inplace=True)
                X_train_df = X_train_df[selected_cols_metab]
                X_train_norm = scaler_metab.transform(X_train_df.values)
                X_train_df = pd.DataFrame(X_train_norm, columns=X_train_df.columns)

                ### Write out train parents and metabolites file
                X_train_complete_df = pd.concat([pd.DataFrame(smiles_train, columns=["SMILES (Canonical)"]), pd.DataFrame(y_train, columns=[endpoint]), X_train_df], axis=1)
                X_train_complete_df.to_csv(f"{output_path}/{endpoint}_trainingset_metab_{count}.csv", index=False)

            ### X_test with parent descriptor preparation
            y_test = X_test_df[endpoint_col].values
            smiles_test = X_test_df["SMILES (Canonical)"].values
            X_test_df.drop([endpoint_col, "SMILES (Canonical)", "parent_SMILES"], axis=1, inplace=True)
            X_test_df_parents = X_test_df[selected_cols_parent]
            X_test_norm = scaler_parent.transform(X_test_df_parents.values)
            X_test_df_parents = pd.DataFrame(X_test_norm, columns=X_test_df_parents.columns)
            ## Write out test file (only parents)
            X_test_complete_df = pd.concat([pd.DataFrame(smiles_test, columns=["SMILES (Canonical)"]), 
                                            pd.DataFrame(y_test, columns=[endpoint]), X_test_df_parents], axis=1)
            
            X_test_complete_df.to_csv(f"{output_path}/{endpoint}_testset_parent_{count}.csv", index=False)
            
            ### X_test with metab descriptor preparation
            X_test_df_metab = X_test_df[selected_cols_metab]
            X_test_norm = scaler_metab.transform(X_test_df_metab.values)
            X_test_df_metab = pd.DataFrame(X_test_norm, columns=X_test_df_metab.columns)
            ## Write out test file (only parents)
            X_test_complete_df = pd.concat([pd.DataFrame(smiles_test, columns=["SMILES (Canonical)"]), 
                                            pd.DataFrame(y_test, columns=[endpoint]), X_test_df_metab], axis=1)
            
            X_test_complete_df.to_csv(f"{output_path}/{endpoint}_testset_metab_{count}.csv", index=False)

        count = count + 1
    
    return