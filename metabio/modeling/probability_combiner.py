from operator import mod
import pandas as pd
import numpy as np
import pickle
from metabio.modeling.evaluate import calc_scores, calc_mean_results

class ProbabilityCombiner():
    def __init__(self, endpoint, metabolites_df, desc_type='chem'):
        '''
        Input:
            endpoint: str - label of the toxic endpoint
            metabolites_df: df - contains the metabolites predicted with Meteor and the corresponding parent smiles (in a column named "parent_SMILES")
            desc_type: str - chem or cddd
        '''
        self.endpoint = endpoint
        self.metabolites_df = metabolites_df
        self.desc_type = desc_type


    def predict_probabilities(self, X_test_df, model_path, cv_run, feat_sel=False, prepare_data=False, method='RF', metab_labels=False):
        '''
        Make predictions on the test set (parents and metabolites) with the baseline or the metab (including labeled metabolites in training data) models
        Input:
            X_test_df: df - dataframe containing the descriptors from the test set
            model_path: str - path to the folder where the model is located
            cv_run: int - number of cv run
            feat_sel: bool - whether to use a model trained on a selected set of features
            prepare_data: bool - whether to apply the variance filter and normalize the data before making the predictions
            method: str - machine learning method used to train the models (for loading the model)
            metab_labels: bool - whether to use the metab model (including labeled metabolites in training data) or the baseline model (=False)

        Output:
            df containing the predicted probabilites for each compound in X_test_df
        '''
        if metab_labels == True:
            baseline_or_metab = 'metab'
        else:
            baseline_or_metab = 'parent'
            
        ### Prepare data 
        if prepare_data == True:
            # Remove columns with low variance
            vf_cols = pd.read_csv(f'{model_path}/model_{self.endpoint}_{self.desc_type}_{baseline_or_metab}_variance_filter_columns.csv').values
            X_test_df = X_test_df[np.squeeze(vf_cols)]
            # Normalize all descriptors
            cols = X_test_df.columns
            scalerfile = f'{model_path}/normalizer/model_{self.endpoint}_{self.desc_type}_{baseline_or_metab}.pkl'
            scaler = pickle.load(open(scalerfile, 'rb'))
            X_test_norm = scaler.transform(X_test_df.values)
            X_test_df = pd.DataFrame(X_test_norm, columns=cols)

        
        ### Load model from CV run
        filename = f'{model_path}/model_{self.endpoint}_chem_featSel={feat_sel}_{method}_{baseline_or_metab}_{cv_run}.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        if feat_sel == True:
            selectedColumns = pd.read_csv(f'{model_path}/model_{self.endpoint}_{self.desc_type}_{baseline_or_metab}_{cv_run}_columns.csv').values.squeeze()
            X_test_df = X_test_df[[c for c in X_test_df.columns if c in selectedColumns]].values

        predictions = loaded_model.predict_proba(X_test_df.values)
        temp = pd.DataFrame(predictions, columns=['0', '1'])
        pred_prob = temp['1'].values #column with the probability of class 1 
        
        pred_prob_df = pd.DataFrame(pred_prob, columns=['probability'])
        
        return pred_prob_df
    
    def get_metabolites_from_parent(self, parent_smiles, score_threshold=0, detox_phaseII=False, logP=-99):
        '''
        Get the metabolites predicted for a parent compound
        Filters regarding the Meteor score, the logP or the involvement in phaseII reactions can be applied
        '''
        metabolites_subset = self.metabolites_df.copy()
        # Filter out compounds that are further metabolized in phase II reactions
        if detox_phaseII == True:
            metabolites_subset = metabolites_subset[metabolites_subset['info_Detox_phaseII'] == False]
            metabolites_subset = metabolites_subset[metabolites_subset['info_Phase'] != 'Phase II']
        
        if logP != -99:
            metabolites_subset = metabolites_subset[metabolites_subset['info_LogP'] > logP]
            
        if score_threshold != 0: # all predicted metabolites
            metabolites_subset = metabolites_subset[metabolites_subset['info_Absolute Score'] > score_threshold]
        
        metabolites_parent_df = metabolites_subset[metabolites_subset['parent_SMILES'] == parent_smiles]
        
        info_cols = [c for c in metabolites_parent_df.columns if 'info_' in c]
        metabolites_parent_df = metabolites_parent_df.drop(info_cols, axis=1)

        return metabolites_parent_df

    def format_parents_df(self, parents_df):
        drop_cols = [c for c in parents_df.columns if 'transf_' in c]
        parents_df.drop(drop_cols, axis=1, inplace=True)
        
        y_test = parents_df[self.endpoint]
        smiles_col = [c for c in parents_df.columns if 'smiles' in c.lower()][0]
        smiles_parents = parents_df[smiles_col]
        parents_df.drop([self.endpoint, smiles_col], axis=1, inplace=True)
        
        return parents_df, y_test, smiles_parents

    def evaluate_combined_probabilities(self, test_parents_path, model_path, results_path, approach='hybrid',
                                        combinations=['baseline', 'only_parents', 'mean', 'median', 'max_all', 'max_metab'], 
                                        cv_runs=5, feat_sel=False, score_threshold=0, detox_phaseII=False, logP=-99, method='RF'):
    
        '''
        Evaluate the predictions on the test set based on a combination of the predicted probabilites for each parent compound and its metabolites
        The probabilites may be combined by taking the mean or median predicted probability over all compounds (i.e. the parent compound and all predicted metabolites), 
        the maximum predicted probability among the parent compound and its predicted metabolites ('max_all'), 
        or the mean between the predicted probability of the parent compound and the maximum probability among all predicted metabolites ('max_metab')
        Input:
            test_parents_path: str - path where the data splits are saved (as output from split_data.create_CV_splits)
            model_path: str - path to the folder where the model is located
            results_path: str - path to the folder to save the results
            approach: str - 'hybrid' (metab_model_for_metab=True and metab_model_for_parent=False) or 'baseline' (metab_model_for_metab=False and metab_model_for_parent=False)
            combinations: array - may contain all or some of the following combinations: ['baseline', 'only_parents', 'mean', 'median', 'max_all', 'max_metab']
            cv_runs: int - number of cross-validation runs to evaluate
            feat_sel: bool - whether to use models trained on a selected set of features
            score_threshold: int - score threshold of the considered metabolites
            detox_phaseII: bool - if True -> filter out compounds further metabolized by phase II reactions or products from those reactions
            logP: float - minimum logP used to filter metabolites
            method: str - machine learning method used to train the models (for loading the model)

        Output:
            df containing the predicted probabilites for each compound in X_test_df
        '''

        if approach == 'hybrid':
            metab_model_for_metab=True
            metab_model_for_parent=False
        elif approach == 'baseline':
            metab_model_for_metab=False
            metab_model_for_parent=False
        elif approach == 'metab':
            metab_model_for_metab=True
            metab_model_for_parent=True

        # Initialization
        all_results = {}
        all_cv_results_dict = {}
        all_mean_results = pd.DataFrame()
        
        if metab_model_for_parent == True: # if the metab model is used to make predictions on parent cmpds, also the predictions on parent cmpds are evaluated (without combining)
            if not 'only_parents' in combinations:
                combinations.append('only_parents')
        else:
            if 'only_parents' in combinations: # if the baseline model is used to make predictions on parent cmpds: only_parents == baseline
                combinations.remove('only_parents')
        for combination in combinations:
            all_results[combination] = pd.DataFrame()
            all_cv_results_dict[combination] = pd.DataFrame()

        ### Evaluate on all cv_runs
        for run in range(1, cv_runs+1):
            print(f'Predicting probabilites of CV run {run}')
            # save dataframes with predicted probabilitities (baseline and combinations)
            probabilities = {'baseline': [], 'only_parents': [], 'mean': [], 'median': [], 'max_all': [], 'max_metab': []}

            # Use the test set as prepared (variance filter and normalizer) for the model trained on the parents only or also on the labeled metabolites            
            ### Load parent compound data and calculate probabilities (with metab or baseline model)
            if metab_model_for_parent == False: 
                parents_df = pd.read_csv(f'{test_parents_path}/{self.endpoint}_testset_parent_{run}.csv')
                parents_df, y_test, smiles_parents = self.format_parents_df(parents_df)
                # Predict probabilities of parent compounds
                baseline_prob = self.predict_probabilities(parents_df, model_path, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=False)
                parents_prob = baseline_prob.copy()
                if 'only_parents' in combinations:
                    combinations.remove('only_parents')
            else:
                parents_df = pd.read_csv(f'{test_parents_path}/{self.endpoint}_testset_metab_{run}.csv')
                # Prepare also parent file for baseline -> used to calculate significance of results difference
                parents_df_baseline = pd.read_csv(f'{test_parents_path}/{self.endpoint}_testset_parent_{run}.csv')
                drop_cols = [c for c in parents_df_baseline.columns if 'transf_' in c or 'smiles' in c.lower() or self.endpoint in c]
                parents_df_baseline.drop(drop_cols, axis=1, inplace=True)
                
                parents_df, y_test, smiles_parents = self.format_parents_df(parents_df)
                # Predict probabilities of parent compounds
                # Baseline model (with parent model)
                baseline_prob = self.predict_probabilities(parents_df_baseline, model_path, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=False)
                parents_prob = self.predict_probabilities(parents_df, model_path, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=True)
                probabilities['only_parents'] = parents_prob.copy()
                if not 'only_parents' in combinations:
                    combinations.append('only_parents')
            
            probabilities['baseline'] = baseline_prob
            parents_prob['parent_SMILES'] = smiles_parents.values

            for parent in smiles_parents:
                # Get probability of this parent
                parent_prob = parents_prob[parents_prob['parent_SMILES'] == parent]
                #parent_prob = parent_prob['probability'].values[0]
                
                ### Predict probabilities of metabolites
                if combinations != ['baseline'] or combinations != ['only_parents']: # not needed for baseline model
                    metabolites_parent_df = self.get_metabolites_from_parent(parent, score_threshold, detox_phaseII=detox_phaseII, logP=logP)
                    #print(len(metabolites_parent_df))
                    if metabolites_parent_df.empty:
                        metabolites_prob = pd.DataFrame(columns=['probability'])
                    else:
                        drop_cols = [c for c in metabolites_parent_df.columns if 'smiles' in c.lower()]
                        metabolites_desc = metabolites_parent_df.drop(drop_cols, axis=1)
                        metabolites_prob = self.predict_probabilities(metabolites_desc, model_path, cv_run=run, feat_sel=feat_sel, prepare_data=True, method=method, metab_labels=metab_model_for_metab)
                
                ### Averaging combination for the final predicted probability
                assert 'baseline' in combinations or 'mean' in combinations or 'median' in combinations or 'max_all' in combinations or 'max_metab' in combinations, 'Invalid mode. Accepted combinations: baseline, mean, median, max_all, max_metab'

                if 'mean' in combinations:
                    # Mean probability of metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    mean_prob = all_prob['probability'].mean(axis=0)
                    probabilities['mean'].append(mean_prob)
                    
                if 'median' in combinations:
                    # Median probability of metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    median_prob = all_prob['probability'].median(axis=0)
                    probabilities['median'].append(median_prob)
                    
                if 'max_all' in combinations:
                    # Max probability from metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    max_all_prob = all_prob['probability'].max(axis=0)
                    probabilities['max_all'].append(max_all_prob)
                    
                if 'max_metab' in combinations:
                    # Mean between the max probability for a metabolite and the parent probability
                    max_metabolites_prob = metabolites_prob['probability'].max(axis=0)
                    all_prob = pd.concat([parent_prob, pd.DataFrame([max_metabolites_prob], columns=['probability'])])
                    max_metab_prob = all_prob['probability'].mean(axis=0)
                    probabilities['max_metab'].append(max_metab_prob)
                    
            
            # Calculate class based on the combined probability
            for combination in combinations:
                parents_prob[f'{combination}_probability'] = probabilities[combination]
                parents_prob[f'{combination}_class'] = np.where(parents_prob[f'{combination}_probability']>=0.5, 1, 0)

            
            ### Calculate results of CV run
            for combination in combinations:
                result = calc_scores(y_test.values.astype('int'), parents_prob[f'{combination}_class'].values.astype('int'), y_pred_prob=parents_prob[f'{combination}_probability'].values, silent=True)
                all_results[combination] = pd.concat([all_results[combination], result])

                # Save all information for output file
                cv_results = pd.DataFrame({'smiles': smiles_parents, f'{self.endpoint}': y_test, 'probability': parents_prob[f'{combination}_probability'].values, 
                                            'prediction': parents_prob[f'{combination}_class'].values, 'cv': run})
                all_cv_results_dict[combination] = pd.concat([all_cv_results_dict[combination], cv_results], axis=0)

        for combination in combinations:
            ### Print out cross-validation file with predictions
            all_cv_results_dict[combination].to_csv(f'{results_path}/results_{self.endpoint}_{approach}-approach_{combination}_featSel={feat_sel}_scoreThres={score_threshold}_detoxPhaseII={detox_phaseII}_logP={logP}.csv', index=False)
            
            ### Calculate mean results of CV runs on test sets with metabolites
            mean_results = calc_mean_results(all_results[combination])

            ### Prepare output file with mean results
            mean_results.insert(loc=0, column='combination', value=combination)
            mean_results.insert(loc=0, column='detox_phaseII', value=detox_phaseII)
            mean_results.insert(loc=0, column='logP', value=logP)
            mean_results.insert(loc=0, column='score_threshold', value=score_threshold)
            mean_results.insert(loc=0, column='endpoint', value=self.endpoint)
            all_mean_results = pd.concat([all_mean_results, mean_results], axis=0)
                
        return all_mean_results