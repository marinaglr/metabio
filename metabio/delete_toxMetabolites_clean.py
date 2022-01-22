import pandas as pd
import numpy as np
import re
import os
import sys
sys.path.append("/home/garcim64/Documents/CPModels/scripts")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, roc_auc_score
from scipy.stats import mannwhitneyu

import cloudpickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator



class toxMetabolites:
    def __init__(self, completeDB=None, MAIN_PATH="/home/", missingVal=-1,
                chembio_complete=None):
        self.completeDB = completeDB # complete database with in vitro and in vivo endpoints
        self.mainPath = MAIN_PATH
        self.missingVal = missingVal # value assigned to missing values
        self.test_portion = 0.2 # ratio to divide data into test and training set
        self.crossvalidation_output_file = f'{self.mainPath}/results/crossvalidation/crossvalidation'
        self.model_pickle_output_path= f'{self.mainPath}/models/'
        self.model_pickle_output_file = f'{self.model_pickle_output_path}/model'
        self.prediction_output_file = f'{self.mainPath}/results/predictions/prediction'
        
        # Initialize results folders
        if os.path.isdir(f"{self.mainPath}/results/crossvalidation") == False:
            if os.path.isdir(f"{self.mainPath}/results") == False:
                os.mkdir(f"{self.mainPath}/results")
            os.mkdir(f"{self.mainPath}/results/crossvalidation")
        if os.path.isdir(f"{self.mainPath}/results/plots") == False:
                os.mkdir(f"{self.mainPath}/results/plots")
        if os.path.isdir(f"{self.mainPath}/results/predictions") == False:
                os.mkdir(f"{self.mainPath}/results/predictions")
        if os.path.isdir(f"{self.model_pickle_output_path}/normalizer") == False:
            if os.path.isdir(f"{self.model_pickle_output_path}") == False:
                os.mkdir(f"{self.model_pickle_output_path}")
            os.mkdir(f"{self.model_pickle_output_path}/normalizer")
        if os.path.isdir(f"{self.mainPath}/data/splits") == False:
            if os.path.isdir(f"{self.mainPath}/data") == False:
                os.mkdir(f"{self.mainPath}/data")
            os.mkdir(f"{self.mainPath}/data/splits")
            
    def calculate_fingerprint(self, smiles, fp_size, radius, metab_num=1):
        """
        Calculates count Morgan fingerprints as pandas columns
        
        Parameters:
        ----------
        smiles: str - smiles from one compound
        fp_size: int - size of the Morgan fingerprint
        radius: int - radius of the Morgan fingerprint
        metab_num: int - metabolite number from the list of best scored metabolites. Number used to name the fingerprint columns of each metabolite.
        
        Output:
        -------
        return: dataframe (with one row) containing each fingerprint byte in one column
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(False)
        Chem.GetSSSR(mol)

        ### Count fingerprint
        fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=fp_size)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        fp = array

        fp_column_names = [f"metab_{metab_num}_bit_{i}" for i in range(len(fp))]
        fp = pd.DataFrame([fp], columns=fp_column_names)
    
        return fp

    def prepare_datasets(self, endpoint, metabolites_df=pd.DataFrame(), score_threshold=0,
                         detox_phaseII=False, logP=-99, morgan_metab=False, physchem_metab=False):
        """
        Prepare following data sets:
            chem: with molecular physchem descriptors and fingerprint
            metab: physchem + fingerprint + descriptors from metabolites
        
        Parameters:
        ----------
        endpoint: str - name of the endpoint -> class column in data set must be "Toxicity_{endpoint}"
        metabolites_df: dataframe - contains the metabolites predicted with Meteor and the corresponding parent smiles (in a column named "parent_SMILES")
        score_threshold: int - score threshold of the considered metabolites
        detox_phaseII: bool - if True -> filter out compounds further metabolized by phase II reactions or products from those reactions
        logP: float - minimum logP used to filter metabolites
        morgan_metab: bool - if True -> calculate the Morgan count fingerprint from metabolites
        physchem_metab: bool - if True -> calculate the RDKit physchem properties from metabolites
        
        Output:
        -------
        return: dictionary containing: metab (df with metabolite descriptors), chem (only parent descriptors), class_endp, smiles_endp
        """
        
        cols = [c for c in self.completeDB.columns if 'smiles' in c.lower() or 'CAS' in c]
        dfraw = self.completeDB.drop(cols, axis=1)

        endpoint_col = f"Toxicity_{endpoint}"
            
        ## endpoint smiles
        smilesCol = [c for c in self.completeDB.columns if c.lower() == 'smiles' or c.lower() == "smiles (canonical)"]
        smiles_endp = self.completeDB[self.completeDB[endpoint_col] != self.missingVal][smilesCol]

        df_endp_smiles = self.completeDB[self.completeDB[endpoint_col] != self.missingVal]
        df_endp = dfraw[dfraw[endpoint_col] != self.missingVal]

        # Calculate descriptors for the 5 top ranked metabolites
        if morgan_metab == True or physchem_metab == True:
            desc_all_parents = pd.DataFrame()
            if physchem_metab == True:
                ### get RDKit descriptor list
                mol_descriptors = [x[0] for x in Descriptors._descList]
                calculator = MolecularDescriptorCalculator(mol_descriptors)

            for parent_smiles in smiles_endp[smilesCol].values:
                # get metabolites for parent (with given filters)
                metabolites = self.get_metabolites_from_parent(parent_smiles[0], metabolites_df, score_threshold=score_threshold,
                                                               detox_phaseII=detox_phaseII, logP=logP)
                all_desc_df = pd.DataFrame()
                all_fp_df = pd.DataFrame()
                metab_num = 1

                # Calculate descriptors for the top 5 metabolites of each parent
                for metabolite_smi in metabolites["SMILES (Canonical)"].values[:5]:
                    
                    if morgan_metab == True:
                        ### Calculate Morgan (count) fingerprint of metabolites
                        fp = self.calculate_fingerprint(metabolite_smi, 1024, radius=2, metab_num=metab_num)
                        all_fp_df = pd.concat([all_fp_df, fp], axis=1)
                    
                    if physchem_metab == True:
                        ### Calculate RDKit descriptors of metabolites
                        desc = calculator.CalcDescriptors(mol=Chem.MolFromSmiles(metabolite_smi))
                        desc = pd.DataFrame([desc], columns=[f"metab_{metab_num}_{j}" for j in mol_descriptors])
                        all_desc_df = pd.concat([all_desc_df, desc], axis=1)
                    
                    metab_num = metab_num + 1

                # Concatenate the descriptors of the metabolites from each parent compound
                if morgan_metab == True and physchem_metab == True:
                    all_input_desc = pd.concat([all_fp_df, all_desc_df], axis=1)
                elif morgan_metab == False and physchem_metab == True:
                    all_input_desc = all_desc_df.copy()
                elif morgan_metab == True and physchem_metab == False:
                    all_input_desc = all_fp_df.copy()
                all_input_desc["SMILES (Canonical)"] = parent_smiles
                desc_all_parents = pd.concat([desc_all_parents, all_input_desc], axis=0)
            
            if morgan_metab == True:
                fing_cols = [c for c in desc_all_parents.columns if "bit_" in c]
                desc_all_parents[fing_cols].fillna(value=0, inplace=True) # fill fps for parents with less than 5 metabolites with 0
            if physchem_metab == True:
                desc_all_parents.replace([np.inf, -np.inf], np.nan, inplace=True) # avoid infinite numbers
                desc_all_parents.fillna(desc_all_parents.mean(), inplace=True) # fill the rest of descriptors for parents with less than 5 metabolites with the average of the column

            desc_all_parents.reset_index(drop=True, inplace=True)
            
            # concatenate fps to the other descriptors
            df_desc_metab = df_endp_smiles.merge(desc_all_parents, how="left", on="SMILES (Canonical)")
            drop_invivocols = [c for c in df_desc_metab.columns if 'vivo' in c.lower() or 'toxicity' in c.lower() or 'smiles' in c.lower()]
            df_desc_metab = df_desc_metab.drop(drop_invivocols, axis=1) # with fps of metabolites
        

        else:
            drop_invivocols = [c for c in df_endp.columns if 'vivo' in c.lower() or 'toxicity' in c.lower()]
            df_desc_metab = df_endp.drop(drop_invivocols, axis=1) # with fps of metabolites

        
        ## Parent compound descriptors + fingerprint
        drop_cols = [c for c in df_endp.columns if 'vivo' in c.lower() or 'vitro' in c.lower() or 'toxicity' in c.lower() or 'metab' in c.lower() or 'p0' in c or 'p1' in c or "Split" in c or "info_" in c.lower()]
        df_chem_endp = df_endp.drop(drop_cols, axis=1)

        ## endpoint class label
        class_endp = df_endp[endpoint_col]
        encoder = LabelEncoder()
        encoder.fit(class_endp)
        class_endp = encoder.transform(class_endp)
        
        endp_data = {"metab": df_desc_metab, "chem": df_chem_endp, "class": class_endp, "smiles": smiles_endp}
            
        return endp_data
    
    #TODO -> should I keep this one or the one with the splits?
    def train_models(self, data_X_df, class_y, smiles, cv_folds, endpoint, desc_type, num_trees, method="RF", feat_imp=True, 
                     feat_sel=False, class_weight="balanced", oversampling=False):
        
        """
        Train models within a crossvalidation and save internal predictions
        
        Parameters:
        ----------
        data_X_df: dataframe - input dataframe with descriptors
        class_y: list - with the class values
        smiles: list - with the smiles matching the class_y array
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        desc_type: str - chem or metab
        num_trees: int - number of trees of RF
        method: str - RF, LR, SVM or XGB
        feat_imp: bool - calculate feature importances for RF
        feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
        
        Output:
        -------
        return: models
        print: file "{self.mainPath}/results/crossvalidation/crossvalidation_{endpoint}_{desc_type}_featSel={feat_sel}.csv" with cross-validation p-values and prediction
        """
    
        data_X = data_X_df.values
        all_results = pd.DataFrame()
        
        ### Split data for crossvalidation
        sss = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=2020)
        models_cv = [] # Save all models to use for further prediction
        count = 1

        ### Prepare output file
        cv_outfile = open(f'{self.crossvalidation_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}.csv', 'w')
        header = f'name,{endpoint},probability,prediction,smiles,cv\n'
        cv_outfile.write(header)

        if feat_sel == True:
            lasso_coefficients_cv = pd.DataFrame({"Feature": data_X_df.columns}) # store coefficients from all CV runs

        ### Variance filter
        n_cols_1 = len(data_X_df.columns)
        selector = VarianceThreshold(0.001)
        dfraw_selFeatures = pd.DataFrame(selector.fit_transform(data_X_df), columns=[data_X_df.columns[selector.get_support(indices=True)]])
        data_X_df = data_X_df[data_X_df.columns[selector.get_support(indices=True)]]
        data_X = data_X_df.values
        n_cols_2 = len(data_X_df.columns)
        print(f"Variance filter removed {n_cols_1-n_cols_2} columns")
        pd.DataFrame(data_X_df.columns, 
                     columns=["Feature"]).to_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_parent_variance_filter_columns.csv', index=False)

        if feat_imp == True:
            feat_importances_cv = pd.DataFrame({"Feature": data_X_df.columns})

        ### Normalize all descriptors
        scaler = StandardScaler()
        data_X = scaler.fit_transform(data_X_df)
        data_X_df = pd.DataFrame(data_X, columns=data_X_df.columns)
        data_X_df.reset_index(drop=True, inplace=True)
        print(data_X_df.shape)
        # Save the scaler
        scalerfile = f'{self.model_pickle_output_path}/normalizer/model_{endpoint}_{desc_type}_parent.pkl'
        cloudpickle.dump(scaler, open(scalerfile, 'wb'))
        #data_X_df.to_csv(f"{self.mainPath}/data/internal/{endpoint}_training_set_normalized.csv")

        ### Fit model within crossvalidation and make prediction for respective test set
        for train_index, test_index in sss.split(data_X, class_y):
            print(f"CV number {count}")

            ### Prepare the data splits
            X_train, X_test = data_X[train_index], data_X[test_index]
            y_train, y_test = class_y[train_index], class_y[test_index]
            smiles_train, smiles_test = smiles[train_index], smiles[test_index]

            X_train_df = pd.DataFrame(X_train, columns=data_X_df.columns)
            X_test_df = pd.DataFrame(X_test, columns=data_X_df.columns)
            
            
            ### Write out test files
            X_test_complete_df = pd.concat([pd.DataFrame(smiles_test, columns=["SMILES (Canonical)"]), pd.DataFrame(y_test, columns=[endpoint]), X_test_df], axis=1)
            #display(X_test_smiles_df)
            #X_test_complete_df.to_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_{count}.csv", index=False)


            ### Feature selection with lasso
            if feat_sel == True:                    
                model_sel = LassoCV(cv=sss, random_state=2020, n_jobs=4, max_iter=1000000, normalize=False)
                model_sel.fit(X_train_df.values, y_train)

                selected_feat = []
                for i in range(len(model_sel.coef_)):
                    if model_sel.coef_[i] != 0:
                        selected_feat.append(X_train_df.columns[i])

                # Store selected columns to filter them when applying the model on new compounds
                pd.DataFrame(selected_feat, columns=["Feature"]).to_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_parent_{count}_columns.csv', index=False)

                #selected_feat = X_train_df.columns[(model_sel.coef_ > 0)]
                print('total features: {}'.format((X_train_df.shape[1])))
                print('selected features: {}'.format(len(selected_feat)))
                print('features with coefficients shrank to zero: {}'.format(np.sum(model_sel.coef_ == 0)))

                # Save coefficients
                coeff = pd.DataFrame({"Coefficients": abs(model_sel.coef_)})
                lasso_coefficients = pd.DataFrame({"Feature": X_train_df.columns, "Coefficient": coeff["Coefficients"].values})
                selected_lasso_coefficients = lasso_coefficients[lasso_coefficients["Coefficient"] != 0]
                lasso_coefficients_cv = lasso_coefficients_cv.merge(lasso_coefficients, on="Feature", validate="one_to_one")


                X_train = X_train_df[selected_feat].values
                X_test = X_test_df[selected_feat].values
                X_train_df = X_train_df[selected_feat]
                X_test_df = X_test_df[selected_feat]
            
            ### Oversampling with SMOTEC
            if oversampling == True:
                # Get location of fingerprint columns (not just name of columns)
                fingColumns=[]
                col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower() or "bit_" in c.lower() or "transf" in c.lower() or "Num" in c or "Count" in c or "_fr_" in c]]
                #col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower()]]
                for i in col:
                    fingColumns.append(X_train_df.columns.get_loc(i))

                sm = SMOTENC(categorical_features=fingColumns, sampling_strategy=0.8)
                #sm = SMOTE(sampling_strategy=0.8)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_train_df = pd.DataFrame(X_train)

            ### Create and train model on training set
            if method == "RF":
                model = RandomForestClassifier(n_estimators=num_trees, class_weight=class_weight, min_samples_leaf=3, random_state=2020, n_jobs=4)

            elif method == "LR":
                '''
                if feat_sel == False:
                    model = LogisticRegression(max_iter=800, C=0.8, random_state=2020, n_jobs=4, class_weight=class_weight)
                else:
                '''
                model = LogisticRegression(max_iter=1000, C=0.8, random_state=2020, n_jobs=4, class_weight=class_weight)
                #model = LogisticRegressionCV(Cs=10, penalty="l1", random_state=2020, n_jobs=4, cv=sss, class_weight=class_weight, solver="liblinear", max_iter=1000)

            elif method == "XGB":
                model = GradientBoostingClassifier(n_estimators=num_trees, random_state=2020, min_samples_leaf=2)

            elif method == "SVM":
                model = SVC(max_iter=-1, random_state=2020, probability=True, class_weight=class_weight)

            model.fit(X_train, y_train)

            ### Make predictions for test set
            pred_class = model.predict(X_test)
            temp = pd.DataFrame(model.predict_proba(X_test), columns=["0", "1"])
            pred_prob = temp["1"].values #column with the probability of class 1            
            
            # Print performance on test set
            result = self.calc_scores(y_test, pred_class, y_pred_prob=pred_prob, silent=True, modelname=f"{endpoint}_{desc_type}")
            all_results = pd.concat([all_results, result])
            
            # Extract feature importance values
            if feat_imp == True and method == "RF":
                if feat_sel == True:
                    X_columns = X_train_df[selected_feat].columns
                else:
                    X_columns = data_X_df.columns
                feat_importances = pd.DataFrame({'Importance':model.feature_importances_})    
                feat_importances['Feature'] = X_columns
                feat_importances_cv = feat_importances_cv.merge(feat_importances, on="Feature", validate="one_to_one")

            # Write prediction and further compound information for each split into output file
            long_line = ''
            for n, i in enumerate(X_test): 
                newline = f'{n}, {y_test[n]}, {pred_prob[n]}, {pred_class[n]}, {smiles_test[n][0]}, {count}\n'
                long_line += newline
            cv_outfile.write(long_line)

            # Collect models for later predictions of external test data
            models_cv.append(model)

            ### Save the model to disk
            filename = f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_parent_{count}.sav'
            cloudpickle.dump(model, open(filename, 'wb'))

            count = count + 1
            
        
        ### Calculate mean results on test set
        mean_results = self.calc_mean_results(all_results)
        #print(mean_results)
        

        if feat_sel == True:
            print("-------------------Coefficients CV-----------------")
            #lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_{count-1}.csv')
            mean_coef = lasso_coefficients_cv.mean(axis=1).values
            lasso_coefficients_cv_mean = pd.DataFrame({"Feature": lasso_coefficients_cv.Feature, "Mean coeff.": mean_coef})
            print(lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False)[:20])
            lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False).to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}.csv')
            lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_allCV.csv')

        if feat_imp == True and method == "RF":
            print("-------------------Importances CV-----------------")
            mean_imp = feat_importances_cv.mean(axis=1).values
            feat_importances_cv_mean = pd.DataFrame({"Feature": feat_importances_cv.Feature, "Mean importance": mean_imp})
            #display(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            print(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False).to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}.csv')
            
            feat_importances_cv.to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}_allCV.csv')
            std_imp = feat_importances_cv.std(axis=1).values           

        cv_outfile.close()

        # return trained models
        return models_cv
    
    def prepare_descriptors(self, X_df, parent_or_metab, endpoint, desc_type="chem", save_normalizer=True, save_columns=True):
        """
        Filter low variance columns and normalize features
        
        Parameters:
        ----------
        X_df: dataframe - input dataframe with descriptors
        parent_or_metab: str #TODO 
        endpoint: str - name of the endpoint
        desc_type: str - #TODO
        save_normalizer: bool - save the model used to scale the features
        save_columns: bool - save a CSV file containing the name of the remaining columns after the variance filter
        
        Output:
        -------
        X_df: dataframe - X data after variance filter and feature normalization
        selected_cols: list - columns after variance filter
        scaler: model fited on X_df to normalize the features
        """
        ### Variance filter
        num_all_cols = len(X_df.columns)
        selector = VarianceThreshold(0.001)
        dfraw_selFeatures = pd.DataFrame(selector.fit_transform(X_df), columns=[X_df.columns[selector.get_support(indices=True)]])
        X_df = X_df[X_df.columns[selector.get_support(indices=True)]]
        X = X_df.values
        selected_cols = X_df.columns
        num_selected_cols = len(X_df.columns)
        print(f"Variance filter removed {num_all_cols-num_selected_cols} columns")
        if save_columns == True:
            pd.DataFrame(X_df.columns, 
                        columns=["Feature"]).to_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_{parent_or_metab}_variance_filter_columns.csv', index=False)

        
        ### Normalize all descriptors
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df)
        X_df = pd.DataFrame(X, columns=X_df.columns)
        print(X_df.shape)
        # Save the scaler
        scalerfile = f'{self.model_pickle_output_path}/normalizer/model_{endpoint}_{desc_type}_{parent_or_metab}.pkl'
        if save_normalizer == True:
            cloudpickle.dump(scaler, open(scalerfile, 'wb'))
        #X_df.to_csv(f"{self.mainPath}/data/internal/{endpoint}_training_set_normalized.csv")
        
        return X_df, selected_cols, scaler
        
    #TODO
    def create_CV_splits(self, parents_df, metabolites_df, cv_folds, endpoint, endpoint_col, only_parents=False, split_num=-1):
        """
        Create and save the training and test sets for CV. 
        The training set contains both parent compounds and metabolites which are not among the test compounds.
        The test set is separated in parent and metabolite compounds for separate evaluation.
        
        Parameters:
        ----------
        parents_df: dataframe - input dataframe of parent compounds with descriptors
        metabolites_df: dataframe - input dataframe of metabolites with descriptors (and labels for training)
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        endpoint_col: str - name of the column containing the class label
        only_parents: bool - True to write out only the files containing parent compounds
        split_num: int - split number to create the splits for. To create all, set value to -1
        
        Output:
        -------
        output files: 
            testset parents: f"{self.mainPath}/data/splits/{endpoint}_testset_metabLabels_{count}.csv"
            training set (parents and metabolites): "{self.mainPath}/data/splits/{endpoint}_trainingset_metabLabels_{count}.csv"
        """
        
        parents_df.reset_index(drop=True, inplace=True) # make parent index from 0 to X
        info_cols = [c for c in metabolites_df.columns if "info_" in c]
        metabolites_df = metabolites_df[metabolites_df['parent_SMILES'].notna()] # remove parent compounds (in case they are found here)
        metabolites_df.drop(info_cols, axis=1, inplace=True)

        class_y_parents = parents_df[endpoint_col].values
        
        data_X_df_smiles = pd.concat([parents_df, metabolites_df], axis=0)  
        data_X_df_smiles.reset_index(drop=True, inplace=True) # make df index from 0 to X
        data_X_df_smiles.to_csv(f"{self.mainPath}/test/all_data_with_metabolites.csv", index=True, index_label="idx")
        
        data_X_df = data_X_df_smiles.drop(["SMILES (Canonical)", "parent_SMILES", endpoint_col], axis=1)
        
        ### Split data for crossvalidation
        sss = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=2020)
        count = 1
        
        ### Aplly variance filter and normalizer for parent and metab data sets
        # Parent data set
        parents_X_df = parents_df.drop(["SMILES (Canonical)", endpoint_col], axis=1)
        parents_X_df, selected_cols_parent, scaler_parent = self.prepare_descriptors(X_df=parents_X_df, parent_or_metab="parent", endpoint=endpoint)
        # Metab data set
        data_X_df, selected_cols_metab, scaler_metab = self.prepare_descriptors(X_df=data_X_df, parent_or_metab="metab", endpoint=endpoint)
        
        
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
                print(type(X_train_formatted))
                print(selected_cols_parent)
                X_train_formatted = X_train_formatted[selected_cols_parent]
                X_train_formatted_norm = scaler_parent.transform(X_train_formatted.values)
                X_train_formatted = pd.DataFrame(X_train_formatted_norm, columns=X_train_formatted.columns)
                X_train_formatted = pd.concat([pd.DataFrame(smi, columns=["SMILES (Canonical)"]), pd.DataFrame(endp, columns=[endpoint]), X_train_formatted], axis=1)
                #if "parent_SMILES" in X_train_df:
                #    X_train_formatted.drop(["parent_SMILES"], axis=1, inplace=True)
                X_train_formatted.to_csv(f"{self.mainPath}/data/splits/{endpoint}_trainingset_parent_{count}.csv", index=False)

                if only_parents == False:
                    ### Get the metabolites of the parents in the test set (to remove them from training set)
                    test_compounds = pd.DataFrame()
                    for parent_smiles in X_test_df["SMILES (Canonical)"].values:
                        parent_smiles_re = re.escape(parent_smiles)
                        parent_re = f"(^| ){parent_smiles_re}(\,|$)" # ensure that is the parent compound (and not a substructure)
                        current_metab = data_X_df_smiles[data_X_df_smiles["parent_SMILES"].str.contains(parent_re)==True]
                        #current_metab = data_X_df_smiles[data_X_df_smiles[parent_smiles in data_X_df_smiles["parent_SMILES"]]]
                        test_compounds = pd.concat([test_compounds, current_metab], axis=0)
                    test_compounds = pd.concat([test_compounds, X_test_df], axis=0)
                    print(len(test_compounds))

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
                    X_train = X_train_df.values

                    ### Write out train parents and metabolites file
                    X_train_complete_df = pd.concat([pd.DataFrame(smiles_train, columns=["SMILES (Canonical)"]), pd.DataFrame(y_train, columns=[endpoint]), X_train_df], axis=1)
                    X_train_complete_df.to_csv(f"{self.mainPath}/data/splits/{endpoint}_trainingset_metab_{count}.csv", index=False)

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
                
                X_test_complete_df.to_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_parent_{count}.csv", index=False)
                
                ### X_test with metab descriptor preparation
                X_test_df_metab = X_test_df[selected_cols_metab]
                X_test_norm = scaler_metab.transform(X_test_df_metab.values)
                X_test_df_metab = pd.DataFrame(X_test_norm, columns=X_test_df_metab.columns)
                ## Write out test file (only parents)
                X_test_complete_df = pd.concat([pd.DataFrame(smiles_test, columns=["SMILES (Canonical)"]), 
                                                pd.DataFrame(y_test, columns=[endpoint]), X_test_df_metab], axis=1)
                
                X_test_complete_df.to_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_metab_{count}.csv", index=False)

            count = count + 1
        
        return
    
    
    def train_models_from_CV_files(self, cv_folds, endpoint, desc_type, num_trees, method="RF", feat_imp=True, 
                                 feat_sel=False, class_weight="balanced", oversampling=False):
        
        """
        Train models within a crossvalidation -> CV files are loaded (and not directly calculated)
        Internal predictions are saved
        
        Parameters:
        ----------
        cv_folds: int - number of folds
        endpoint: str - name of the endpoint
        desc_type: str - in vitro or in vivo
        num_trees: int - number of trees of RF
        method: str - RF, LR, SVM or XGB
        feat_imp: bool - calculate feature importances for RF
        feat_sel: bool - perform feature selection with lasso (without optimization of regularization parameter)
        
        Output:
        -------
        return: models
        print: file "{self.mainPath}/results/crossvalidation/crossvalidation_{endpoint}_{desc_type}_featSel={feat_sel}.csv" with cross-validation p-values and prediction
        """
        
        ### Prepare output file
        cv_outfile = open(f'{self.crossvalidation_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}.csv', 'w')
        header = f'name,{endpoint},probability,prediction,smiles,cv\n'
        cv_outfile.write(header)

        ### Initialization
        all_results = pd.DataFrame()
        models_cv = [] # Save all models to use for further prediction
        data_X_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_trainingset_parent_1.csv") # to get the column names
        if feat_sel == True:
            lasso_coefficients_cv = pd.DataFrame({"Feature": data_X_df.columns}) # store coefficients from all CV runs
        if feat_imp == True:
            feat_importances_cv = pd.DataFrame({"Feature": data_X_df.columns})

        
        ### Train CV models
        for count in range(1, cv_folds+1):
            print(f"Training CV {count}...")
            X_train_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_trainingset_parent_{count}.csv")
            y_train = X_train_df[endpoint].values
            smiles_train = X_train_df["SMILES (Canonical)"].values
            X_train_df = X_train_df.drop(["SMILES (Canonical)", endpoint], axis=1)
            if "parent_SMILES" in X_train_df:
                X_train_df.drop(["parent_SMILES"], axis=1, inplace=True)
            X_train = X_train_df.values

            X_test_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_parent_{count}.csv")
            y_test = X_test_df[endpoint].values
            smiles_test = X_test_df["SMILES (Canonical)"].values
            X_test_df = X_test_df.drop(["SMILES (Canonical)", endpoint], axis=1)
            X_test = X_test_df.values
            
            ### Feature selection with lasso
            if feat_sel == True:       
                sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2020)
                model_sel = LassoCV(cv=sss, random_state=2020, n_jobs=4, max_iter=1000000, normalize=False)
                model_sel.fit(X_train_df.values, y_train)

                selected_feat = []
                for i in range(len(model_sel.coef_)):
                    if model_sel.coef_[i] != 0:
                        selected_feat.append(X_train_df.columns[i])

                # Store selected columns to filter them when applying the model on new compounds
                pd.DataFrame(selected_feat, columns=["Feature"]).to_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_parent_{count}_columns.csv', index=False)

                #selected_feat = X_train_df.columns[(model_sel.coef_ > 0)]
                print('total features: {}'.format((X_train_df.shape[1])))
                print('selected features: {}'.format(len(selected_feat)))
                print('features with coefficients shrank to zero: {}'.format(np.sum(model_sel.coef_ == 0)))

                # Save coefficients
                coeff = pd.DataFrame({"Coefficients": abs(model_sel.coef_)})
                lasso_coefficients = pd.DataFrame({"Feature": X_train_df.columns, "Coefficient": coeff["Coefficients"].values})
                selected_lasso_coefficients = lasso_coefficients[lasso_coefficients["Coefficient"] != 0]
                lasso_coefficients_cv = lasso_coefficients_cv.merge(lasso_coefficients, on="Feature", validate="one_to_one")

                X_train = X_train_df[selected_feat].values
                X_test = X_test_df[selected_feat].values
                X_train_df = X_train_df[selected_feat]
                X_test_df = X_test_df[selected_feat]
            
            ### Oversampling with SMOTEC
            if oversampling == True:
                # Get location of fingerprint columns (not just name of columns)
                fingColumns=[]
                col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower() or "bit_" in c.lower() or "transf" in c.lower() or "Num" in c or "Count" in c]]
                #col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower()]]
                for i in col:
                    fingColumns.append(X_train_df.columns.get_loc(i))

                sm = SMOTENC(categorical_features=fingColumns, sampling_strategy=0.8)
                #sm = SMOTE(sampling_strategy=0.8)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_train_df = pd.DataFrame(X_train)

            ### Create and train model on training set
            if method == "RF":
                model = RandomForestClassifier(n_estimators=num_trees, class_weight=class_weight, min_samples_leaf=3, random_state=2020, n_jobs=4)

            elif method == "LR":
                '''
                if feat_sel == False:
                    model = LogisticRegression(max_iter=800, C=0.8, random_state=2020, n_jobs=4, class_weight=class_weight)
                else:
                '''
                model = LogisticRegression(max_iter=1000, C=0.8, random_state=2020, n_jobs=4, class_weight=class_weight)
                #model = LogisticRegressionCV(Cs=10, penalty="l1", random_state=2020, n_jobs=4, cv=sss, class_weight=class_weight, solver="liblinear", max_iter=1000)

            elif method == "XGB":
                model = GradientBoostingClassifier(n_estimators=num_trees, random_state=2020, min_samples_leaf=2)

            elif method == "SVM":
                model = SVC(max_iter=-1, random_state=2020, probability=True, class_weight=class_weight)

            model.fit(X_train, y_train)

            ### Make predictions for test set
            pred_class = model.predict(X_test)
            temp = pd.DataFrame(model.predict_proba(X_test), columns=["0", "1"])
            pred_prob = temp["1"].values #column with the probability of class 1            
            
            # Print performance on test set
            result = self.calc_scores(y_test, pred_class, y_pred_prob=pred_prob, silent=True, modelname=f"{endpoint}_{desc_type}")
            all_results = pd.concat([all_results, result])
            
            # Extract feature importance values
            if feat_imp == True and method == "RF":
                if feat_sel == True:
                    X_columns = X_train_df[selected_feat].columns
                else:
                    X_columns = data_X_df.columns
                feat_importances = pd.DataFrame({'Importance':model.feature_importances_})    
                feat_importances['Feature'] = X_columns
                feat_importances_cv = feat_importances_cv.merge(feat_importances, on="Feature", validate="one_to_one")

            # Write prediction and further compound information for each split into output file
            long_line = ''
            for n, i in enumerate(X_test): 
                newline = f'{n}, {y_test[n]}, {pred_prob[n]}, {pred_class[n]}, {smiles_test[n][0]}, {count}\n'
                long_line += newline
            cv_outfile.write(long_line)

            # Collect models for later predictions of external test data
            models_cv.append(model)

            ### Save the model to disk
            filename = f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_parent_{count}.sav'
            cloudpickle.dump(model, open(filename, 'wb'))

            count = count + 1
            
        
        ### Calculate mean results on test set
        mean_results = self.calc_mean_results(all_results)
        #display(mean_results)
        

        if feat_sel == True:
            print("-------------------Coefficients CV-----------------")
            #lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_{count-1}.csv')
            mean_coef = lasso_coefficients_cv.mean(axis=1).values
            lasso_coefficients_cv_mean = pd.DataFrame({"Feature": lasso_coefficients_cv.Feature, "Mean coeff.": mean_coef})
            print(lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False)[:20])
            lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False).to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}.csv')
            lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_allCV.csv')

        if feat_imp == True and method == "RF":
            print("-------------------Importances CV-----------------")
            mean_imp = feat_importances_cv.mean(axis=1).values
            feat_importances_cv_mean = pd.DataFrame({"Feature": feat_importances_cv.Feature, "Mean importance": mean_imp})
            #display(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            print(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False).to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}.csv')
            
            feat_importances_cv.to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}_allCV.csv')
            std_imp = feat_importances_cv.std(axis=1).values           

        cv_outfile.close()

        # Return trained models
        return models_cv
    
    def train_models_metabolite_label(self, parents_df, metabolites_df, cv_folds, endpoint, endpoint_col, desc_type, num_trees, create_splits=False, method="RF", 
                                      feat_imp=True, feat_sel=False, class_weight="balanced", oversampling=False):
        
        ### Prepare output file
        cv_outfile = open(f'{self.crossvalidation_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}_metab.csv', 'w')
        header = f'name,{endpoint},probability,prediction,smiles,cv\n'
        cv_outfile.write(header)
        
        ### Create CV splits if the files don't exist already
        if create_splits == True:
            self.create_CV_splits(parents_df, metabolites_df, cv_folds, endpoint, endpoint_col)
        
        ### Initialization
        all_results = pd.DataFrame()
        models_cv = [] # Save all models to use for further prediction
        data_X_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_metab_1.csv") # to get the column names
        if feat_sel == True:
            lasso_coefficients_cv = pd.DataFrame({"Feature": data_X_df.columns}) # store coefficients from all CV runs
        if feat_imp == True:
            feat_importances_cv = pd.DataFrame({"Feature": data_X_df.columns})
        
        ### Train CV models
        for count in range(1, cv_folds+1):
            X_train_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_trainingset_metab_{count}.csv")
            y_train = X_train_df[endpoint].values
            smiles_train = X_train_df["SMILES (Canonical)"].values
            X_train_df = X_train_df.drop(["SMILES (Canonical)", endpoint], axis=1)
            X_train = X_train_df.values

            X_test_df = pd.read_csv(f"{self.mainPath}/data/splits/{endpoint}_testset_metab_{count}.csv")
            y_test = X_test_df[endpoint].values
            smiles_test = X_test_df["SMILES (Canonical)"].values
            X_test_df = X_test_df.drop(["SMILES (Canonical)", endpoint], axis=1)
            X_test = X_test_df.values
            
            #'''
            print(f"Training CV {count}...")
            ### Feature selection with lasso
            if feat_sel == True:    
                sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2020)
                model_sel = LassoCV(cv=sss, random_state=2020, n_jobs=4, max_iter=1000000, normalize=False)
                model_sel.fit(X_train_df.values, y_train)

                selected_feat = []
                for i in range(len(model_sel.coef_)):
                    if model_sel.coef_[i] != 0:
                        selected_feat.append(X_train_df.columns[i])

                # Store selected columns to filter them when applying the model on new compounds
                pd.DataFrame(selected_feat, columns=["Feature"]).to_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_metab_{count}_columns.csv', index=False)

                #selected_feat = X_train_df.columns[(model_sel.coef_ > 0)]
                print('total features: {}'.format((X_train_df.shape[1])))
                print('selected features: {}'.format(len(selected_feat)))
                print('features with coefficients shrank to zero: {}'.format(np.sum(model_sel.coef_ == 0)))

                # Save coefficients
                coeff = pd.DataFrame({"Coefficients": abs(model_sel.coef_)})
                lasso_coefficients = pd.DataFrame({"Feature": X_train_df.columns, "Coefficient": coeff["Coefficients"].values})
                selected_lasso_coefficients = lasso_coefficients[lasso_coefficients["Coefficient"] != 0]
                lasso_coefficients_cv = lasso_coefficients_cv.merge(lasso_coefficients, on="Feature", validate="one_to_one")

                X_train = X_train_df[selected_feat].values
                X_test = X_test_df[selected_feat].values
                X_train_df = X_train_df[selected_feat]
                X_test_df = X_test_df[selected_feat]
            
            ### Oversampling with SMOTEC
            if oversampling == True:
                # Get location of fingerprint columns (not just name of columns)
                fingColumns=[]
                col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower() or "bit_" in c.lower() or "transf" in c.lower() or "Num" in c or "Count" in c]]
                #col = X_train_df[[c for c in X_train_df.columns if 'byte vector' in c.lower()]]
                for i in col:
                    fingColumns.append(X_train_df.columns.get_loc(i))

                sm = SMOTENC(categorical_features=fingColumns, sampling_strategy=0.8)
                #sm = SMOTE(sampling_strategy=0.8)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_train_df = pd.DataFrame(X_train)

            ### Create and train model on training set
            if method == "RF":
                model = RandomForestClassifier(n_estimators=num_trees, class_weight=class_weight, min_samples_leaf=3, random_state=2020, n_jobs=4)

            elif method == "LR":
                model = LogisticRegression(max_iter=1000, C=0.8, random_state=2020, n_jobs=4, class_weight=class_weight)
                #model = LogisticRegressionCV(Cs=10, penalty="l1", random_state=2020, n_jobs=4, cv=sss, class_weight=class_weight, solver="liblinear", max_iter=1000)

            elif method == "XGB":
                model = GradientBoostingClassifier(n_estimators=num_trees, random_state=2020, min_samples_leaf=2)

            elif method == "SVM":
                model = SVC(max_iter=-1, random_state=2020, probability=True, class_weight=class_weight)

            model.fit(X_train, y_train)

            ### Make predictions for test set
            pred_class = model.predict(X_test)
            temp = pd.DataFrame(model.predict_proba(X_test), columns=["0", "1"])
            pred_prob = temp["1"].values #column with the probability of class 1            
            
            # Calculate performance on test set
            result = self.calc_scores(y_test, pred_class, y_pred_prob=pred_prob, silent=True, modelname=f"{endpoint}_{desc_type}")
            all_results = pd.concat([all_results, result])
            
            # Extract feature importance values
            if feat_imp == True and method == "RF":
                if feat_sel == True:
                    X_columns = X_train_df[selected_feat].columns
                else:
                    X_columns = data_X_df.columns
                feat_importances = pd.DataFrame({'Importance':model.feature_importances_})    
                feat_importances['Feature'] = X_columns
                feat_importances_cv = feat_importances_cv.merge(feat_importances, on="Feature", validate="one_to_one")

            # Write prediction and further compound information for each split into output file
            long_line = ''
            for n, i in enumerate(X_test): 
                newline = f'{n}, {y_test[n]}, {pred_prob[n]}, {pred_class[n]}, {smiles_test[n][0]}, {count}\n'
                long_line += newline
            cv_outfile.write(long_line)

            # Collect models for later predictions of external test data
            models_cv.append(model)

            ### Save the model to disk
            filename = f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_metab_{count}.sav'
            cloudpickle.dump(model, open(filename, 'wb'))
            #'''
            count = count + 1
            
        
        ### Calculate mean results on test set
        mean_results = self.calc_mean_results(all_results)
        #display(mean_results)
        

        if feat_sel == True:
            print("-------------------Coefficients CV-----------------")
            #lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_{count-1}.csv')
            mean_coef = lasso_coefficients_cv.mean(axis=1).values
            lasso_coefficients_cv_mean = pd.DataFrame({"Feature": lasso_coefficients_cv.Feature, "Mean coeff.": mean_coef})
            print(lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False)[:20])
            lasso_coefficients_cv_mean.sort_values(by="Mean coeff.", ascending=False).to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_metabLabels.csv')
            lasso_coefficients_cv.to_csv(f'{self.mainPath}/results/lassoCoefficients_{endpoint}_{desc_type}_allCV.csv')

        if feat_imp == True and method == "RF":
            print("-------------------Importances CV-----------------")
            mean_imp = feat_importances_cv.mean(axis=1).values
            feat_importances_cv_mean = pd.DataFrame({"Feature": feat_importances_cv.Feature, "Mean importance": mean_imp})
            #display(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            print(feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False)[:20])
            feat_importances_cv_mean.sort_values(by="Mean importance", ascending=False).to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}_metabLabels.csv')
            
            feat_importances_cv.to_csv(f'{self.mainPath}/results/featureImportance_{endpoint}_{desc_type}_metabLabels_allCV.csv')
            std_imp = feat_importances_cv.std(axis=1).values           

        cv_outfile.close()
        
        return
    
    
    
    def calc_probabilities(self, endpoint, X_test_df, cv_run, desc_type="chem", feat_sel=False, prepare_data=False, method="RF", metab_labels=False):
        
        if metab_labels == True:
            parent_or_metab = "metab"
        else:
            parent_or_metab = "parent"
            
        ### Prepare data 
        if prepare_data == True:
            # Remove columns with low variance
            vf_cols = pd.read_csv(f'{self.model_pickle_output_path}/model_{endpoint}_{desc_type}_{parent_or_metab}_variance_filter_columns.csv').values
            X_test_df = X_test_df[np.squeeze(vf_cols)]
            # Normalize all descriptors
            cols = X_test_df.columns
            scalerfile = f'{self.model_pickle_output_path}/normalizer/model_{endpoint}_{desc_type}_{parent_or_metab}.pkl'
            scaler = cloudpickle.load(open(scalerfile, 'rb'))
            X_test_norm = scaler.transform(X_test_df.values)
            X_test_df = pd.DataFrame(X_test_norm, columns=cols)

        
        ### Load model from CV run
        filename = f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_featSel={feat_sel}_{method}_{parent_or_metab}_{cv_run}.sav'
        loaded_model = cloudpickle.load(open(filename, 'rb'))
        if feat_sel == True:
            selectedColumns = pd.read_csv(f'{self.model_pickle_output_file}_{endpoint}_{desc_type}_{parent_or_metab}_{cv_run}_columns.csv').values.squeeze()
            X_test_df = X_test_df[[c for c in X_test_df.columns if c in selectedColumns]].values
        #pred_prob = loaded_model.predict_proba(X_test_df)   

        temp = pd.DataFrame(loaded_model.predict_proba(X_test_df), columns=["0", "1"])
        pred_prob = temp["1"].values #column with the probability of class 1 
        
        pred_prob_df = pd.DataFrame(pred_prob, columns=["probability"])
        
        return pred_prob_df

    
    def get_metabolites_from_parent(self, parent_smiles, metabolites_df, score_threshold=0, detox_phaseII=False, logP=-99):
            
        # Filter out compounds that are further metabolized in phase II reactions
        if detox_phaseII == True:
            metabolites_df = metabolites_df[metabolites_df["info_Detox_phaseII"] == False]
            metabolites_df = metabolites_df[metabolites_df["info_Phase"] != "Phase II"]
        
        if logP != -99:
            metabolites_df = metabolites_df[metabolites_df["info_LogP"] > logP]
            #metabolites_df = metabolites_df[metabolites_df["SlogP"] > logP]
            
        if score_threshold != 0: # all predicted metabolites
            metabolites_df = metabolites_df[metabolites_df["info_Absolute Score"] > score_threshold]
        
        metabolites_parent_df = metabolites_df[metabolites_df["parent_SMILES"] == parent_smiles]
        
        info_cols = [c for c in metabolites_parent_df.columns if "info_" in c]
        metabolites_parent_df = metabolites_parent_df.drop(info_cols, axis=1)

        return metabolites_parent_df
    
    def evaluate_test_sets(self, endpoint, metabolites_df, test_parents_path="", metab_labels=False, metab_model_for_parent=False, score_threshold=0, 
                           modes=["baseline", "only_parents", "mean", "median", "max_all", "max_metab"], 
                           cv_runs=5, feat_sel=False, silent=False, detox_phaseII=False, logP=-99, method="RF"):
        
        if test_parents_path == "":
            test_parents_path = f"{self.mainPath}/data/splits/"
        
        all_results = {}
        all_cv_results_dict = {}
        all_mean_results = pd.DataFrame()
        
        if metab_model_for_parent == True:
            if not "only_parent" in modes:
                modes.append("only_parents")
        else:
            if "only_parents" in modes:
                modes.remove("only_parents")
        for mode in modes:
            all_results[mode] = pd.DataFrame()
            all_cv_results_dict[mode] = pd.DataFrame()
    
        ### Evaluate on all cv_runs
        for run in range(1, cv_runs+1):
            print(f"Predicting probabilites of CV run {run}")
            # use the test set as prepared (variance filter and normalizer) for the model trained on the parents only or also on the labeled metabolites
            
            if metab_model_for_parent == False: 
                parents_df = pd.read_csv(f"{test_parents_path}/{endpoint}_testset_parent_{run}.csv")
            else:
                parents_df = pd.read_csv(f"{test_parents_path}/{endpoint}_testset_metab_{run}.csv")
                # Prepare also parent file for baseline
                parents_df_baseline = pd.read_csv(f"{test_parents_path}/{endpoint}_testset_parent_{run}.csv")
                drop_cols = [c for c in parents_df_baseline.columns if "transf_" in c or "smiles" in c.lower() or endpoint in c]
                parents_df_baseline.drop(drop_cols, axis=1, inplace=True)
            
            drop_cols = [c for c in parents_df.columns if "transf_" in c]
            parents_df.drop(drop_cols, axis=1, inplace=True)
            
            y_test = parents_df[endpoint]
            smiles_col = [c for c in parents_df.columns if "smiles" in c.lower()][0]
            smiles_parents = parents_df[smiles_col]
            parents_df.drop([endpoint, smiles_col], axis=1, inplace=True)

            ### Predict probabilities of parent compounds
            # Metab model for parent?
            if metab_model_for_parent == True:
                # Baseline model (with parent model)
                baseline_prob = self.calc_probabilities(endpoint, parents_df_baseline, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=False)
                parents_prob = self.calc_probabilities(endpoint, parents_df, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=True)
                only_parents_prob = parents_prob.copy()
                if not "only_parent" in modes:
                    modes.append("only_parents")
                
            else:
                baseline_prob = self.calc_probabilities(endpoint, parents_df, cv_run=run, feat_sel=feat_sel, prepare_data=False, method=method, metab_labels=False)
                parents_prob = baseline_prob.copy()
                if "only_parents" in modes:
                    modes.remove("only_parents")
            
            parents_prob["parent_SMILES"] = smiles_parents.values
            parents_mean_prob, parents_median_prob, parents_max_all_prob,  parents_max_metab_prob = [], [], [], []
                
            for parent in smiles_parents:
                # Get probability of this parent
                parent_prob = parents_prob[parents_prob["parent_SMILES"] == parent]
                #parent_prob = parent_prob["probability"].values[0]
                
                ### Predict probabilities of metabolites
                if modes != ["baseline"] or modes != ["only_parents"]: # not needed for baseline model
                    metabolites_parent_df = self.get_metabolites_from_parent(parent, metabolites_df, score_threshold,
                                                                             detox_phaseII=detox_phaseII, logP=logP)
                    #print(len(metabolites_parent_df))
                    if metabolites_parent_df.empty:
                        metabolites_prob = pd.DataFrame(columns=["probability"])
                    else:
                        drop_cols = [c for c in metabolites_parent_df.columns if "smiles" in c.lower()]
                        metabolites_desc = metabolites_parent_df.drop(drop_cols, axis=1)
                        metabolites_prob = self.calc_probabilities(endpoint, metabolites_desc, cv_run=run, feat_sel=feat_sel, prepare_data=True, method=method, metab_labels=metab_labels)
                #display(metabolites_parent_df)
                
                ### Averaging mode for the final predicted probability
                #if not ["baseline", "mean", "median", "max_all", "max_metab"] in modes:
                #    print("Accepted modes: baseline, mean, median, max_all, max_metab")
                #    return
                '''
                if "baseline" in modes:
                    # Return only the probability of the parent compound (to test if we get the same result than in the CV workflow)
                    baseline_prob.append(parent_prob["probability"].values[0])
                '''    
                if "mean" in modes:
                    # Mean probability of metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    mean_prob = all_prob["probability"].mean(axis=0)
                    parents_mean_prob.append(mean_prob)
                    
                if "median" in modes:
                    # Median probability of metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    median_prob = all_prob["probability"].median(axis=0)
                    parents_median_prob.append(median_prob)
                    
                if "max_all" in modes:
                    # Max probability from metabolites and the parent probability
                    all_prob = pd.concat([parent_prob, metabolites_prob])
                    max_all_prob = all_prob["probability"].max(axis=0)
                    parents_max_all_prob.append(max_all_prob)
                    
                if "max_metab" in modes:
                    # Mean between the max probability for a metabolite and the parent probability
                    max_metabolites_prob = metabolites_prob["probability"].max(axis=0)
                    all_prob = pd.concat([parent_prob, pd.DataFrame([max_metabolites_prob], columns=["probability"])])
                    max_metab_prob = all_prob["probability"].mean(axis=0)
                    parents_max_metab_prob.append(max_metab_prob)
                    
            
            if "baseline" in modes:
                parents_prob["baseline_probability"] = baseline_prob
                parents_prob["baseline_class"] = np.where(parents_prob['baseline_probability']>=0.5, 1, 0)
                
            if "only_parents" in modes:
                parents_prob["only_parents_probability"] = only_parents_prob
                parents_prob["only_parents_class"] = np.where(parents_prob['only_parents_probability']>=0.5, 1, 0)
                    
            if "mean" in modes:
                parents_prob["mean_probability"] = parents_mean_prob
                parents_prob["mean_class"] = np.where(parents_prob['mean_probability']>=0.5, 1, 0)
            
            if "median" in modes:
                parents_prob["median_probability"] = parents_median_prob
                parents_prob["median_class"] = np.where(parents_prob['median_probability']>=0.5, 1, 0)
            
            if "max_all" in modes:
                parents_prob["max_all_probability"] = parents_max_all_prob
                parents_prob["max_all_class"] = np.where(parents_prob['max_all_probability']>=0.5, 1, 0)
                
            if "max_metab" in modes:
                parents_prob["max_metab_probability"] = parents_max_metab_prob
                parents_prob["max_metab_class"] = np.where(parents_prob['max_metab_probability']>=0.5, 1, 0)
            
            
            ### Calculate results of CV run
            for mode in modes:
                result = self.calc_scores(y_test.values.astype('int'), parents_prob[f"{mode}_class"].values.astype('int'), y_pred_prob=parents_prob[f"{mode}_probability"].values, silent=True, modelname=f"{endpoint}_metab_{mode}")
                all_results[mode] = pd.concat([all_results[mode], result])

                # Save all information for output file
                cv_results = pd.DataFrame({"smiles": smiles_parents, f"{endpoint}": y_test, "probability": parents_prob[f"{mode}_probability"].values, 
                                           "prediction": parents_prob[f"{mode}_class"].values, "cv": run})
                all_cv_results_dict[mode] = pd.concat([all_cv_results_dict[mode], cv_results], axis=0)
        
        for mode in modes:
            ### Print out cross-validation file with predictions
            all_cv_results_dict[mode].to_csv(f'{self.crossvalidation_output_file}_{endpoint}_metab_{mode}_featSel=False_scoreThres={score_threshold}_detoxPhaseII={detox_phaseII}_logP={logP}.csv', index=False)
            
            ### Calculate mean results of CV runs on test sets with metabolites
            mean_results = self.calc_mean_results(all_results[mode])
            
            ### Calculate significance for each model and print it out
            p_values = self.calc_significance_one_model(endpoint, mode=mode, results=all_cv_results_dict[mode], results_baseline=all_cv_results_dict["baseline"], feat_sel=feat_sel, significance_level=0.2, score_threshold=score_threshold,
                                                        detox_phaseII=detox_phaseII, logP=logP)
            
            mean_results = pd.concat([mean_results, p_values], axis=1)

            ### Prepare output file with mean results
            mean_results.insert(loc=0, column='mode', value=mode)
            mean_results.insert(loc=0, column='detox_phaseII', value=detox_phaseII)
            mean_results.insert(loc=0, column='logP', value=logP)
            mean_results.insert(loc=0, column='score_threshold', value=score_threshold)
            mean_results.insert(loc=0, column='endpoint', value=endpoint)
            all_mean_results = pd.concat([all_mean_results, mean_results], axis=0)
            if silent == False:
                print(all_mean_results)
        
        ### Calculate significance
        significance_result = self.calc_significance(endpoint, modes=modes, feat_sel=feat_sel, significance_level=0.2,
                                                     score_threshold=score_threshold, detox_phaseII=detox_phaseII, logP=logP)
        
        
        return all_mean_results
    
    
    def calc_metrics_per_run(self, all_results, cv_numbers, endpoint, mode, significance_level, eval_df, metrics_dict_f1={}, 
                             metrics_dict_mcc={}, metrics_dict_recall={}, metrics_dict_precision={}):
        
        # Evaluate model internal (CV) and external data
        eval_df_1 = pd.DataFrame(columns=['Value', 'Metric', 'Descriptor'])
        
        f1_cv = pd.DataFrame(columns=['F1 score'])
        mcc_cv = pd.DataFrame(columns=['MCC'])
        recall_cv = pd.DataFrame(columns=['Recall'])
        precision_cv = pd.DataFrame(columns=['Precision'])
        
        for i in cv_numbers:
            results_df_cv = all_results[all_results["cv"] == i]
            result_metrics = self.calc_scores(results_df_cv[endpoint].values, results_df_cv["prediction"].values, results_df_cv["probability"].values, silent=True)

            f1_cv = pd.concat([f1_cv, pd.DataFrame(result_metrics["F1 score"].values, columns=["F1 score"])], axis=0)
            mcc_cv = pd.concat([mcc_cv, pd.DataFrame(result_metrics["MCC"].values, columns=["MCC"])], axis=0)
            recall_cv = pd.concat([recall_cv, pd.DataFrame(result_metrics["Recall"])])
            precision_cv = pd.concat([precision_cv, pd.DataFrame(result_metrics["Precision"])])
            
        metrics = {'F1 score': f1_cv, 'MCC': mcc_cv, 'Recall': recall_cv, 'Precision': precision_cv}
        #display(metrics)

        metrics_dict_f1[mode] = metrics["F1 score"]["F1 score"].values
        metrics_dict_mcc[mode] = metrics["MCC"]["MCC"].values
        metrics_dict_recall[mode] = metrics["Recall"]["Recall"].values
        metrics_dict_precision[mode] = metrics["Precision"]["Precision"].values

        for m in ['F1 score', 'MCC', 'Recall', 'Precision']:
            eval_df_1["Value"] = metrics[m][m]
            eval_df_1["Metric"] = m
            eval_df_1["Mode"] = mode
            eval_df = pd.concat([eval_df, eval_df_1])
        
        return eval_df, metrics_dict_f1, metrics_dict_mcc, metrics_dict_recall, metrics_dict_precision

    
    def calc_significance(self, endpoint, modes=['baseline', 'only_parents', 'mean', 'median', 'max_all', 'max_metab'], feat_sel=False, significance_level=0.2, 
                          score_threshold=0, detox_phaseII=True, logP=-99):
        
        eval_df = pd.DataFrame(columns=['Value', 'Metric', 'Mode'])
        metrics_dict_precision = {}
        metrics_dict_recall = {}
        metrics_dict_f1 = {}
        metrics_dict_mcc = {}
        
        result_base_parents = {}
        result_base_mean = {}
        result_base_median = {}
        result_base_max = {}
        result_base_max_metab = {}

        for mode in modes:
            # Load output dataframe with p-values
            cv_filename = f'{self.crossvalidation_output_file}_{endpoint}_metab_{mode}_featSel={feat_sel}_scoreThres={score_threshold}_detoxPhaseII={detox_phaseII}_logP={logP}.csv'
            all_results = pd.read_csv(cv_filename)
            cv_numbers = all_results.cv.unique()

            eval_df, metrics_dict_f1, metrics_dict_mcc, metrics_dict_recall, metrics_dict_precision = self.calc_metrics_per_run(all_results, cv_numbers, endpoint, mode,
                                                                                                                significance_level,eval_df, metrics_dict_f1, 
                                                                                                                metrics_dict_mcc,
                                                                                                                metrics_dict_recall, metrics_dict_precision) 
        signif_results = {}
        # Calculate significance
        for metric in ["f1 score", "mcc", "recall", "precision"]:

            if metric == "f1 score":
                metrics_dict = metrics_dict_f1
            if metric == "mcc":
                metrics_dict = metrics_dict_mcc
            if metric == "recall":
                metrics_dict = metrics_dict_recall
            elif metric == "precision":
                metrics_dict = metrics_dict_precision

            signif_results[metric] = []
            print (f"{metric}:")
            if "only_parents" in modes:
                result_base_parents[metric] = mannwhitneyu(metrics_dict["baseline"], metrics_dict["only_parents"], use_continuity=True, alternative='two-sided')[1]
                if result_base_parents[metric] < 0.05:
                    signif_results[metric].append("only_parents")
                print ("p value baseline vs. only parents:", round(result_base_parents[metric], 3))
            
            if "mean" in modes:
                result_base_mean[metric] = mannwhitneyu(metrics_dict["baseline"], metrics_dict["mean"], use_continuity=True, alternative='two-sided')[1]
                if result_base_mean[metric] < 0.05:
                    signif_results[metric].append("mean")
                print ("p value baseline vs. mean:", round(result_base_mean[metric], 3))
                
            if "median" in modes:
                result_base_median[metric] = mannwhitneyu(metrics_dict["baseline"], metrics_dict["median"], use_continuity=True, alternative='two-sided')[1]
                if result_base_median[metric] < 0.05:
                    signif_results[metric].append("median")
                print ("p value baseline vs. median:", round(result_base_median[metric], 3))
                
            if "max_all" in modes:
                result_base_max[metric] = mannwhitneyu(metrics_dict["baseline"], metrics_dict["max_all"], use_continuity=True, alternative='two-sided')[1]
                if result_base_max[metric] < 0.05:
                    signif_results[metric].append("max_all")
                print ("p value baseline vs. max all:", round(result_base_max[metric], 3))
                
            if "max_metab" in modes:
                result_base_max_metab[metric] = mannwhitneyu(metrics_dict["baseline"], metrics_dict["max_metab"], use_continuity=True, alternative='two-sided')[1]
                if result_base_max_metab[metric] < 0.05:
                    signif_results[metric].append("max_metab")
                print ("p value baseline vs. max metab:", round(result_base_max_metab[metric], 3))
                
        print(f"\nSummary: significant result at p-value 0.05 for:")        
        for metric in ["f1 score", "mcc", "recall", "precision"]:
            print(f"{metric}: {signif_results[metric]}")
            
        return metrics_dict_f1, metrics_dict_mcc, metrics_dict_recall, metrics_dict_precision
    
    
    def calc_significance_one_model(self, endpoint, mode, results, results_baseline, feat_sel=False, significance_level=0.2, score_threshold=0, 
                                    detox_phaseII=True, logP=-99, silent=False):
        eval_df = pd.DataFrame(columns=['Value', 'Metric', 'Mode'])
        metrics_dict_precision = {}
        metrics_dict_recall = {}
        metrics_dict_f1 = {}
        metrics_dict_mcc = {}
        
        cv_numbers = results.cv.unique()
        
        cv_results = {mode: results, "baseline": results_baseline}

        for md in ["baseline", mode]:
            eval_df, metrics_dict_f1, metrics_dict_mcc, metrics_dict_recall, metrics_dict_precision = self.calc_metrics_per_run(cv_results[md], cv_numbers, endpoint, md,
                                                                                                            significance_level,eval_df, metrics_dict_f1, 
                                                                                                            metrics_dict_mcc,
                                                                                                            metrics_dict_recall, metrics_dict_precision) 
        signif_results = {}
        p_values = pd.DataFrame({"p-value f1 score": [np.NaN], "p-value mcc": [np.NaN], "p-value precision": [np.NaN], "p-value recall": [np.NaN]})
        # Calculate significance
        for metric in ["f1 score", "mcc", "recall", "precision"]:

            if metric == "f1 score":
                metrics_dict = metrics_dict_f1
            if metric == "mcc":
                metrics_dict = metrics_dict_mcc
            if metric == "recall":
                metrics_dict = metrics_dict_recall
            elif metric == "precision":
                metrics_dict = metrics_dict_precision

            signif_results[metric] = []
            p_values[f"p-value {metric}"].loc[0] = mannwhitneyu(metrics_dict["baseline"], metrics_dict[mode], use_continuity=True, alternative='two-sided')[1]
            
            if silent == False:
                print (f"{metric}:")
                print(p_values[f"p-value {metric}"].loc[0])
                       
        return p_values