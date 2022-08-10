import argparse
from cgi import test
from random import random
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from numpy import sqrt, argmax
from sklearn.metrics import roc_curve
import imblearn
import ast
from operator import itemgetter
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from functools import reduce
from sklearn import  model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import pdb 
import warnings
from key_term_search import primary_outcome_improved
from text_utils import visuals, tuning, data_dist
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/mayatnf/HEAL/original_daya/HEAL_GRANTS.xlsx", help="path for where excel for where HEAL grant data is saved", type=str)
    parser.add_argument("--matched_path", default="/Users/mayatnf/HEAL/results/matched_models.xlsx", help="path for where excel for where ML model and NLP rule based models data matches/mismatches is saved", type=str)
    parser.add_argument("--clintrials_path", default="/Users/mayatnf/HEAL/cleaned_data/heal_clintrials_cleaned.csv", help="path for where excel for where clinical grant data is saved", type=str)
    parser.add_argument("--cleaned", default="/Users/mayatnf/HEAL/cleaned_data/cleaned_data.xlsx", help="path for where excel for clean data", type=str)
    parser.add_argument("--outcome_combined", default="/Users/mayatnf/HEAL/cleaned_data/outcome_combined_data.xlsx", help="path for where excel for where additional pain dataset is added", type=str)
    parser.add_argument("--science_preds", default="/Users/mayatnf/HEAL/results/predictions_ML/science_preds_ML.xlsx", help="path for where excel for where science type predictions are saved", type=str)
    parser.add_argument("--outcome_preds", default="/Users/mayatnf/HEAL/results/predictions_ML/outcome_preds_ML.xlsx", help="path for where excel for where outcome type predictions are saved", type=str)
    parser.add_argument("--milestone_preds", default="/Users/mayatnf/HEAL/results/predictions_ML/milestone_preds_ML.xlsx", help="path for where excel for where milestone type predictions are saved", type=str)
    parser.add_argument("--visuals", default="/Users/mayatnf/HEAL/results/visuals/", help="path for where excel for where visuals are saved", type=str)

    args = parser.parse_args()
    return args

#FOR OUTCOME AND MILESTONE
def all_pipelines(df, text_col, label, use_smote):

    args = get_args()
    visuals(df[label], args.visuals, f"overall_{label}")

    # Split the data into training and testing sets
    train_df, test_df = model_selection.train_test_split(df, test_size=0.25)
    Train_X = train_df[text_col]
    Train_Y = train_df[label]
    Test_X = test_df[text_col]
    Test_Y = test_df[label]

    # Instantiate a random forest classifier using pipeline method
    if use_smote == True: 
        rf_textclassifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=100)),
            ('smote', SMOTE()),
            ('rf', RandomForestClassifier(n_estimators = 1000))
            ])
        visuals(Train_Y, args.visuals, f"before_smote_train_{label}")
    else:
        rf_textclassifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=100)),
            ('rf', RandomForestClassifier(n_estimators = 1000))
            ])
    
    if use_smote == True:
        oversample = SMOTE()
        vectorizer = TfidfVectorizer()
        x_tf = vectorizer.fit_transform(Train_X)
        x, y = oversample.fit_resample(x_tf, Train_Y)
        visuals(y, args.visuals,  f"after_smote_train_{label}")
    
    #Fit the model
    rf_textclassifier.fit(Train_X, Train_Y)
    
    #Make predictions
    rf_predictions = rf_textclassifier.predict(Test_X)

    #Compute Accuracy
    print("RF Accuracy Score -> ",accuracy_score(rf_predictions, Test_Y)*100)

    # Use f1 score function to compute accuracy
    micro_f1 = f1_score(Test_Y, rf_predictions, average='micro')
    #print(f"F1 Score: {micro_f1*100}")

    test_df[f"{label}_ML"] = rf_predictions
    return test_df[['Appl ID', 'Combined Cleaned', label, f"{label}_ML"]]

#Science- Basic, Clinical, Health Services Research, Implementation Research
def knn_classifier(train_df, test_df, text_col, label):
    args = get_args()

    Train_X = train_df[text_col]
    Train_Y = train_df[label]
    Test_X = test_df[text_col]
    Test_Y = test_df[label]

    #Hyperparameter tuning to choose best leaf size, k and p
    vectorizer = TfidfVectorizer()
    x_tf = vectorizer.fit_transform(Train_X)
    best_leaf_size, best_p, best_n = tuning(x_tf, Train_Y)

    #Build the K-Nearest Neighbors Classifier
    if label != 'Science- Clinical':
        knn_textclassifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=100)),
            ('knn', KNeighborsClassifier())
            ])
    else:
        knn_textclassifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=100)),
            ('smote', SMOTE()),
            ('knn', KNeighborsClassifier())
            ])

    #Fit the model
    knn_textclassifier.fit(Train_X, Train_Y)
    
    knn_predictions = knn_textclassifier.predict(Test_X)

    # Print accuracy score
    print(f"KNN Accuracy Score {label} -> ",accuracy_score(knn_predictions, Test_Y)*100)
    test_df[f"{label}_ML"] = knn_predictions 
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(1, 'Yes')
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(0, 'No')
    return test_df[['Appl ID', 'Combined Cleaned', 'Science', label, f"{label}_ML"]]   

def rf_classifier(train_df, test_df, text_col, label):
    # Split the data into training and testing sets
    Train_X = train_df[text_col]
    Test_X = test_df[text_col]
    Train_Y = train_df[label]
    Test_Y = test_df[label]

    # Instantiate a random forest classifier using pipeline method
    rf_textclassifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=100)),
        ('smote', SMOTE()), 
        ('rf', RandomForestClassifier(n_estimators = 1000))
        ])
    
    #Fit the model
    rf_textclassifier.fit(Train_X, Train_Y)
    rf_predictions = rf_textclassifier.predict(Test_X)
    
    print(f"RF Accuracy Score {label} -> ",accuracy_score(rf_predictions, Test_Y)*100)
    test_df[f"{label}_ML"] = rf_predictions 
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(1, 'Yes')
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(0, 'No')
    return test_df[[ 'Appl ID', 'Combined Cleaned', 'Science', label, f"{label}_ML"]] 

def svm_classifier(train_df, test_df, text_col, label):
    
    # Split the data into training and testing sets
    Train_X = train_df[text_col]
    Test_X = test_df[text_col]
    Train_Y = train_df[label]
    Test_Y = test_df[label]

    #Build the SVM Classifier
    svm_textclassifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=100)),
        ('smote', SMOTE()),
        ('svm', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
        ])

    #Fit the model
    svm_textclassifier.fit(Train_X, Train_Y)
    svm_predictions = svm_textclassifier.predict(Test_X)

    print(f"SVM Accuracy Score {label} -> ",accuracy_score(svm_predictions, Test_Y)*100) 
    test_df[f"{label}_ML"] = svm_predictions
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(1, 'Yes')
    test_df[f"{label}_ML"] =test_df[f"{label}_ML"].replace(0, 'No')
    return test_df[['Appl ID', 'Combined Cleaned', 'Science', label, f"{label}_ML"]] 

def lr_classifier(train_df, test_df, text_col, label):
    # Split the data into training and testing sets
    Train_X = train_df[text_col]
    Test_X = test_df[text_col]
    Train_Y = train_df[label]
    Test_Y = test_df[label]

    #Build Logistic Regression Classifier
    lr_textclassifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=100)),
        ('smote', SMOTE()), 
        ('log', LogisticRegression())
        ])

    #Fit the model
    lr_textclassifier.fit(Train_X, Train_Y)
    lr_predictions = lr_textclassifier.predict(Test_X)

    print(f"Logistic Regression Accuracy Score {label} -> ",accuracy_score(lr_predictions, Test_Y)*100)
    test_df[f"{label}_ML"] = lr_predictions 
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(1, 'Yes')
    test_df[f"{label}_ML"] = test_df[f"{label}_ML"].replace(0, 'No')
    return test_df[['Appl ID', 'Combined Cleaned', 'Science', label, f"{label}_ML"]] 

def main(): 

    args = get_args()

    #Read in excel files into dataframes
    df_heal = pd.read_excel(args.cleaned)
    df_pain_heal = pd.read_excel(args.outcome_combined)
    
    # For Science Type & Milestones
    # Split the data into training and testing sets
    train_df, test_df = model_selection.train_test_split(df_heal, test_size=0.25)
    
    #Visual training data distribution
    #data_dist(train_df, ['Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses', 'EPIDEMIOLOGICAL', 'DISEASE-RELATED BASIC', 'HEALTH SERVICES RESEARCH', 'IMPLEMENTATION RESEARCH'], args.visuals)

    #Categorize Outcomes using model-- using smote 
    # Drops empty rows-- or, make sure EVERY row is filled with correct info, otherwise there will be missing unlabeled studies in the product.
    #df_cleaned = df_pain_heal[['Combined Cleaned', 'HEAL Category- Primary Outcome', 'Appl ID']].dropna()
    #all_pipelines(df_cleaned, 'Combined Cleaned', 'HEAL Category- Primary Outcome', True).to_excel(args.outcome_preds)
    
    #Categorize Milestones using model
    #all_pipelines(df_heal, 'Combined Cleaned', 'Milestones', False).to_excel(args.milestone_preds)
    
    #K-Nearest Neighbors Classifiers
    dfs = []
    for label in ['Science- Basic', 'Science- Clinical', 'HEALTH SERVICES RESEARCH', 'IMPLEMENTATION RESEARCH']:
        new = knn_classifier(train_df, test_df, 'Combined Cleaned', label)
        dfs.append(new)

    #Logistic Regression Classifiers
    new = lr_classifier(train_df, test_df, 'Combined Cleaned', 'DISEASE-RELATED BASIC')
    dfs.append(new)
    lr_classifier(train_df, test_df, 'Combined Cleaned', 'Science- Clinical')

    #Random Forest Classifiers
    for label in ['Science- Translational', 'Science- Systematic Meta-analyses']:
        new = rf_classifier(train_df, test_df, 'Combined Cleaned', label)
        dfs.append(new)

    #SVM Classifiers
    for label in ['Science- Core Services', 'EPIDEMIOLOGICAL']:
        new = svm_classifier(train_df, test_df, 'Combined Cleaned', label)
        dfs.append(new)
    
    predictions = reduce(lambda  left,right: pd.merge(left,right,on=['Appl ID'], how='outer'), dfs)
    predictions[['Appl ID', 'Combined Cleaned', 'Science', 'Science- Basic_ML', 'Science- Translational_ML', 'Science- Clinical_ML', 'Science- Core Services_ML', 'Science- Systematic Meta-analyses_ML', 'EPIDEMIOLOGICAL_ML', 'DISEASE-RELATED BASIC_ML', 'HEALTH SERVICES RESEARCH_ML', 'IMPLEMENTATION RESEARCH_ML']].to_excel(args.science_preds)
    
    pdb.set_trace()

    #needs work
    predictions['Science_1'] = np.empty((len(predictions), 0)).tolist()
    for i, row in predictions.iterrows():
        for scitype in ['Science- Basic_ML', 'Science- Translational_ML', 'Science- Clinical_ML', 'Science- Core Services_ML', 'Science- Systematic Meta-analyses_ML', 'EPIDEMIOLOGICAL_ML', 'DISEASE-RELATED BASIC_ML', 'HEALTH SERVICES RESEARCH_ML', 'IMPLEMENTATION RESEARCH_ML']:
            if row[scitype] == 'Yes':
                if 'Science' in scitype:
                    st = scitype.split('- ')
                    st = st[1].split('_')[0].upper()
                else:
                    st = scitype.split('_')[0]
                predictions['Science_1'][i].append(str(st)) 
    
    predictions.Science = predictions.Science.apply(ast.literal_eval)
    predictions['match'] = np.where(predictions['Science'].apply(set) == predictions['Science_1'].apply(set), 'yes', 'no')
    print(f'Accuracy:', len(predictions.loc[predictions['match'] == 'yes']) / len(predictions) * 100)
    predictions[['Appl ID', 'Combined Cleaned', 'Science', 'Science- Basic_ML', 'Science- Translational_ML', 'Science- Clinical_ML', 'Science- Core Services_ML', 'Science- Systematic Meta-analyses_ML', 'EPIDEMIOLOGICAL_ML', 'DISEASE-RELATED BASIC_ML', 'HEALTH SERVICES RESEARCH_ML', 'IMPLEMENTATION RESEARCH_ML', 'Science_1']].to_excel(args.science_preds)

if __name__ == "__main__":
    main()