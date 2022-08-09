import argparse
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import re 
from text_utils import clean_data, create_columns, filter_sen, change_columns, create_columns, find_words
import pdb 
import warnings
warnings.filterwarnings("ignore")

#Create arguments to access and save new excel files
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/mayatnf/HEAL/HEAL_GRANTS.xlsx", help="path for where excel for where HEAL grant data is saved", type=str)
    parser.add_argument("--alldata_path", default="/Users/mayatnf/HEAL/ALL_GRANTS.xlsx", help="path for where excel for where ALL grant data is saved", type=str)
    parser.add_argument("--pain_path", default="/Users/mayatnf/HEAL/pain_data.xlsx", help="path for where excel for where additional pain dataset is saved", type=str)
    parser.add_argument("--cleaned", default="/Users/mayatnf/HEAL/cleaned_data.xlsx", help="path for where excel for clean data", type=str)
    parser.add_argument("--outcome_combined", default="/Users/mayatnf/HEAL/outcome_combined_data.xlsx", help="path for where excel for where additional pain dataset is added", type=str)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    #Read in excel files into dataframes
    df_heal = pd.read_excel(args.data_path, sheet_name = 0)
    #df_all = pd.read_excel(args.alldata_path, sheet_name = 0)
    #df_pain = pd.read_excel(args.pain_path, sheet_name = 1)
    
    #Remove stopwords from abstracts. Alter variable text_cols based on column names for text 
    text_cols = ['Abstract', 'Specific Aims', 'Public Health Relevance']
    df_heal = clean_data(df_heal, text_cols)
    
    #note that the number in brackets depends on how many columns passed in text_cols. alter for less columns
    df_heal['Combined Cleaned'] = df_heal[f'Cleaned {text_cols[0]}'] + ' ' + df_heal[f'Cleaned {text_cols[1]}'] + ' ' + df_heal[f'Cleaned {text_cols[2]}']
    df_heal['Combined Filtered'] = df_heal[f'Filtered {text_cols[0]}'] + df_heal[f'Filtered {text_cols[1]}'] + df_heal[f'Filtered {text_cols[2]}']
    
    #Create Science Type Columns
    df_heal.replace('', np.nan, inplace=True)
    df_heal = df_heal.dropna(subset = ['Science', 'Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses'])
    df_heal = create_columns(df_heal, ['EPIDEMIOLOGICAL', 'DISEASE-RELATED BASIC', 'HEALTH SERVICES RESEARCH', 'IMPLEMENTATION RESEARCH'])
    df_heal = change_columns(df_heal, ['Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses'])
    
    #Save Cleaned data so don't have to rerun
    #df_heal.to_excel(args.cleaned)
    df_cleaned = pd.read_excel(args.cleaned)

    #Save Cleaned pain data
    #df_pain_clean = clean_data(df_pain, ['Abstract Text'])
    #df_pain_clean = df_pain_clean.rename(columns={'Cleaned Abstract Text': 'Combined Cleaned'})
    #df_pain_clean = df_pain_clean.rename(columns={'Filtered Abstract Text': 'Combined Filtered'})
    #df_pain_clean = df_pain_clean.rename(columns={'APPL ID': 'Appl ID'})
    
    #Save Cleaned and combined pain data
    #combined = df_heal[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'HEAL Category- Primary Outcome']].append(df_pain_clean[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'HEAL Category- Primary Outcome']])
    #combined.to_excel(args.outcome_combined) 
   
    find_words(df_cleaned, 'HEAL Category- Primary Outcome', ['Pain', 'OUD', 'Both'], 'Combined Cleaned')
    #find_words(df_cleaned, '', [], 'Combined Cleaned')

if __name__ == "__main__":
    main()