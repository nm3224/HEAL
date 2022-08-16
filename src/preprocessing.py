import argparse
from cgitb import text
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import re 
from text_utils import RCDC_terms, clean_data, create_columns, filter_sen, change_columns, create_columns, pain_oud_split
import pdb 
import warnings
warnings.filterwarnings("ignore")

#Create arguments to access and save new excel files
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/mayatnf/HEAL/original_data/HEAL_GRANTS.xlsx", help="path for where excel for where HEAL grant data is saved", type=str)
    parser.add_argument("--pain_path", default="/Users/mayatnf/HEAL/original_data/pain_data.xlsx", help="path for where excel for where additional pain dataset is saved", type=str)
    parser.add_argument("--cleaned", default="/Users/mayatnf/HEAL/cleaned_data/cleaned_HEAL_data.xlsx", help="path for where excel for clean data", type=str)
    parser.add_argument("--outcome_combined", default="/Users/mayatnf/HEAL/cleaned_data/outcome_combined_data.xlsx", help="path for where excel for where additional pain dataset is added", type=str)
    parser.add_argument("--oud_path", default="/Users/mayatnf/HEAL/original_data/NIDA_oud.xlsx", help="path for where excel for where additional oud dataset is added", type=str)
    parser.add_argument("--pain_mentions", default="/Users/mayatnf/HEAL/cleaned_data/oud_RCDC_mentions_pain_data.xlsx", help="path for where excel for where additional oud dataset is added", type=str)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    
    #Read in excel files into dataframes
    df_heal = pd.read_excel(args.data_path, sheet_name = 0)
    df_pain = pd.read_excel(args.pain_path, sheet_name = 1)
    df_oud = pd.read_excel(args.oud_path, sheet_name = 0)
    
    #Check for pain terms
    RCDC_terms(df_oud, 'RCDC Categories', args.pain_mentions)
    
    #Remove stopwords from abstracts. Alter variable text_cols based on column names for text 
    text_cols = ['Abstract', 'Specific Aims', 'Public Health Relevance']
    df_heal = clean_data(df_heal, text_cols)
    
    #note that the number in brackets depends on how many columns passed in text_cols. alter for less columns
    df_heal['Combined Cleaned'] = df_heal[f'Cleaned {text_cols[0]}'] + ' ' + df_heal[f'Cleaned {text_cols[1]}'] + ' ' + df_heal[f'Cleaned {text_cols[2]}']
    df_heal['Combined Filtered'] = df_heal[f'Filtered {text_cols[0]}'] + df_heal[f'Filtered {text_cols[1]}'] + df_heal[f'Filtered {text_cols[2]}']

    #Create Science Type Columns for HEAL dataset
    df_heal.replace('', np.nan, inplace=True)
    df_heal = df_heal.dropna(subset = ['Science', 'Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses'])
    df_heal = create_columns(df_heal, ['EPIDEMIOLOGICAL', 'DISEASE-RELATED BASIC', 'HEALTH SERVICES RESEARCH', 'IMPLEMENTATION RESEARCH'])
    df_heal = change_columns(df_heal, ['Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses'])
    
    #Save Cleaned HEAL data so don't have to rerun in future
    df_heal.to_excel(args.cleaned)
    
    #Repeat cleaning for OUD data
    text_cols = ['Abstract', 'Specific Aims', 'Public Health Relevance']
    df_oud = clean_data(df_oud, text_cols)
    
    df_oud['Combined Cleaned'] = df_oud[f'Cleaned {text_cols[0]}'] + ' ' + df_oud[f'Cleaned {text_cols[1]}'] + ' ' + df_oud[f'Cleaned {text_cols[2]}']
    df_oud['Combined Filtered'] = df_oud[f'Filtered {text_cols[0]}'] + df_oud[f'Filtered {text_cols[1]}'] + df_oud[f'Filtered {text_cols[2]}']
    
    #Add a primary outcome column for training/testing
    df_oud['HEAL Category- Primary Outcome'] = 'OUD'

    #Cleaning for pain data--some differences from HEAL/OUD datasets b/c of excel sheet formatting 
    df_pain = clean_data(df_pain, ['Abstract Text'])
    df_pain = df_pain.rename(columns={'Cleaned Abstract Text': 'Combined Cleaned'})
    df_pain = df_pain.rename(columns={'Filtered Abstract Text': 'Combined Filtered'})
    df_pain = df_pain.rename(columns={'APPL ID': 'Appl ID'})
    
    #Save Cleaned and combined data for all 3 datasets
    combined = df_heal[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'HEAL Category- Primary Outcome']].append(df_pain[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'HEAL Category- Primary Outcome']])
    combined.append(df_oud[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'HEAL Category- Primary Outcome']])
    
    #Split pain/oud columns
    combined = pain_oud_split(combined)
    pdb.set_trace()
    combined.to_excel(args.outcome_combined) 

if __name__ == "__main__":
    main()