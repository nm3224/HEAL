import argparse
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import re 
from codes import pain, oud, milestones, clinical, translational, implementation, basic, epi, systematic, core_services, disease, health_services 
import pdb 
import warnings
import ast
warnings.filterwarnings("ignore")

#Create arguments to access and save new excel files
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/mayatnf/HEAL/original_data/HEAL_GRANTS.xlsx", help="path for where excel for where HEAL grant data is saved", type=str)
    parser.add_argument("--clintrials_path", default="/Users/mayatnf/HEAL/cleaned_data/heal_clintrials_cleaned.csv", help="path for where excel for where clinical grant data is saved", type=str)
    parser.add_argument("--pain_path", default="/Users/mayatnf/HEAL/cleaned_data/pain_data.xlsx", help="path for where excel for where additional pain dataset is saved", type=str)
    parser.add_argument("--cleaned", default="/Users/mayatnf/HEAL/cleaned_data/cleaned_HEAL_data.xlsx", help="path for where excel for clean data", type=str)
    parser.add_argument("--outcome_combined", default="/Users/mayatnf/HEAL/cleaned_data/outcome_combined_data.xlsx", help="path for where excel for where additional pain dataset is added", type=str)
    parser.add_argument("--science_preds", default="/Users/mayatnf/HEAL/results/predictions_NLP/science_preds_NLP.xlsx", help="path for where excel for where science type predictions are saved", type=str)
    parser.add_argument("--outcome_preds", default="/Users/mayatnf/HEAL/results/predictions_NLP/outcome_preds_NLP.xlsx", help="path for where excel for where outcome type predictions are saved", type=str)
    parser.add_argument("--milestone_preds", default="/Users/mayatnf/HEAL/results/predictions_NLP/milestone_preds_NLP.xlsx", help="path for where excel for where milestone type predictions are saved", type=str)
    
    args = parser.parse_args()
    return args

#Create clinical and non-clinical dfs based on data
def clin_nonclin(df):
    args = get_args()
    
    #Top Tier; Clinical vs. Non-Clinical
    clin_df = pd.read_csv(args.clintrials_path).rename(columns={'appl_id': "Appl ID"})
    clin_df = pd.merge(df, clin_df, on = ['Appl ID']).drop_duplicates('Appl ID')
    non_clin_df = pd.merge(df, clin_df, how = 'outer', on = ['Appl ID'], indicator = True)
    non_clin_df = non_clin_df[non_clin_df._merge == 'left_only'].drop('_merge', axis=1)    
    clin_df['Science_1'] = [['CLINICAL']]*len(clin_df)
    non_clin_df['Science_1'] = np.empty((len(non_clin_df), 0)).tolist()
    
    clin_df.Science = clin_df.Science.apply(ast.literal_eval)
    non_clin_df.Science = non_clin_df.Science_x.apply(ast.literal_eval)

    #adds clinical label to those lists missing it
    for sci in clin_df.Science:
        if 'CLINICAL' not in sci:
            sci.append('CLINICAL')
   
    return clin_df[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'Science', 'Science_1']], non_clin_df[['Appl ID', 'Combined Cleaned_x', 'Combined Filtered_x', 'Science_x', 'Science_1']]

# Below is a rough skeleton of what the bagging using NLP should look like. 
# Here's some key-word, rule-based code I wrote. Again, it's not perfect, and definitely needs work. It is merely a rough base/draft.
# The key-word regular expressions for science type still need to be defined using the excel files in the folder "Science_Type". This should be done in the codes.py file. 

#Categorizes a study's science types; clinical tree
def clinicals(df, col):
    args = get_args()
    
    for i, row in df.iterrows():
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])
        #checks if label should be Epidemiological
        if re.search(epi, text):
            df['Science_1'][i].append('EPIDEMIOLOGICAL')
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Implementation
                if re.search(implementation, text):
                    df['Science_1'][i].append('IMPLEMENTATION RESEARCH')
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
            else:
                #checks if label should be Implementation
                if re.search(implementation, text):
                    df['Science_1'][i].append('IMPLEMENTATION RESEARCH')
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
        
        #Same for NON-Epidemiological
        else:
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Implementation
                if re.search(implementation, text):
                    df['Science_1'][i].append('IMPLEMENTATION RESEARCH')
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
            else:
                #checks if label should be Implementation
                if re.search(implementation, text):
                    df['Science_1'][i].append('IMPLEMENTATION RESEARCH')
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Health Services
                    if re.search(health_services, text):
                        df['Science_1'][i].append('HEALTH SERVICES RESEARCH')
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                    else:
                        #checks if label should be Core Services
                        if re.search(core_services, text):
                            df['Science_1'][i].append('CORE SERVICES')
                            
    #non_stored = non_clin_df.Science_x.apply(lambda x: any(item for item in ['Translational'] if item in x))
    #exp_num_nonclin = len(non_clin_df[non_stored])

    #mismatches = df.loc[(df['match'] == 'no')]
    pdb.set_trace()
    df['match'] = np.where(df['Science'].apply(set) == df['Science_1'].apply(set), 'yes', 'no')
    print(f'Accuracy:', len(df.loc[df['match'] == 'yes']) / len(df) * 100)
    return df

#Labels a study as basic, translational or implementation
def b_t_i(df, col):
 
    df['found'] = np.empty((len(df), 0)).tolist()
    for i, row in df.iterrows():
        matches_tran = []
        matches_imp = []
        matches_basic = []
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])

        if re.search(translational, text): 
            matches_tran.extend(re.findall(translational, text))
        else:
            if re.search(implementation, text): 
                matches_imp.extend(re.findall(implementation, text))
            else: 
                if re.search(basic, text): 
                    matches_basic.extend(re.findall(basic, text))
        matches_tran.append('TRANSLATIONAL')
        matches_imp.append('IMPLEMENTATION RESEARCH')
        matches_basic.append('BASIC')
        matches = [matches_tran, matches_imp, matches_basic]
        df['Science_1'][i].append(max(matches, key=len)[-1])
        df['found'][i] = matches
    transl = df.Science_1.apply(lambda x: 'TRANSLATIONAL' in x)
    tran_df = df[transl]
    imps = df.Science_1.apply(lambda x: 'IMPLEMENTATION RESEARCH' in x)
    imp_df = df[imps]
    basic_r = df.Science_1.apply(lambda x: 'BASIC' in x)
    basic_df = df[basic_r]
    return tran_df, imp_df, basic_df

#Categorizes a study's science types; basic tree
def basics(df, col):
    args = get_args()
    trues = 0
    for sci in df['Science_x']:
        if 'Basic' in sci:
            trues+=1
    print('Initial Accuracy: ', trues/len(df)*100)

    for i, row in df.iterrows():
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])
        
        #checks if label should be Disease-related Basic
        if re.search(disease, text):
            df['Science_1'][i].append('DISEASE-RELATED BASIC')
            #checks if label should be Translational
            if re.search(translational, text):
                df['Science_1'][i].append('TRANSLATIONAL')
                #checks if label should be systematic
                if re.search(systematic, text):
                    df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
                else: 
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')      
            else:
                if re.search(systematic, text):
                    df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
        else:
            df['Science_1'][i].append('DISEASE-RELATED BASIC')
            #checks if label should be Translational
            if re.search(translational, text):
                df['Science_1'][i].append('TRANSLATIONAL')
                #checks if label should be systematic
                if re.search(systematic, text):
                    df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
                else: 
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')      
            else:
                if re.search(systematic, text):
                    df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
                else:
                    #checks if label should be Core Services
                    if re.search(core_services, text):
                        df['Science_1'][i].append('CORE SERVICES')
    return df 

#Categorizes a study's science types; translational tree
def translation(df, col):
    args = get_args()
    trues = 0
    for sci in df['Science_x']:
        if 'TRANSLATIONAL' in sci:
            trues+=1
    print('Initial Accuracy: ', trues/len(df)*100)

    for i, row in df.iterrows():
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])

        #checks if label should be Disease-related Basic
        if re.search(disease, text):
            df['Science_1'][i].append('DISEASE-RELATED BASIC')
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
            else:
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
        else:
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
            else:
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
    return df 

#Categorizes a study's science types; implementation tree
def implementations(df, col):
    args = get_args()
    trues = 0
    for sci in df['Science_x']:
        if 'Implementation Research' in sci:
            trues+=1
    print('Initial Accuracy: ', trues/len(df)*100)

    for i, row in df.iterrows():
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])
        
        #checks if label should be Epidemiological
        if re.search(epi, text):
            df['Science_1'][i].append('EPIDEMIOLOGICAL')
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
            else:
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
        else:
            #checks if label should be Systematic
            if re.search(systematic, text):
                df['Science_1'][i].append('SYSTEMATIC STUDY OR META-ANALYSIS')
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
            else:
                #checks if label should be Core Services
                if re.search(core_services, text):
                    df['Science_1'][i].append('CORE SERVICES')
    return df 

#Checks if a study is a milestone project
def milestone(df, col):
    #Instantiates list to default "No"
    df['Milestones_NLP'] = 'No'
    for i, row in df.iterrows():
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])
        #checks if label should be milestone
        if re.search(milestones, text):
            df['Milestones_NLP'][i] = 'Yes'
    df['match'] = np.where(df['Milestones'] == df['Milestones_NLP'], 'yes', 'no')
    print('Accuracy:', round(len(df.loc[df['match'] == 'yes']) / len(df) * 100, 2))
    return df[['Appl ID', 'Combined Cleaned', 'Activity Code', 'Milestones', 'Milestones_NLP']]

#Labels a study's primary outcome
def primary_outcome_improved(df, col):
    df['Outcome_NLP'] = ''
    df['found'] = np.empty((len(df), 0)).tolist()
    for i, row in df.iterrows():
        matches_pain = []
        matches_oud = []
        matches_both = []
        if type(row[col]) == list:
            text = " ".join(row[col])
        else:
            text = str(row[col])
        if re.search(oud, text): 
            matches_oud.extend(re.findall(oud, text))
            if re.search(pain, text): 
                matches_both.extend(re.findall(pain, text))
                matches_both.extend(matches_oud)
        if re.search(pain, text): 
            matches_pain.extend(re.findall(pain, text))
        matches_pain.append('Pain')
        matches_oud.append('OUD')
        matches_both.append('Both')
        matches = [matches_pain, matches_oud, matches_both]
        df['Outcome_NLP'][i] = max(matches, key=len)[-1]
        df['found'][i] = matches
    df['match'] = np.where(df['HEAL Category- Primary Outcome'] == df['Outcome_NLP'], 'yes', 'no')
    print('Accuracy:', round(len(df.loc[df['match'] == 'yes']) / len(df) * 100, 2))
    return df[['Appl ID', 'Combined Filtered', 'found', 'HEAL Category- Primary Outcome', 'Outcome_NLP']] 

def main():

    args = get_args()
    df_cleaned = pd.read_excel(args.cleaned)
    df_combined = pd.read_excel(args.outcome_combined)

    #Label Y/N Milestone
    df_cleaned_m = df_cleaned[['Appl ID', 'Combined Cleaned', 'Milestones', 'Activity Code']]
    milestone(df_cleaned_m, 'Combined Cleaned').to_excel(args.milestone_preds)
    
    #Classify science type
    clin_df, non_clin_df = clin_nonclin(df_cleaned)
    translational_df, implementation_df, basic_df = b_t_i(non_clin_df, 'Combined Cleaned_x')

    #clinicals(clin_df, 'Combined Cleaned')
    #translation(translational_df, 'Combined Cleaned_x')
    #basics(basic_df, 'Combined Cleaned_x')
    #implementations(implementation_df, 'Combined Cleaned_x')
    
    #df_cleaned_s = df_cleaned[['Appl ID', 'Combined Cleaned', 'Combined Filtered', 'Science']]
    #categorize_science(df_cleaned_s, 'Combined Cleaned').to_excel(args.science_path)

    #Categorize Outcomes
    df_combined_o = df_combined[['Appl ID', 'Combined Filtered', 'HEAL Category- Primary Outcome']]
    primary_outcome_improved(df_combined_o, 'Combined Filtered').to_excel(args.outcome_preds)
    
if __name__ == "__main__":
    main()