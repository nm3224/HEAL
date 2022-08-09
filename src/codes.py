from audioop import add
import pandas as pd
import re 
import pdb #used as a debugger by implementing pdb.set_trace() where needed
import argparse

#Create arguments to access and save new excel files
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/Users/mayatnf/HEAL/original_data/HEAL_GRANTS.xlsx", help="path for where excel for where HEAL grant data is saved", type=str)
    parser.add_argument("--pain_term_path", default="/Users/mayatnf/HEAL/RCDC/_rcdc_includes_Pain.xlsx", help="path for where excel for pain term sets is saved", type=str)
    parser.add_argument("--opioid_term_path", default="/Users/mayatnf/HEAL/RCDC/_rcdc_includes_Opioids.xlsx", help="path for where excel for pain term sets is saved", type=str)
    parser.add_argument("--oud_term_path", default="/Users/mayatnf/HEAL/RCDC/rcdc_includes_Opioid Misuse and Addiction.xlsx", help="path for where excel for oud term sets is saved", type=str)
    parser.add_argument("--new_pain_terms", default="/Users/mayatnf/HEAL/term_sets/pain_terms.xlsx", help="path for where to save excel of new pain term set", type=str)
    parser.add_argument("--new_oud_terms", default="/Users/mayatnf/HEAL/term_sets/oud_terms.xlsx", help="path for where to save excel of new oud term set", type=str)

    args = parser.parse_args()
    return args  

#Loads excel files for RCDC terms and pulls the terms with weights of 100; run for each category path
def load_excel(exc):
    xls = pd.ExcelFile(exc)
    cat_terms = ''
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name = sheet)
        #renaming makes referencing easier
        #checks if this excel is the pain one, which has multiple sheets
        if len(xls.sheet_names) > 2:
            df = df.rename(columns={df.columns[0]: "Words", df.columns[1]: "Weights"})
        else:
            df = df.rename(columns={df.columns[1]: "Words", df.columns[2]: "Weights"})
        term_df = df.loc[df['Weights'] == '100']
        cat_terms += "|".join([word.lower() for word in term_df['Words'] if not pd.isnull(word)])
        # Synonyms
        # if 'Unnamed: 3' in pain_term_df.columns:
        #    cat_terms += "|".join(["|".join(str(word.lower()).splitlines()) for word in df['Unnamed: 3'] if not pd.isnull(word)])
        #cat_terms += '|'
    return cat_terms

#Extracts Activity Codes for Grants which are classified as Milestone Studies to add to regex list
def extract_act_codes(df):
    codes = df.loc[df['Milestones'] == 'Yes'][['Activity Code']]
    code_list = codes['Activity Code'].value_counts().index.to_list()[:15]
    return code_list

#If ever need to add to regex search
def add_to_regex(terms, word):
    if word not in terms:
        terms+=f"|{word}"
    return terms

args = get_args()

#Filtration Regexes for Sentence Wrapping
aims = 'aim|goal|object|purpose|intent|target|idea|hope|motiv|ambition|plan|design|justification|trajectory|mission|hypothesis|theory|thesis|propos'

#Science Regexes
clinical = 'human|random|placebo|trial|clinical|control'
translational = 'translational'
implementation = 'implementation|dissemination'
basic = 'basic'
epi = 'epidemiological'
systematic = 'systematic'
core_services = 'core services' 
disease = 'disease'
health_services = 'health services'

#Milestones Regexes
df_heal = pd.read_excel(args.data_path, sheet_name = 0)
milestones = 'milestone|go/no-go|restriction|benchmark|metric|deliverable|statement of work'
milestones += "|".join([word.lower() for word in extract_act_codes(df_heal)])

#Pain regexes from RCDC
pain = load_excel(args.pain_term_path)
add_pain = ['clbp', 'chronic']
for word in add_pain:
    pain = add_to_regex(pain, word)
    
#OUD regexes from RCDC
oud_2 = load_excel(args.opioid_term_path)
oud = load_excel(args.oud_term_path)
oud = oud + '|' + oud_2

add_oud = ['oud', 'opioid disorder', 'opioid misuse disorder', 'neonatal abstinence syndrome', 'dependency', 'relapse', 'harm reduction', 'opioid withdrawal syndrome', 'neo-natal', 'outcomes babies opioid exposure', 'oboe', 'neonatal', 'child', 'substance use', 'substance abuse', 'substance', 'depression', 'cocaine', 'benzo', 'amphetamine', 'polysubstance', 'poly-substance', 'narcotic']
for word in add_oud:
    oud = add_to_regex(oud, word)

p_term_df = pd.DataFrame({'Pain Terms': pain.split('|')}).to_excel(args.new_pain_terms)
oud_df = pd.DataFrame({'OUD Terms': oud.split('|')}).to_excel(args.new_oud_terms)
