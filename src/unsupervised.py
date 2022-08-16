import pdb
import pandas as pd
from text_utils import find_words
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", default="/Users/mayatnf/HEAL/cleaned_data/cleaned_HEAL_data.xlsx", help="path for where excel for clean data", type=str)
    parser.add_argument("--milestone_tfidfs", default="/Users/mayatnf/HEAL/term_sets/Milestones/", help="path for where excel file where words with highest tfidf scores are saved", type=str)
    parser.add_argument("--science_tfidfs", default="/Users/mayatnf/HEAL/term_sets/Science_Types/TFIDFs/", help="path for where excel file where words with highest tfidf scores are saved", type=str)
    parser.add_argument("--outcome_tfidfs", default="/Users/mayatnf/HEAL/term_sets/Outcomes/", help="path for where excel file where words with highest tfidf scores are saved", type=str)

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    #DON'T RUN THIS FILE BEFORE CLEANING  
    df_cleaned = pd.read_excel(args.cleaned)

    #For primary outcome
    find_words(df_cleaned, 'HEAL Category- Primary Outcome', ['Pain', 'OUD', 'Both'], 'Combined Cleaned', args.outcome_tfidfs)
    
    #For science types
    for science in ['Science- Basic', 'Science- Translational', 'Science- Clinical', 'Science- Core Services', 'Science- Systematic Meta-analyses', 'EPIDEMIOLOGICAL', 'DISEASE-RELATED BASIC', 'HEALTH SERVICES RESEARCH', 'IMPLEMENTATION RESEARCH']:
        find_words(df_cleaned, science, [1], 'Combined Cleaned', args.science_tfidfs)
    
    #For milestones
    find_words(df_cleaned, 'Milestones', ['Yes'], 'Combined Cleaned', args.milestone_tfidfs)

if __name__ == "__main__":
    main()