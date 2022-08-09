import argparse
from pickle import TRUE
import pandas as pd
import numpy as np
import pdb
import ast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--science_preds_ML", default="/Users/mayatnf/HEAL/science_preds_ML.xlsx", help="path for where excel for where science type predictions are saved", type=str)
    parser.add_argument("--outcome_preds_ML", default="/Users/mayatnf/HEAL/outcome_preds_ML.xlsx", help="path for where excel for where outcome type predictions are saved", type=str)
    parser.add_argument("--milestone_preds_ML", default="/Users/mayatnf/HEAL/milestone_preds_ML.xlsx", help="path for where excel for where milestone type predictions are saved", type=str)
    parser.add_argument("--science_preds_NLP", default="/Users/mayatnf/HEAL/science_preds_NLP.xlsx", help="path for where excel for where science type predictions are saved", type=str)
    parser.add_argument("--outcome_preds_NLP", default="/Users/mayatnf/HEAL/outcome_preds_NLP.xlsx", help="path for where excel for where outcome type predictions are saved", type=str)
    parser.add_argument("--milestone_preds_NLP", default="/Users/mayatnf/HEAL/milestone_preds_NLP.xlsx", help="path for where excel for where milestone type predictions are saved", type=str)
    parser.add_argument("--science_preds_final", default="/Users/mayatnf/HEAL/science_preds_final.xlsx", help="path for where excel for where science type predictions are saved", type=str)
    parser.add_argument("--outcome_preds_final", default="/Users/mayatnf/HEAL/outcome_preds_final.xlsx", help="path for where excel for where outcome type predictions are saved", type=str)
    parser.add_argument("--milestone_preds_final", default="/Users/mayatnf/HEAL/milestone_preds_final.xlsx", help="path for where excel for where milestone type predictions are saved", type=str)
    
    args = parser.parse_args()
    return args

def combine(df_1, df_2, col_1, col_2, col_3):
    df = pd.merge(df_1, df_2, on = 'Appl ID', how = 'inner')
    final_labels = []
    for i, row in df.iterrows():
        if row[col_1] == row[col_2]:
            final_labels.append(row[col_1])
        else:
            final_labels.append('Check Again')
    df['Final Label'] = final_labels
    print(f"Correct: {np.round(len(df.loc[df['Final Label'] != 'Check Again'])/len(df)*100, 2)}")
    return df[['Appl ID', col_1, col_2, f"{col_3}_x", 'Final Label']]

def main():

    args = get_args()
    df_milestones_ML = pd.read_excel(args.milestone_preds_ML)
    df_milestones_NLP = pd.read_excel(args.milestone_preds_NLP)
    df_outcomes_ML = pd.read_excel(args.outcome_preds_ML)
    df_outcomes_NLP = pd.read_excel(args.outcome_preds_NLP)
    milestone_df = combine(df_milestones_ML, df_milestones_NLP, 'Milestones_ML', 'Milestones_NLP', 'Milestones')
    milestone_df.to_excel(args.milestone_preds_final)
    outcome_df = combine(df_outcomes_ML, df_outcomes_NLP, 'HEAL Category- Primary Outcome_ML', 'Outcome_NLP', 'HEAL Category- Primary Outcome')
    outcome_df.to_excel(args.outcome_preds_final)
    milestone_df.loc[milestone_df['Final Label'] == 'Check Again'].to_excel('milestone_mismatches.xlsx')
    outcome_df.loc[outcome_df['Final Label'] == 'Check Again'].to_excel('outcome_mismatches.xlsx')

if __name__ == "__main__":
    main()