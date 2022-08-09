import argparse
from pickle import TRUE
import pandas as pd
import numpy as np
import pdb
import ast
from text_utils import find_words
import argparse

def main():

    args = get_args()   
    df_cleaned = pd.read_excel(args.cleaned)

    find_words(df_cleaned, 'HEAL Category- Primary Outcome', ['Pain', 'OUD', 'Both'], 'Combined Cleaned')
    #find_words(df_cleaned, '', [], 'Combined Cleaned')

if __name__ == "__main__":
    main()