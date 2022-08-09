### ALL PYTHON CODE

- `codes.py` - contains all keyterms/regular expressions used for labeling. Contains some terms pulled from RCDC--will need to import those files into a respective RCDC folder or change the location in argparse before running. 
- `preprocessing.py` - code to pre-process text; remove stop words, sentence filter, etc. current dataset files are saved in `cleaned_data` folder.
- `text_utils.py` - contains various functions used in pre-processing, data visualizations, sentence wrapper, etc. Look at comments above each function for more details. 
- `key_term_search.py` - NLP, rule-based approaches to search for and count key terms to label. Label predictions saved to `predictions_NLP` under `results/'  
- `supervised_models.py` - supervised machine learning approaches to automate labeling. Includes various pipelines ex. Random Forest, KNN, SVM, Logistic Regression. Label predictions saved to `predictions_ML` under `results/'. Remember when using on new datasets to train using entire HEAL dataset (all 956 studies)--right now, for testing purposes, it is only being trained on 75% of the data.  
- `combine.py` - combined NLP and ML results to find where those models agree vs. disagree. Trust the labels where they both agree--double-check the mismatches. Final label predictions saved to `preds_final` under `results/'
- `unsupervised.py` - unsupervised machine learning approaches to uncover potential trends/patterns in the textual data.
