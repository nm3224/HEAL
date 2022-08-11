import pandas as pd
from matplotlib import pyplot as plt
from  collections import Counter
import numpy as np
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import re 
import pdb
from codes import aims
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#Function to clean and filter data
def clean_data(df, cols):
    for col in cols:
        cleaned = []
        cleaned_sens = []
        #remove stop words, and lower 
        for text in df[col]: 
            stop = stopwords.words('english')
            stop += ['background', 'description']
            tokenized = word_tokenize(str(text).lower())
            tokenizer = RegexpTokenizer(r'\w+')
            tokenized_clean = tokenizer.tokenize(str(text).lower())
            t = " ".join([word for word in tokenized if word not in stop if not pd.isnull(word) if word != 'nan'])           
            cleaned_t = " ".join([word for word in tokenized_clean if word not in stop if not pd.isnull(word) if word != 'nan'])           
            new = sent_tokenize(t)
            cleaned_sens.append(new)
            cleaned.append(cleaned_t)
        df[f'Cleaned {col}'] = cleaned
        #look for aim, goal, objective, purpose, etc.
        for item in cleaned_sens:
            filter_sen(item, aims)
            item = " ".join(item)
        df[f'Filtered {col}'] = cleaned_sens
    return df

#Function to split science type column labels
def splitter(item):
    return str(item).upper().split(';')

#Smaller sentence wrapper algorithm
def filter_sen(ls, words):
    new = ls[1:-1]
    for sen in new:
        b = False 
        if re.search(words, sen):
            b = True
        if b == False:
            ls.remove(sen)
    return ls

#Find words with highest frequencies/counts using countvec for science types to help build ontologies
def find_words(df, label, labels, text):
    for lab in labels:
        corpus = []
        l_df = df.loc[df[label] == lab]
        for t in l_df[text]:
            corpus.append(str(t))
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([' '.join(corpus)])
            my_dict= dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
            dict_df = pd.DataFrame({'words': list(my_dict.keys()), 'counts': list(my_dict.values())}).sort_values(by = ['counts'], ascending = True)
        print(lab, dict_df.head(15))

#Change Science Type Lists into 1-0 columns
def change_columns(df, sci_types):
    for scitype in sci_types:
        df[scitype] = df[scitype].replace({'No': 0, 'Yes': 1})
    return df 

#Create columns of 1-0 
def create_columns(df, sci_types):
    df['Science'] = df['Science'].map(splitter)
    for scitype in sci_types:
        df[scitype] = 0
    for i, row in df.iterrows():
        for scitype in sci_types:
            if scitype in row['Science']:
                df[scitype][i] = 1
    return df 

#Visualize distribution of training and testing data
def visuals(vals, path_output, name):
    # summarize distribution
    counter = Counter(vals.values)
    # plot the distribution
    plt.bar(counter.keys(), counter.values(), color = ['blue', 'gray'])
    plt.savefig(f"{path_output}{name}.png")
    plt.clf()

def data_dist(df, columns, name):
    dict = {}
    index = []
    yes = []
    no = []
    for col in columns:
        yes.append(len(df[df[col] == 1]))
        no.append(len(df[df[col] == 0]))
        index.append(col)
    dict["Yes"] = yes
    dict["No"] = no
    df_bar = pd.DataFrame(dict, index)
    df_bar.plot(kind = 'bar', color=['blue', 'gray'])
    plt.xticks(rotation='vertical')
    plt.xlabel('Science Types')
    plt.ylabel('Counts')
    plt.title('Data Distribution')
    fig = plt.gcf()
    fig.savefig(f'{name}/data_dist.png')
    plt.clf()

def tuning(x, y):
    #List Hyperparameters that we want to tune.
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]
    
    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    
    #Create new KNN object
    knn_2 = KNeighborsClassifier()
    
    #Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)
    
    #Fit the model
    best_model = clf.fit(x,y)

    best_leaf_size = best_model.best_estimator_.get_params()['leaf_size']
    best_p = best_model.best_estimator_.get_params()['p']
    best_n = best_model.best_estimator_.get_params()['n_neighbors']

    return best_leaf_size, best_p, best_n

#Find mention of pain in RCDC category columb to determine if studies' primary outcomes are only OUD or both
def RCDC_terms(df, col, name):
    pain = []
    for text in df[col]:
        if 'pain' in str(text).lower():
            pain.append('Yes')
        else:
            pain.append('No')
    df['Pain'] = pain
    print(len(df.loc[df['Pain'] == 'Yes'])/len(df)*100)
    print(len(df))
    df.loc[df['Pain'] == 'Yes'].to_excel(name)