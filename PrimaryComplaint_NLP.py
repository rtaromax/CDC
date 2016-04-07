# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:57:21 2016

@Medical PrimaryComplaint NLP

@author: rtaromax
"""

import operator
import nltk
import pandas as pd
import tqdm as tq
import numpy as np
from textblob import TextBlob
from collections import Counter

def read_hospitalization(path_to_csv):      
    ph = pd.read_csv(path_to_csv, sep = '|', encoding="ISO-8859-1")
    #ph = pd.read_csv(path_to_csv, sep = '|')
    ph.rename(columns=lambda x: x.strip(),inplace=True)      
    ph = ph[:-1]
    
    return ph

#if word is title then lowercase it, otherwise keep unchanged
func = lambda s: s[:1].lower() + s[1:] if s.istitle() else s


ph = read_hospitalization("~/Documents/cdc/views3_cleaned/ExtUse_dwvw_PatientHospitalization.csv")    
ph_hosdiag = ph[['PrimaryComplaint','HospitalAdmissionDiagnosis','OtherHospitalAdmissionDiagnosis']]

#replace 'Other' in HospialAdmisionDiagnosis
ph_list = []
for dia, odia in tq.tqdm(zip(ph_hosdiag['HospitalAdmissionDiagnosis'], ph_hosdiag['OtherHospitalAdmissionDiagnosis']), total = len(ph_hosdiag['OtherHospitalAdmissionDiagnosis'])):
    if (dia == 'Other'):
        ph_list.append(odia)
    else:
        ph_list.append(np.nan)
        
ph_hosdiag['HospitalDiagnosis'] = ph_list

ph_nonna_list = ph_hosdiag['HospitalDiagnosis'].dropna().tolist()
zen = TextBlob(' '.join(ph_nonna_list).replace('/',' ').replace('-',' '))
list_words = list(zen.words)

untitled_words = []
for word in list_words:
    word_lemma = word.lemmatize()
    word_lemma = word.lemmatize('v')
    untitled_words.append(func(word_lemma))
    
    
dict_words = dict(Counter(untitled_words))
sorted_list = pd.Series(dict_words, name='Counter')

tagged=nltk.pos_tag(untitled_words)

