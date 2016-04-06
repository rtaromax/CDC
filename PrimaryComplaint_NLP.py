# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:57:21 2016

@Medical PrimaryComplaint NLP

@author: rtaromax
"""

import pandas as pd
import tqdm as tq
import numpy as np

def read_hospitalization(path_to_csv):      
    ph = pd.read_csv(path_to_csv, sep = '|', encoding="ISO-8859-1")
    #ph = pd.read_csv(path_to_csv, sep = '|')
    ph.rename(columns=lambda x: x.strip(),inplace=True)      
    ph = ph[:-1]
    
    return ph

ph = read_hospitalization()    
ph_hosdiag = ph[['PrimaryComplaint','HospitalAdmissionDiagnosis','OtherHospitalAdmissionDiagnosis']]

