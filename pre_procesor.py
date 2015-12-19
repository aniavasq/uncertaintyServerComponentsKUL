from pandas import read_csv as pd_read_csv
import pandas as pd
import numpy as np
from json import load as json_load

ha_df = pd_read_csv('./data/students_courses.csv')

def data_structure_from_file(filepath):
    """ 
    Get a JSON file as a python data structure.
    
    The structures soported are list and dict.
    
    Params:
        filepath: The path and name from the file
        
    Returns:
        A dict, list, or nested structure.
    """
    _var = None
    with open(filepath) as infile:
        _var = json_load( infile )
    return _var

FACTORS = data_structure_from_file('./data/_factors.json')

_f_d = [ 'factor%d'%i for i in xrange(1, 7) ]

ha_gb = ha_df.groupby('student')

def factors_record(chunk):
	tmp = {'student': chunk['student'].values[0]}
	for factor in _f_d:
		_f = chunk[ chunk['course'].isin( FACTORS[factor] ) ]
		_m_gpa = _f['grade'].values.mean()
		N = len( _f )
		_m_rate = len( _f[ _f['status']=='Passed' ] ) / N if N!=0 else 0.
		tmp['%s_measure'%factor] = _m_rate*_m_gpa
	return tmp

l_val = ha_gb.apply( factors_record )

d_val = pd.DataFrame.from_records(l_val.values)

d_val.to_csv('./data/students_factors.csv')

def get_gpa(chunk):
	chunk['GPA'] = chunk['grade'].values.mean()
	chunk['grade_GPA'] = chunk['grade'] - chunk['GPA']
	return chunk

ht_df = ha_gb.apply( get_gpa )
###################################################################################################
def alpha_calc(chunk):
    alpha = ( chunk['GPA'].values**2 ).sum() / ( chunk['grade'].values * chunk['GPA'].values ).sum()
    return alpha
    
def beta_calc(chunk):
    beta = ( chunk['grade_GPA'].values ).sum() / len( chunk )
    return beta

def skewness_calc(chunk):
	skewness = chunk['grade_GPA'].skew()
	return skewness
    
def count_calc(chunk):
    _count = len( chunk )
    return _count
    
def course_features_record(academic_history):
    cod_materia_acad = academic_history['course'].values[0]
    try:
        cod_materia_acad = cod_materia_acad[:cod_materia_acad.index(' ')]
    except: 
        cod_materia_acad = cod_materia_acad
    tmp = {'course': cod_materia_acad,
           'alpha': alpha_calc( academic_history ),
           'beta': beta_calc( academic_history ),
           'skewness': skewness_calc( academic_history ),
           'count': count_calc( academic_history ),
           }
    return tmp
ht_gb = ht_df.groupby('course')
abs_df = ht_gb.apply( course_features_record )
abs_df = pd.DataFrame.from_records(abs_df.values)
abs_df.to_csv('./data/courses_features.csv')