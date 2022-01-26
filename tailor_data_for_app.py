import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objects as go
import warnings
import sys
import os
import re
import numpy as np
from scipy import stats
from urllib.request import urlopen
import urllib

import base64
import io
import json
import ast
import time

import random
import glob

np.random.seed(1)
random.seed(1)

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

#########################################################################################
################################# LOAD DATA #############################################
#########################################################################################

mydir = (os.getcwd()).replace('\\','/')+'/'
sys.path.append(mydir)

counties_df = pd.read_pickle(mydir + 'data/counties_df.pkl')
main_df = pd.read_pickle(mydir + 'data/main_df.pkl')

#########################################################################################
######################## Define static variables ########################################
#########################################################################################

main_df = main_df.rename(columns={'date': 'date',
                'Confirmed_US':  'Confirmed cases (US)',
                'Deaths_US': 'Deaths (US)',
                'total_vaccinations_US': 'Total vaccinations (US)',
                'total_distributed_US': 'Total vaccines distributed (US)' ,
                'people_vaccinated_US':  'People vaccinated (US)',
                'people_fully_vaccinated_per_hundred_US':  'People fully vaccinated per hundred (US)',
                'total_vaccinations_per_hundred_US':  'Total vaccinations per hundred (US)',
                'people_fully_vaccinated_US': 'People fully vaccinated (US)',
                'people_vaccinated_per_hundred_US': 'People vaccinated per hundred (US)',
                'distributed_per_hundred_US': 'Distributed per hundred (US)',
                'daily_vaccinations_US': 'Daily vaccinations (US)',
                'daily_vaccinations_per_million_US':  'Daily vaccinations per million (US)',
                'share_doses_used_US':  'Share doses used (US)',
                'total_boosters_US':  'Total boosters (US)',
                'total_boosters_per_hundred_US': 'Total boosters per hundred (US)',
                'daily_vaccinations_I_US': 'Daily vaccinations, Interpolated (US)',
                'daily_distributed_US': 'Daily vaccines distributed (US)',
                'daily_distributed_I_US': 'Daily distributed, Interpolated (US)',
                'daily_people_vaccinated_US': 'Daily people vaccinated (US)',
                'daily_people_vaccinated_I_US': 'Daily people vaccinated, Interpolated (US)',
                'daily_people_fully_vaccinated_US': 'Daily people fully vaccinated (US)',
                'daily_people_fully_vaccinated_I_US': 'Daily people fully vaccinated, Interpolated (US)',
                'daily_total_boosters_US': 'Daily total boosters (US)',
                'daily_total_boosters_I_US': 'Daily total boosters, Interpolated (US)',
                'Total Vax / Total Distributed_US':  'Total Vaccinated / Total Distributed (US)',
                'Total Fully Vax / Total Cases_US': 'Total Fully Vaccinated / Total Cases (US)',
                'Total Boosters / Total Cases_US': 'Total Boosters / Total Cases (US)',
                #'Province/State':  ,
                #'Deaths': 'Deaths',
                #'Recovered': 'Recovered',
                'Confirmed': 'Confirmed cases',
                'total_vaccinations': 'Total vaccinations',
                'total_distributed': 'Total vaccines distributed',
                'people_vaccinated': 'People vaccinated',
                'people_fully_vaccinated_per_hundred': 'People fully vaccinated per hundred',
                'total_vaccinations_per_hundred': 'Total vaccinations per hundred',
                'people_fully_vaccinated': 'People fully vaccinated (1)',
                'people_vaccinated_per_hundred': 'People vaccinated per hundred',
                'distributed_per_hundred': 'Distributed per hundred',
                'daily_vaccinations_raw': 'Daily vaccinations raw',
                'daily_vaccinations': 'Daily vaccinations',
                'daily_vaccinations_per_million': 'Daily vaccinations per million',
                'share_doses_used': 'share doses used',
                'total_boosters': 'Total boosters',
                'total_boosters_per_hundred': 'Total boosters per hundred',
                'daily_vaccinations_I': 'Daily vaccinations, Interpolated',
                'daily_distributed': 'Daily vaccines distributed',
                'daily_distributed_I': 'Daily distributed, Interpolated',
                'daily_people_vaccinated': 'Daily people vaccinated',
                'daily_people_vaccinated_I': 'Daily people vaccinated, Interpolated',
                'daily_people_fully_vaccinated': 'Daily people fully vaccinated',
                'daily_people_fully_vaccinated_I': 'Daily people fully vaccinated, Interpolated',
                'daily_total_boosters': 'Daily total boosters',
                'daily_total_boosters_I': 'Daily total boosters, Interpolated',
                'Total Vax / Total Distributed': 'Total Vaccinated / Total Distributed',
                'Total Fully Vax / Total Cases': 'Total Fully Vaccinated / Total Cases',
                'Total Boosters / Total Cases': 'Total Boosters / Total Cases',
                #'Vaccines admn. (rolling avg)': 'Vaccines admn. (rolling avg)',
                'CLI Admissions (interpolated) WMADD':  'COVID-like illness admissions (interpolated) WMADD',
                #'% Positivity (interpolated) WMADD':  ,
                #'Tests Performed (interpolated) WMADD':  ,
                #'Total Beds':  ,
                #'Total Open Beds':  ,
                #'Total In Use Beds Non-COVID':  ,
                #'Total In Use Beds COVID':  ,
                #'ICU Beds':  ,
                #'ICU Open Beds':  ,
                #'ICU In Use Beds Non-COVID':  ,
                #'ICU In Use Beds COVID':  ,
                #'Ventilator Capacity':  ,
                #'Ventilator Available':  ,
                #'Ventilator In Use Non-COVID':  ,
                #'Ventilator In Use COVID':  ,
                #'Total Beds WMADD':  ,
                #'Total Open Beds WMADD':  ,
                #'Total In Use Beds Non-COVID WMADD':  ,
                #'Total In Use Beds COVID WMADD':  ,
                #'ICU Beds WMADD':  ,
                #'ICU Open Beds WMADD':  ,
                #'ICU In Use Beds Non-COVID WMADD':  ,
                #'ICU In Use Beds COVID WMADD':  ,
                #'Ventilator Capacity WMADD':  ,
                #'Ventilator Available WMADD':  ,
                #'Ventilator In Use Non-COVID WMADD':  ,
                #'Ventilator In Use COVID WMADD':  ,
                'CLI Admissions':  'COVID-like illness admissions',
                'Positive Tests': 'Positive Tests',
                'Tests Performed': 'Tests Performed',
                '% Positivity': '% Positivity',
                'Vaccines administered': 'Vaccines administered',
                'People Fully Vaccinated': 'People fully vaccinated',
                '% of population vaccinated': '% of population vaccinated',
                'CLI Admissions (interpolated)': 'COVID-like illness admissions, Interpolated',
                '% Positivity (interpolated)': '% Positivity, Interpolated',
                'Tests Performed (interpolated)': 'Tests Performed, Interpolated',
                },)


variant_features = ['Omicron (B.1.1.529) Cumulative Count', 'Omicron (B.1.1.529) Count', 'Gamma (P.1) Cumulative Count', 'Gamma (P.1) Count', 'Epsilon (B.1.429) Cumulative Count', 'Epsilon (B.1.429) Count', 'Epsilon (B.1.427/429) Cumulative Count', 'Epsilon (B.1.427/429) Count', 'Epsilon (B.1.427) Cumulative Count', 'Epsilon (B.1.427) Count', 'Delta (B.1.617.2) Cumulative Count', 'Delta (B.1.617.2) Count', 'Delta (AY.3) Cumulative Count', 'Delta (AY.3) Count', 'Delta (AY.2) Cumulative Count', 'Delta (AY.2) Count', 'Delta (AY.1) Cumulative Count', 'Delta (AY.1) Count', 'Beta (B.1.351) Cumulative Count', 'Beta (B.1.351) Count', 'Alpha (B.1.1.7) Cumulative Count', 'Alpha (B.1.1.7) Count', 'Total variant counts, daily', 'Omicron (B.1.1.529) freq', 'Gamma (P.1) freq', 'Epsilon (B.1.429) freq', 'Epsilon (B.1.427/429) freq', 'Epsilon (B.1.427) freq', 'Delta (B.1.617.2) freq', 'Delta (AY.3) freq', 'Delta (AY.2) freq', 'Delta (AY.1) freq', 'Beta (B.1.351) freq', 'Alpha (B.1.1.7) freq', 'Omicron (B.1.1.529) exp freq', 'Gamma (P.1) exp freq', 'Epsilon (B.1.429) exp freq', 'Epsilon (B.1.427/429) exp freq', 'Epsilon (B.1.427) exp freq', 'Delta (B.1.617.2) exp freq', 'Delta (AY.3) exp freq', 'Delta (AY.2) exp freq', 'Delta (AY.1) exp freq', 'Beta (B.1.351) exp freq', 'Alpha (B.1.1.7) exp freq']

main_df.drop(labels=variant_features, axis=1, inplace=True)
#print(list(main_df))
#sys.exit()

features = ['date',
            'Confirmed cases (US)',
            'Deaths (US)',
            'Total vaccinations (US)',
            'Total vaccines distributed (US)' ,
            'People vaccinated (US)',
            'People fully vaccinated per hundred (US)',
            'Total vaccinations per hundred (US)',
            'People fully vaccinated (US)',
            'People vaccinated per hundred (US)',
            'Distributed per hundred (US)',
            'Daily vaccinations (US)',
            'Daily vaccinations per million (US)',
            'Share doses used (US)',
            'Total boosters (US)',
            'Total boosters per hundred (US)',
            'Daily vaccinations, Interpolated (US)',
            'Daily vaccines distributed (US)',
            'Daily distributed, Interpolated (US)',
            'Daily people vaccinated (US)',
            'Daily people vaccinated, Interpolated (US)',
            'Daily people fully vaccinated (US)',
            'Daily people fully vaccinated, Interpolated (US)',
            'Daily total boosters (US)',
            'Daily total boosters, Interpolated (US)',
            'Total Vaccinated / Total Distributed (US)',
            'Total Fully Vaccinated / Total Cases (US)',
            'Total Boosters / Total Cases (US)',
            'Confirmed cases',
            'Deaths',
            'Recovered',
            'Total vaccinations',
            'Total vaccines distributed',
            'People vaccinated',
            'People fully vaccinated per hundred',
            'Total vaccinations per hundred',
            'People vaccinated per hundred',
            'Distributed per hundred',
            'Daily vaccinations raw',
            'Daily vaccinations',
            'Daily vaccinations per million',
            'share doses used',
            'Total boosters',
            'Total boosters per hundred',
            'Daily vaccinations, Interpolated',
            'Daily vaccines distributed',
            'Daily distributed, Interpolated',
            'Daily people vaccinated',
            'Daily people vaccinated, Interpolated',
            'Daily people fully vaccinated',
            'Daily people fully vaccinated, Interpolated',
            'Daily total boosters',
            'Daily total boosters, Interpolated',
            'Total Vaccinated / Total Distributed',
            'Total Fully Vaccinated / Total Cases',
            'Total Boosters / Total Cases',
            'COVID-like illness admissions',
            'Positive Tests',
            'Tests Performed',
            '% Positivity',
            'Vaccines administered',
            '% of population vaccinated',
            'COVID-like illness admissions, Interpolated',
            '% Positivity, Interpolated',
            'Tests Performed, Interpolated',
            'COVID-like illness admissions (interpolated) WMADD',
            '% Positivity (interpolated) WMADD',
            'Tests Performed (interpolated) WMADD',
            'Total Beds',
            'Total Open Beds',
            'Total In Use Beds Non-COVID',
            'Total In Use Beds COVID',
            'ICU Beds',
            'ICU Open Beds',
            'ICU In Use Beds Non-COVID',
            'ICU In Use Beds COVID',
            'Ventilator Capacity',
            'Ventilator Available',
            'Ventilator In Use Non-COVID',
            'Ventilator In Use COVID',
            'Total Beds WMADD',
            'Total Open Beds WMADD',
            'Total In Use Beds Non-COVID WMADD',
            'Total In Use Beds COVID WMADD',
            'ICU Beds WMADD',
            'ICU Open Beds WMADD',
            'ICU In Use Beds Non-COVID WMADD',
            'ICU In Use Beds COVID WMADD',
            'Ventilator Capacity WMADD',
            'Ventilator Available WMADD',
            'Ventilator In Use Non-COVID WMADD',
            'Ventilator In Use COVID WMADD',
            #'People fully vaccinated (1)',
            'People fully vaccinated',
            ]

print(len(features))
main_df.drop(labels=['Vaccines admn. (rolling avg)',
                    'People fully vaccinated (1)',
                    'daily_vaccinations_raw_US',
                    'Province/State'], axis=1, inplace=True)
print(main_df.shape[1])

main_list = np.setdiff1d(features, list(main_df))
print(main_list, '\n')

main_list = np.setdiff1d(list(main_df), features)
print(main_list)

#sys.exit()

operators = ['/', '*', '+', '-']

def fill_cumulative(xs):
    xs = sorted(xs, reverse=True)
    for i, val in enumerate(xs):
        if i == len(xs) - 1:
            break
        
        if val == xs[i+1]:
            xs[i] = (xs[i-1] + xs[i])/2

    xs = sorted(xs)
    return xs

def get_daily_cases(xs):
    xs2 = []
    for i, x in enumerate(xs):
        if i == 0:
            xs2.append(x)
        else:
            xs2.append(xs[i] - xs[i-1])
    
    return xs2
    

main_df['Confirmed cases'].fillna(0, inplace=True)
main_df['Confirmed cases (US)'].fillna(0, inplace=True)

xs = main_df['Confirmed cases'].tolist()
xs = fill_cumulative(xs)
xs = get_daily_cases(xs)
main_df['Daily New Cases (IL)'] = xs

xs = main_df['Confirmed cases (US)'].tolist()
xs = fill_cumulative(xs)
xs = get_daily_cases(xs)
main_df['Daily New Cases (US)'] = xs

features.extend(['Daily New Cases (IL)', 'Daily New Cases (US)'])
x1_features = list(features)
x2_features = features[1:]
y_features = features[1:]

main_ls = list(main_df)
main_set = set([x for x in main_ls if main_ls.count(x) > 1])
print(main_set)
main_ls.sort()
print(main_ls)
print(len(list(main_df)))
print(len(list(set(list(main_df)))))

main_df.to_pickle(mydir + 'data/dat_for_app.pkl', protocol=4)
sys.exit()

