#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:37:16 2020

@author: mason
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import pickle
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import keras

id_formatter = lambda x: x.split('US')[1]

df = pd.read_json('https://api.census.gov/data/2010/acs/acs5?get=B19013_001E,B19301_001E,B15002_019E,B15002_032E,B15002_033E,B15002_034E,B15002_035E,B15002_002E,B15002_015E,B15002_016E,B15002_017E,B15002_018E,B08301_001E,B08301_003E,B08301_004E,B08301_010E,B08301_019E,B08301_021E,B02001_001E,B02001_002E,B02001_003E,B01002_001E,B01003_001E,GEO_ID,NAME&for=tract:*&in=county:*&in=state:06')
df.columns = df.iloc[0]
df.drop(0, inplace=True)
df['year'] = 2010
df.GEO_ID = df.GEO_ID.map(id_formatter)


for year in range(2011, 2019):
    df2 = pd.read_json('https://api.census.gov/data/'+str(year)+'/acs/acs5?get=B19013_001E,B19301_001E,B15002_019E,B15002_032E,B15002_033E,B15002_034E,B15002_035E,B15002_002E,B15002_015E,B15002_016E,B15002_017E,B15002_018E,B08301_001E,B08301_003E,B08301_004E,B08301_010E,B08301_019E,B08301_021E,B02001_001E,B02001_002E,B02001_003E,B01002_001E,B01003_001E,GEO_ID,NAME&for=tract:*&in=county:*&in=state:06')
    df2.columns = df2.iloc[0]
    df2.drop(0, inplace=True)
    df2['year'] = year
    df2.GEO_ID = df2.GEO_ID.map(id_formatter)
    df = pd.concat([df, df2], axis=0)
    

df.B19013_001E.isna().sum()
df.dropna(how='any', inplace=True)

df.B19013_001E = df.B19013_001E.astype(int)
df.describe()
df.drop(df[df.B19013_001E < 0].index, inplace=True)

for column in df.columns[:-6]:
    df[column] = df[column].astype(float)

df.GEO_ID = df.GEO_ID.astype(int)
df.dtypes    

df['ed_total'] = sum([df.B15002_032E,
                     df.B15002_033E,
                     df.B15002_034E,
                     df.B15002_035E,
                     df.B15002_015E,
                     df.B15002_016E,
                     df.B15002_017E,
                     df.B15002_018E]) / sum([df.B15002_019E, df.B15002_002E])
df['ed_fem'] = sum([df.B15002_032E,
                     df.B15002_033E,
                     df.B15002_034E,
                     df.B15002_035E]) / df.B15002_019E
df['ed_male'] = sum([df.B15002_015E,
                     df.B15002_016E,
                     df.B15002_017E,
                     df.B15002_018E]) / df.B15002_002E
df['trans_solo'] = df.B08301_003E/df.B08301_001E
df['trans_pool'] = df.B08301_004E/df.B08301_001E
df['trans_public'] = df.B08301_010E/df.B08301_001E
df['trans_walk'] = df.B08301_019E/df.B08301_001E
df['trans_remote'] = df.B08301_021E/df.B08301_001E
df['race_white'] = df.B02001_002E/df.B02001_001E
df['race_black'] = df.B02001_003E/df.B02001_001E
df['med_house_inc'] = df.B19013_001E
df['per_cap_inc'] = df.B19301_001E
df['med_age'] = df.B01002_001E
df['population'] = df.B01003_001E
df.drop(columns=['B15002_019E',
                 'B15002_032E',
                 'B15002_033E',
                 'B15002_034E',
                 'B15002_035E',
                 'B15002_002E',
                 'B15002_015E',
                 'B15002_016E',
                 'B15002_017E',
                 'B15002_018E',
                 'B08301_001E',
                 'B08301_003E',
                 'B08301_004E',
                 'B08301_010E',
                 'B08301_019E',
                 'B08301_021E',
                 'B02001_001E',
                 'B02001_002E',
                 'B02001_003E',
                 'B19013_001E',
                 'B19301_001E',
                 'B01002_001E',
                 'B01003_001E'], inplace=True)

df.dtypes



df_elig = pd.concat([df[(df.year == year) & (df.med_house_inc <= income)] for year, income in \
                     zip(range(2010, 2016), df.groupby('year').med_house_inc.quantile(.1))], axis=0)
        


df_elig['gent'] = 0
for row in df_elig.iterrows():
    
    years = [row[1].year+1, row[1].year+2, row[1].year+3]
    incomes = zip(df.groupby('year').med_house_inc.quantile(.2).loc[years], 
        df[(df.year >= row[1].year+1) & 
       (df.year <= row[1].year+3) & 
       (df.county == row[1].county) & 
       (df.tract == row[1].tract)].med_house_inc)
    
    per_cap_inc = df[(df.year >= row[1].year+1) & 
                     (df.year <= row[1].year+3) & 
                     (df.county == row[1].county) & 
                     (df.tract == row[1].tract)].per_cap_inc.values
    
    try:
        if ((max(per_cap_inc) - row[1].per_cap_inc) / row[1].per_cap_inc) >= .2:
            df_elig.loc[row[0], 'gent'] = 1
            continue
        else:
            for bench, tract in incomes:
                if tract >= bench:
                    df_elig.loc[row[0], 'gent'] = 1
                    break
    except:
        None

fig, ax = plt.subplots(figsize=(20, 10))
sns.set_palette('Greens', 1, 1)
sns.set_style('white')
ax.plot(df_elig.groupby('year').gent.sum()/df_elig.groupby('year').gent.count(), linewidth=10)
ax.set_ylim(0)
ax.set_yticklabels([0, .1, .2, .3, .4, .5],fontsize=30)
ax.set_xticklabels(range(2009, 2016, 1), fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


df_elig.gent.value_counts()
df_elig.year.value_counts()
#df_elig = df_elig[df_elig.ALAND10 < 50000000]
geodata10 = gpd.read_file('/home/mason/Metis/sat-gentrification/data/tl_2010_06_tract10.shp')

geodata10.GEOID10 = geodata10.GEOID10.astype(int)
sum([bool(i in geodata10.GEOID10.unique()) for i in df_elig.GEO_ID.unique()])
len(df_elig.GEO_ID.unique())

geodata10.dtypes
df_map = gpd.GeoDataFrame(pd.merge(df_elig, geodata10[['GEOID10', 'ALAND10', 'geometry']], how='inner', left_on='GEO_ID', right_on='GEOID10'))


base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(150, 75))
df_map.plot(ax=base, column='gent', cmap='rocket', linewidth=0);
geodata10.INTPTLAT10 = geodata10.INTPTLAT10.astype(float)
geodata10.INTPTLON10 = geodata10.INTPTLON10.astype(float)

df = pd.merge(df, geodata10[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']], how='inner', left_on='GEO_ID', right_on='GEOID10')
df_elig = pd.merge(df_elig, geodata10[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']], how='inner', left_on='GEO_ID', right_on='GEOID10')
#df_elig = df_elig[df_elig.ALAND10 < 50000000]

def tract_dist(lat1, long1, lat2, long2):
    return np.sqrt((lat1 - lat2)**2 + (long1- long2)**2)

lat_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLAT10)
lon_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLON10)
id_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().GEOID10)
year_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().year)

lat_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLAT10)
lon_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLON10)
id_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().GEOID10)
year_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().year)

neighbors = ['n1', 'n2', 'n3', 'n4']
for n in neighbors:
    df_elig[n] = 0
    df_elig[n+'_ed_total'] = 0
    df_elig[n+'_trans_solo'] = 0
    df_elig[n+'_trans_pool'] = 0
    df_elig[n+'_trans_public'] = 0
    df_elig[n+'_trans_walk'] = 0
    df_elig[n+'_trans_remote'] = 0
    df_elig[n+'_race_white'] = 0
    df_elig[n+'_race_black'] = 0
    df_elig[n+'_per_cap_inc'] = 0
    df_elig[n+'_med_age'] = 0
    df_elig[n+'_population'] = 0


count = 0
for geoid, lat, lon, year in tqdm(zip(id_array_elig, lat_array_elig, lon_array_elig, year_array_elig)):
    distances = []
    for geoid2, lat2, lon2, year2 in zip(id_array, lat_array, lon_array, year_array):
        if year2 == year:
            distances.append((geoid2, tract_dist(lat, lon, lat2, lon2)))
    distances.sort(key = lambda x: x[1])
    for ix, n in enumerate(neighbors):
        df_elig.loc[count, n] = distances[ix+1][0]
    count+=1

df.isna().sum()
df_elig.isna().sum()
for row in tqdm(df_elig.iterrows()):
    df_elig.loc[row[0], 'n1_ed_total'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].ed_total.values[0]
    df_elig.loc[row[0], 'n1_trans_solo'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].trans_solo.values[0]
    df_elig.loc[row[0], 'n1_trans_pool'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].trans_pool.values[0]
    df_elig.loc[row[0], 'n1_trans_public'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].trans_public.values[0]
    df_elig.loc[row[0], 'n1_trans_walk'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].trans_walk.values[0]
    df_elig.loc[row[0], 'n1_trans_remote'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].trans_remote.values[0]
    df_elig.loc[row[0], 'n1_race_white'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].race_white.values[0]
    df_elig.loc[row[0], 'n1_race_black'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].race_black.values[0]
    df_elig.loc[row[0], 'n1_per_cap_inc'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].per_cap_inc.values[0]
    df_elig.loc[row[0], 'n1_med_age'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].med_age.values[0]
    df_elig.loc[row[0], 'n1_population'] = df[(df.GEO_ID == row[1].n1) & (df.year == row[1].year)].population.values[0]
    df_elig.loc[row[0], 'n2_ed_total'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].ed_total.values[0]
    df_elig.loc[row[0], 'n2_trans_solo'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].trans_solo.values[0]
    df_elig.loc[row[0], 'n2_trans_pool'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].trans_pool.values[0]
    df_elig.loc[row[0], 'n2_trans_public'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].trans_public.values[0]
    df_elig.loc[row[0], 'n2_trans_walk'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].trans_walk.values[0]
    df_elig.loc[row[0], 'n2_trans_remote'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].trans_remote.values[0]
    df_elig.loc[row[0], 'n2_race_white'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].race_white.values[0]
    df_elig.loc[row[0], 'n2_race_black'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].race_black.values[0]
    df_elig.loc[row[0], 'n2_per_cap_inc'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].per_cap_inc.values[0]
    df_elig.loc[row[0], 'n2_med_age'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].med_age.values[0]
    df_elig.loc[row[0], 'n2_population'] = df[(df.GEO_ID == row[1].n2) & (df.year == row[1].year)].population.values[0]
    df_elig.loc[row[0], 'n3_ed_total'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].ed_total.values[0]
    df_elig.loc[row[0], 'n3_trans_solo'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].trans_solo.values[0]
    df_elig.loc[row[0], 'n3_trans_pool'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].trans_pool.values[0]
    df_elig.loc[row[0], 'n3_trans_public'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].trans_public.values[0]
    df_elig.loc[row[0], 'n3_trans_walk'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].trans_walk.values[0]
    df_elig.loc[row[0], 'n3_trans_remote'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].trans_remote.values[0]
    df_elig.loc[row[0], 'n3_race_white'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].race_white.values[0]
    df_elig.loc[row[0], 'n3_race_black'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].race_black.values[0]
    df_elig.loc[row[0], 'n3_per_cap_inc'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].per_cap_inc.values[0]
    df_elig.loc[row[0], 'n3_med_age'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].med_age.values[0]
    df_elig.loc[row[0], 'n3_population'] = df[(df.GEO_ID == row[1].n3) & (df.year == row[1].year)].population.values[0]
    df_elig.loc[row[0], 'n4_ed_total'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].ed_total.values[0]
    df_elig.loc[row[0], 'n4_trans_solo'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].trans_solo.values[0]
    df_elig.loc[row[0], 'n4_trans_pool'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].trans_pool.values[0]
    df_elig.loc[row[0], 'n4_trans_public'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].trans_public.values[0]
    df_elig.loc[row[0], 'n4_trans_walk'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].trans_walk.values[0]
    df_elig.loc[row[0], 'n4_trans_remote'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].trans_remote.values[0]
    df_elig.loc[row[0], 'n4_race_white'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].race_white.values[0]
    df_elig.loc[row[0], 'n4_race_black'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].race_black.values[0]
    df_elig.loc[row[0], 'n4_per_cap_inc'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].per_cap_inc.values[0]
    df_elig.loc[row[0], 'n4_med_age'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].med_age.values[0]
    df_elig.loc[row[0], 'n4_population'] = df[(df.GEO_ID == row[1].n4) & (df.year == row[1].year)].population.values[0]
        


with open('/home/mason/Metis/sat-gentrification/df_elig_cal.pickle', 'wb') as to_write:
    pickle.dump(df_elig, to_write)


df_model = copy.deepcopy(df_elig[df_elig.ALAND10 < 50000000])
gent = df_model.gent
df_model.drop(columns=['gent', 'GEO_ID', 'NAME', 'state', 'county', 'tract', 'GEOID10', 'INTPTLAT10', 'INTPTLON10', 'n1', 'n2', 'n3', 'n4'], inplace=True)
gent.value_counts()
# Base Models

X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8)

X_smoted, y_smoted = SMOTE(random_state=42).fit_sample(X_train, y_train)
Counter(y_smoted)

reg = LogisticRegression()
fit = reg.fit(X_smoted, y_smoted)
predict = reg.predict(X_test)
log_confusion = confusion_matrix(y_test, predict)
log_confusion
print('Logistic Precision:', log_confusion[0][0]/(log_confusion[0][0]+log_confusion[0][1]))
print('Logistic Recall:', log_confusion[0][0]/(log_confusion[1][0]+log_confusion[0][0]))
print('Logistic Accuracy:', (log_confusion[0][0]+log_confusion[1][1])/len(predict))

reg.coef_
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_smoted, y_smoted)
rf_predict = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test, rf_predict)
rf_confusion
print('Random Forest Precision:', rf_confusion[1][1]/(rf_confusion[1][1]+rf_confusion[1][0]))
print('Random Forest Recall:', rf_confusion[1][1]/(rf_confusion[0][1]+rf_confusion[1][1]))
print('Random Forest Accuracy:', (rf_confusion[1][1]+rf_confusion[0][0])/len(rf_predict))

# Engineering
population_totals = sum([df_model['n1_population'], 
                         df_model['n2_population'],
                         df_model['n3_population'],
                         df_model['n4_population']]) 


df_model['n_ed_total'] = (df_model['n1_ed_total']*df_model['n1_population'] +
                          df_model['n2_ed_total']*df_model['n2_population'] +
                          df_model['n3_ed_total']*df_model['n3_population'] +
                          df_model['n4_ed_total']*df_model['n4_population']) / population_totals

df_model['n_trans_solo'] = (df_model['n1_trans_solo']*df_model['n1_population'] +
                          df_model['n2_trans_solo']*df_model['n2_population'] +
                          df_model['n3_trans_solo']*df_model['n3_population'] +
                          df_model['n4_trans_solo']*df_model['n4_population']) / population_totals

df_model['n_trans_pool'] = (df_model['n1_trans_pool']*df_model['n1_population'] +
                          df_model['n2_trans_pool']*df_model['n2_population'] +
                          df_model['n3_trans_pool']*df_model['n3_population'] +
                          df_model['n4_trans_pool']*df_model['n4_population']) / population_totals

df_model['n_trans_public'] = (df_model['n1_trans_public']*df_model['n1_population'] +
                          df_model['n2_trans_public']*df_model['n2_population'] +
                          df_model['n3_trans_public']*df_model['n3_population'] +
                          df_model['n4_trans_public']*df_model['n4_population']) / population_totals

df_model['n_trans_walk'] = (df_model['n1_trans_walk']*df_model['n1_population'] +
                          df_model['n2_trans_walk']*df_model['n2_population'] +
                          df_model['n3_trans_walk']*df_model['n3_population'] +
                          df_model['n4_trans_walk']*df_model['n4_population']) / population_totals

df_model['n_trans_remote'] = (df_model['n1_trans_remote']*df_model['n1_population'] +
                          df_model['n2_trans_remote']*df_model['n2_population'] +
                          df_model['n3_trans_remote']*df_model['n3_population'] +
                          df_model['n4_trans_remote']*df_model['n4_population']) / population_totals

df_model['n_race_white'] = (df_model['n1_race_white']*df_model['n1_population'] +
                          df_model['n2_race_white']*df_model['n2_population'] +
                          df_model['n3_race_white']*df_model['n3_population'] +
                          df_model['n4_race_white']*df_model['n4_population']) / population_totals

df_model['n_race_black'] = (df_model['n1_race_black']*df_model['n1_population'] +
                          df_model['n2_race_black']*df_model['n2_population'] +
                          df_model['n3_race_black']*df_model['n3_population'] +
                          df_model['n4_race_black']*df_model['n4_population']) / population_totals

df_model['n_per_cap_inc'] = (df_model['n1_per_cap_inc']*df_model['n1_population'] +
                          df_model['n2_per_cap_inc']*df_model['n2_population'] +
                          df_model['n3_per_cap_inc']*df_model['n3_population'] +
                          df_model['n4_per_cap_inc']*df_model['n4_population']) / population_totals

df_model['n_med_age'] = (df_model['n1_med_age']*df_model['n1_population'] +
                          df_model['n2_med_age']*df_model['n2_population'] +
                          df_model['n3_med_age']*df_model['n3_population'] +
                          df_model['n4_med_age']*df_model['n4_population']) / population_totals


for n in neighbors:
    df_model.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                   n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age'], inplace=True)
    



df_model.columns
df_model.drop(columns=['ed_fem', 'ed_male'], inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8)

X_smoted, y_smoted = SMOTE(random_state=42).fit_sample(X_train, y_train)
Counter(y_smoted)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_smoted)
X_test_scaled = std.transform(X_test)

reg = LogisticRegression()
fit = reg.fit(X_train_scaled, y_smoted)
predict = reg.predict(X_test_scaled)
log_confusion = confusion_matrix(y_test, predict)
log_confusion
print('Logistic Precision:', log_confusion[1][1]/(log_confusion[1][1]+log_confusion[1][0]))
print('Logistic Recall:', log_confusion[1][1]/(log_confusion[0][1]+log_confusion[1][1]))
print('Logistic Accuracy:', (log_confusion[1][1]+log_confusion[0][0])/len(predict))

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_smoted, y_smoted)
rf_predict = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test, rf_predict)
rf_confusion
print('Random Forest Precision:', rf_confusion[1][1]/(rf_confusion[1][1]+rf_confusion[1][0]))
print('Random Forest Recall:', rf_confusion[1][1]/(rf_confusion[0][1]+rf_confusion[1][1]))
print('Random Forest Accuracy:', (rf_confusion[1][1]+rf_confusion[0][0])/len(rf_predict))

rf.feature_importances_

df_model = copy.deepcopy(df_elig)
gent = df_model.gent
df_model.drop(columns=['gent', 'GEO_ID', 'NAME', 'state', 'county', 'tract', 'GEOID10', 'INTPTLAT10', 'INTPTLON10', 'n1', 'n2', 'n3', 'n4'], inplace=True)
gent.value_counts()

# ANN

X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8)

X_smoted, y_smoted = SMOTE(random_state=42).fit_sample(X_train, y_train)
Counter(y_smoted)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_smoted)
X_test_scaled = std.transform(X_test)
X_train.shape[1:]

input_layer = keras.layers.Input(shape=X_train.shape[1:])
hidden_layer = keras.layers.Dense(32, activation='tanh')(input_layer)
dropout = keras.layers.Dropout(.2)(hidden_layer)
hidden_layer2 = keras.layers.Dense(16, activation='tanh')(dropout)
output_layer = keras.layers.Dense(2, activation='sigmoid')(hidden_layer2)

model4 = keras.models.Model(inputs=input_layer, outputs=output_layer)
model4.summary()

model4.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model4.fit(X_train_scaled, y_smoted, epochs=30)

loss_and_metrics = model4.evaluate(X_test_scaled, y_test)
print('\nLoss and Accuracy:\n', loss_and_metrics)

# Make Predictions
classes = model4.predict(X_test_scaled)
#proba = model4.predict_proba(X_test, batch_size=32)
print('\nClass Predictions:\n', classes)

net_predict = [x.argmax() for x in classes]
net_confusion = confusion_matrix(y_test, np.array(net_predict))

sum(net_predict)

print('Neural Net Precision:', net_confusion[1][1]/(net_confusion[1][1]+net_confusion[1][0]))
print('Neural Net Recall:', net_confusion[1][1]/(net_confusion[0][1]+net_confusion[1][1]))
print('Neural Net Accuracy:', (net_confusion[1][1]+net_confusion[0][0])/len(net_predict))

with open('/home/mason/Metis/sat-gentrification/nn_prediictions.pickle', 'wb') as to_write:
    pickle.dump(classes, to_write)

with open('/home/mason/Metis/sat-gentrification/X_train_scaled.pickle', 'wb') as to_write:
    pickle.dump(X_train_scaled, to_write)

with open('/home/mason/Metis/sat-gentrification/y_smoted.pickle', 'wb') as to_write:
    pickle.dump(y_smoted, to_write)

with open('/home/mason/Metis/sat-gentrification/X_test_scaled.pickle', 'wb') as to_write:
    pickle.dump(X_test_scaled, to_write)

with open('/home/mason/Metis/sat-gentrification/y_test.pickle', 'wb') as to_write:
    pickle.dump(y_test, to_write)

with open('/home/mason/Metis/sat-gentrification/nn_prediictions.pickle','rb') as read_file:
    net_predict = pickle.load(read_file)

with open('/home/mason/Metis/sat-gentrification/y_test.pickle','rb') as read_file:
    y_test = pickle.load(read_file)

net_predict = [x.argmax() for x in net_predict]

df_map = df_map.loc[y_test.index.values]
df_map['pred'] = net_predict

base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(100, 50))
df_map.plot(ax=base, color='bisque', linewidth=0)

base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(150, 75))
base2 = df_map[df_map.gent == 1].plot(ax=base, color='bisque', linewidth=0)
df_map[(df_map.gent==1) & (df_map.pred==1)].plot(ax=base2, color='midnightblue', linewidth=0) 

import kerastuner as kt
from kerastuner import tuners
from kerastuner import HyperModel as hp

def make_model(base=20, add=0, drop=0, depth=10, batchnorm=False,
               act_reg=0.01, kern_reg=0.01, lr=0.01, act_funct='relu'):

    layers = [keras.layers.Input(shape=X_train_scaled.shape[1:])]

    
    for mult in range(depth, 0, -1):
        layers.append(
            keras.layers.Dense(
                units=mult * base + add, activation=act_funct)
        )
        if batchnorm and (mult % batchnorm == 0):
            layers.append(keras.layers.BatchNormalization())
        if drop > 0:
            if mult == 0:
                pass
            if mult == 1:
                pass
            if mult == 2:
                layers.append(keras.layers.Dropout(.5*drop))
            else:
                layers.append(keras.layers.Dropout(drop))

    layers.append(keras.layers.Dense(2, activation='sigmoid'))

    model = keras.Sequential(layers)

    model.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

test_model = make_model(base=10, add=0, depth=2, drop=0.01, batchnorm=True, act_reg=0, kern_reg=50)
test_model.summary()
test_model.fit(
    X_train_scaled, y_smoted, epochs=200, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[
        keras.callbacks.EarlyStopping(
            patience=15,
            verbose=1,
            restore_best_weights=(True)
        )])


loss_and_metrics = test_model.evaluate(X_test_scaled, y_test)
print('\nLoss and Accuracy:\n', loss_and_metrics)

# Make Predictions
classes = test_model.predict(X_test_scaled)
#proba = model4.predict_proba(X_test, batch_size=32)
print('\nClass Predictions:\n', classes)

net_predict = [x.argmax() for x in classes]
net_confusion = confusion_matrix(y_test, np.array(net_predict))

sum(net_predict)

print('Neural Net Precision:', net_confusion[1][1]/(net_confusion[1][1]+net_confusion[1][0]))
print('Neural Net Recall:', net_confusion[1][1]/(net_confusion[0][1]+net_confusion[1][1]))
print('Neural Net Accuracy:', (net_confusion[1][1]+net_confusion[0][0])/len(net_predict))

from keras.wrappers import scikit_learn as k_sklearn
from sklearn import model_selection

keras_model = k_sklearn.KerasClassifier(make_model)

validator = model_selection.GridSearchCV(
    keras_model, param_grid={
        'base': [10, 20],
        'depth': [1, 2, 3],
        'drop': [0, 0.01, .001],
        'act_reg':[0],
        'kern_reg':[0],
        'batchnorm': [False, True],
        'act_funct': ['relu', 'tanh']
    }, scoring='f1', n_jobs=-1, cv=10, verbose=0)

# Uncomment when you're ready to run. This one will take a while

validator.fit(
    X_train_scaled, y_smoted, epochs=200, verbose=0, callbacks=[
        keras.callbacks.EarlyStopping(
            patience=15,
            verbose=0,
        )])

validator.best_score_
validator.best_params_
best_model = validator.best_estimator_
loss_and_metrics = best_model.evaluate(X_test_scaled, y_test)
print('\nLoss and Accuracy:\n', loss_and_metrics)


# Make Predictions
classes = best_model.predict(X_test_scaled)
#proba = model4.predict_proba(X_test, batch_size=32)
print('\nClass Predictions:\n', classes)

net_predict = [x.argmax() for x in classes]
net_confusion = confusion_matrix(y_test, classes)

sum(net_predict)

print('Neural Net Precision:', net_confusion[1][1]/(net_confusion[1][1]+net_confusion[1][0]))
print('Neural Net Recall:', net_confusion[1][1]/(net_confusion[0][1]+net_confusion[1][1]))
print('Neural Net Accuracy:', (net_confusion[1][1]+net_confusion[0][0])/len(net_predict))

from sklearn.metrics import f1_score

f1_score(y_test, classes)