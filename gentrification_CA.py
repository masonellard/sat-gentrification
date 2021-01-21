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
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import keras
from xgboost import XGBClassifier

id_formatter = lambda x: x.split('US')[1]

# Lets download the census data from 2010 and do some preliminary cleaning
df = pd.read_json('https://api.census.gov/data/2010/acs/acs5?get=B19013_001E,B19301_001E,B15002_019E,B15002_032E,B15002_033E,B15002_034E,B15002_035E,B15002_002E,B15002_015E,B15002_016E,B15002_017E,B15002_018E,B08301_001E,B08301_003E,B08301_004E,B08301_010E,B08301_019E,B08301_021E,B02001_001E,B02001_002E,B02001_003E,B01002_001E,B01003_001E,GEO_ID,NAME&for=tract:*&in=county:*&in=state:06')
df.columns = df.iloc[0]
df.drop(0, inplace=True)
df['year'] = 2010
df.GEO_ID = df.GEO_ID.map(id_formatter)

# That looks good. Lets do the same for 2011-2019
for year in range(2011, 2019):
    df2 = pd.read_json('https://api.census.gov/data/'+str(year)+'/acs/acs5?get=B19013_001E,B19301_001E,B15002_019E,B15002_032E,B15002_033E,B15002_034E,B15002_035E,B15002_002E,B15002_015E,B15002_016E,B15002_017E,B15002_018E,B08301_001E,B08301_003E,B08301_004E,B08301_010E,B08301_019E,B08301_021E,B02001_001E,B02001_002E,B02001_003E,B01002_001E,B01003_001E,GEO_ID,NAME&for=tract:*&in=county:*&in=state:06')
    df2.columns = df2.iloc[0]
    df2.drop(0, inplace=True)
    df2['year'] = year
    df2.GEO_ID = df2.GEO_ID.map(id_formatter)
    df = pd.concat([df, df2], axis=0) # combining the 2010 with the 2011-2019 data
    
df.B19013_001E.isna().sum()
df.dropna(how='any', inplace=True)

# Additional formatting for later

df.B19013_001E = df.B19013_001E.astype(int)

df.describe()

df.drop(df[df.B19013_001E < 0].index, inplace=True)

for column in df.columns[:-6]:
    df[column] = df[column].astype(float)

df.GEO_ID = df.GEO_ID.astype(int)

df.dtypes  

# data types look good. Lets change the column names to something more intuitive. 
# Also, We can engineer features such as number of workers that drive to work
# (trans_solo) to reflect percentages of the population / sub-population

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

# Lets drop the original columns
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

# Check to make sure data types were maintained
df.dtypes

# We only want to analyze neighborhoods that are 'eligible' to later be gentrified.
# We define eligibility as being in the bottom 10th percentile for median household
# income in the year that it is being analyzed.

df_elig = pd.concat([df[(df.year == year) & (df.med_house_inc <= income)] for year, income in \
                     zip(range(2010, 2016), df.groupby('year').med_house_inc.quantile(.1))], axis=0)
        
# Now that we have our eligible neighborhoods, lets define our gentrification label.
# If any neighborhood in our eligible dataframe (df_elig) jumps to to 20th percentile 
# for median household income in the 3 consecutive years after, or per capita income 
# increases by 20%, we'll define that neighborhood as gentrified (df_elig['gent] = 1)

df_elig['gent'] = 0
for row in df_elig.iterrows():
    
    # A list of the 3 consecutive years after the year in the current row
    years = [row[1].year+1, row[1].year+2, row[1].year+3]
    
    # Zipping together pairs of benchmarks and tract values. Benchmark represents
    # the 20th percentile threshold for median household for that year. Tract value
    # represents the current neighborhood's median household income for the same year.
    incomes = zip(df.groupby('year').med_house_inc.quantile(.2).loc[years], 
        df[(df.year >= row[1].year+1) & 
       (df.year <= row[1].year+3) & 
       (df.county == row[1].county) & 
       (df.tract == row[1].tract)].med_house_inc)
    
    # array of the current neighborhood's per cap income in all three years
    per_cap_inc = df[(df.year >= row[1].year+1) & 
                     (df.year <= row[1].year+3) & 
                     (df.county == row[1].county) & 
                     (df.tract == row[1].tract)].per_cap_inc.values
    
    try:
        if ((max(per_cap_inc) - row[1].per_cap_inc) / row[1].per_cap_inc) >= .2: # if per cap income increase by 20%
            df_elig.loc[row[0], 'gent'] = 1
            continue
        else:
            for bench, tract in incomes:
                if tract >= bench: # if median household income ever reached at least the 20th percentile 
                    df_elig.loc[row[0], 'gent'] = 1
                    break
    except:
        None

# Now that we've defined our label, lets look at what percentage of neighborhoods were gentrified each year
fig, ax = plt.subplots(figsize=(20, 10))
sns.set_palette('Greens', 1, 1)
sns.set_style('white')
ax.plot(df_elig.groupby('year').gent.sum()/df_elig.groupby('year').gent.count(), linewidth=10)
ax.set_ylim(0)
ax.set_yticklabels([0, .1, .2, .3, .4, .5],fontsize=30)
ax.set_xticklabels(range(2009, 2016, 1), fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Looks like there's a solid upward trend. Lets look at total value counts
df_elig.gent.value_counts()
df_elig.year.value_counts()

# geographic data for each census tract
geodata10 = gpd.read_file('/home/mason/Metis/sat-gentrification/data/tl_2010_06_tract10.shp')

geodata10.GEOID10 = geodata10.GEOID10.astype(int)

# Checking to make sure our GEOID's match
sum([bool(i in geodata10.GEOID10.unique()) for i in df_elig.GEO_ID.unique()])
len(df_elig.GEO_ID.unique())
geodata10.dtypes

# merging data on eligible neighborhoods and geographic data into a dataframe that can be used for plotting maps
df_map = gpd.GeoDataFrame(pd.merge(df_elig, geodata10[['GEOID10', 'ALAND10', 'geometry']], how='inner', left_on='GEO_ID', right_on='GEOID10'))

# plotting each county
base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(150, 75))
geodata10.plot(ax=base, column='COUNTYFP10', cmap='rocket', linewidth=0);

geodata10.INTPTLAT10 = geodata10.INTPTLAT10.astype(float)
geodata10.INTPTLON10 = geodata10.INTPTLON10.astype(float)

# Lets merge our census and location data
df = pd.merge(df, geodata10[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']], how='inner', left_on='GEO_ID', right_on='GEOID10')
df_elig = pd.merge(df_elig, geodata10[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']], how='inner', left_on='GEO_ID', right_on='GEOID10')


def tract_dist(lat1, long1, lat2, long2):
    '''
    

    Parameters
    ----------
    lat1 : float
        Internal latitude for first tract
    long1 : float
        Internal longitude for first tract
    lat2 : float
        Internal latitude for second tract
    long2 : float
        Internal longitude for second tract

    Returns
    -------
    float
        Euclidean distance between both tracts' internal points

    '''
    return np.sqrt((lat1 - lat2)**2 + (long1- long2)**2)

# Next, we're going to look at the closest neighborhoods to our target neighborhood. This can be slow, 
# so to make it faster we'll create numpy arrays, which are faster than pandas DataFrames.

# Lets create arrays for latitude, longitude, GEOID, and year for every tract in our original df
lat_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLAT10)
lon_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLON10)
id_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().GEOID10)
year_array = np.array(df[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().year)

# Now lets create arrays for latitude, longitude, GEOID, and year for every tract in our eligible df
lat_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLAT10)
lon_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().INTPTLON10)
id_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().GEOID10)
year_array_elig = np.array(df_elig[['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']].drop_duplicates().year)

# Set the number of closest neighborhoods we want in our dataframe, and create the necessary columns
num_neighbors = 20
neighbors = ['n'+str(i+1) for i in range(num_neighbors)] # a list of strings that will act as column names for the GEOID of each 'neighbor'
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
    df_elig[n+'_land_area'] = 0


count = 0
for geoid, lat, lon, year in tqdm(zip(id_array_elig, lat_array_elig, lon_array_elig, year_array_elig)):
    
    distances = [] # list for storing neighborhood distances
    
    # For each eligible neighborhood, we'll look at all other neighborhoods in the original dataframe
    # from the same year, and append their geoid's and distances to our distances list
    for geoid2, lat2, lon2, year2 in zip(id_array, lat_array, lon_array, year_array):
        if year2 == year:
            distances.append((geoid2, tract_dist(lat, lon, lat2, lon2)))
    
    # After sorting by closest distance, we put the geoid's of the n closest neighbors in their appropriate columns
    distances.sort(key = lambda x: x[1])
    for ix, n in enumerate(neighbors):
        df_elig.loc[count, n] = distances[ix+1][0] # closest neighbor to our target is itself, so we use ix+1 to skip the first index
    count+=1

# Now that we've filled out the geoid's for each neighbor, we can go ahead and fill out the other features for each neighbor
for row in tqdm(df_elig.iterrows()):
    for neighbor in neighbors:
        df_elig.loc[row[0], neighbor+'_ed_total'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].ed_total.values[0]
        df_elig.loc[row[0], neighbor+'_trans_solo'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].trans_solo.values[0]
        df_elig.loc[row[0], neighbor+'_trans_pool'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].trans_pool.values[0]
        df_elig.loc[row[0], neighbor+'_trans_public'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].trans_public.values[0]
        df_elig.loc[row[0], neighbor+'_trans_walk'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].trans_walk.values[0]
        df_elig.loc[row[0], neighbor+'_trans_remote'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].trans_remote.values[0]
        df_elig.loc[row[0], neighbor+'_race_white'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].race_white.values[0]
        df_elig.loc[row[0], neighbor+'_race_black'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].race_black.values[0]
        df_elig.loc[row[0], neighbor+'_per_cap_inc'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].per_cap_inc.values[0]
        df_elig.loc[row[0], neighbor+'_med_age'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].med_age.values[0]
        df_elig.loc[row[0], neighbor+'_population'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].population.values[0]
        df_elig.loc[row[0], neighbor+'_land_area'] = df[(df.GEO_ID == row[1][neighbor]) & (df.year == row[1].year)].ALAND10.values[0]

with open('/home/mason/Metis/sat-gentrification/df_elig_cal.pickle', 'wb') as to_write:
    pickle.dump(df_elig, to_write)

with open('/home/mason/Metis/sat-gentrification/df_elig_cal.pickle','rb') as read_file:
    df_elig = pickle.load(read_file)

# Lets go ahead a create a dataframe for modeling, and define our label
df_model = copy.deepcopy(df_elig)
gent = df_model.gent

df_model.drop(columns=[n for n in neighbors], inplace=True)
df_model.drop(columns=['gent', 'GEO_ID', 'NAME', 'state', 'county', 'tract', 'GEOID10', 'INTPTLAT10', 'INTPTLON10'], inplace=True)

gent.value_counts()

def neighbor_agg(df, neighbors):
    '''
    

    Parameters
    ----------
    df : Pandas DataFrame
        My df of census data, that also contains certain features for neighboring tracts
    neighbors : List
        A list of the neighboring tracts included in df, denoted by ['n1', 'n2', 'n3', ... ]

    Returns
    -------
    df : Pandas DataFrame
        Original df with the features for all neighboring tracts aggregated into single features.

    '''
        
    population_totals = sum([df[n+'_population'] for n in neighbors]) # sum of the population in all neighboring tracts
    
    # Lets aggregate features for neighboring tracts by taking a weighted average based on population
    df['n_ed_total'] = sum([df[n+'_ed_total']*df[n+'_population'] 
                                  for n in neighbors]) / population_totals
    
    df['n_trans_solo'] = sum([df[n+'_trans_solo']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_trans_pool'] = sum([df[n+'_trans_pool']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_trans_public'] = sum([df[n+'_trans_public']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_trans_walk'] = sum([df[n+'_trans_walk']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_trans_remote'] = sum([df[n+'_trans_remote']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_race_white'] = sum([df[n+'_race_white']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_race_black'] = sum([df[n+'_race_black']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_per_cap_inc'] = sum([df[n+'_per_cap_inc']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_med_age'] = sum([df[n+'_med_age']*df[n+'_population'] 
                                     for n in neighbors]) / population_totals
    
    df['n_population'] = population_totals # created a feature containing population total of all neighboring tracts
    
    df['n_land_area'] = sum([df[n+'_land_area'] for n in neighbors]) # created a feature containing total land area of neighboring tracts
    
    # Now that we've created the aggregated features, we can get rid of the original features in the df
    for n in neighbors:
        df.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                       n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)
        
    return df

# Base naive bayes model

X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8, random_state=42) # 80/20 split

# Lets aggregate the features for the 20 neighboring tracts
X_train = neighbor_agg(X_train, neighbors)
X_test = neighbor_agg(X_test, neighbors)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)

nb = GaussianNB()
fit = nb.fit(X_train_scaled, y_train)
predict = nb.predict(X_test_scaled)
nb_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

reg = LogisticRegression()
fit = reg.fit(X_train_scaled, y_train)
predict = reg.predict(X_test_scaled)
log_confusion = confusion_matrix(y_test, predict)
log_confusion
precision_score(y_test, predict)
recall_score(y_test, predict)

# Using 20 neighboring tracts, we achieved -30% recall and 50% precision with naive bayes
# as well as 19% recall and 58% precision with logistic regression

# Lets find the optimal number of neighboring tracts to include in our model
X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8, random_state=42)


auc = {} # dictionary to hold the average roc_auc scores for each number of neighboring tracts   
for num in tqdm((range(0, num_neighbors+1))):
    x = copy.deepcopy(X_train)
    
    # First, lets drop all neighboring tracts that we don't intend to use for this iteration
    for n in neighbors[num:]:
        x.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)
    
    # If we haven't dropped all neighboring tracts, aggregate the features for any remaining neighboring tracts
    # to reduce dimensionality
    if num != 0:
        x = neighbor_agg(x, neighbors[:num])
        
    std = StandardScaler()
    x_train_scaled = std.fit_transform(x)
    
    # Using Logistic Regression and 5-fold cross-validation to determine avg roc_auc score for num number of 
    # neighboring tracts
    cv_scores = cross_validate(LogisticRegression(), x_train_scaled, y_train, cv=5, scoring=('roc_auc'), n_jobs=-1)
    auc[str(num)] = np.mean(cv_scores['test_score'])

print("Outcomes from the Best Logistic Regression Model:")
max_auc = max(auc.values())
print("Maximum Average AUC:", max_auc.round(5))

# Identifying the optimal number of neighboring tracts based on mean cv roc_auc score 
for key, val in auc.items():
    if val == max_auc:
        print("The optimal number of neighbors:", key)
        optimal_n = int(key)

# Base Models with optimal number of neighboring tracts

for n in neighbors[optimal_n:]:
        X_train.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)
        X_test.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)

if optimal_n != 0:
    X_train = neighbor_agg(X_train, neighbors[:optimal_n])
    X_test = neighbor_agg(X_test, neighbors[:optimal_n])


std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)

nb = GaussianNB()
fit = nb.fit(X_train_scaled, y_train)
predict = nb.predict(X_test_scaled)
nb_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

reg = LogisticRegression()
fit = reg.fit(X_train_scaled, y_train)
predict = reg.predict(X_test_scaled)
log_confusion = confusion_matrix(y_test, predict)
log_confusion
precision_score(y_test, predict)
recall_score(y_test, predict)

# Looks like we got higher precision and recall with both models
# Lets look at random forest

rf = RandomForestClassifier()
fit = rf.fit(X_train, y_train)
predict = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

# Looks like we were able to achieve a much higher precision of ~67% and a recall of ~37%

# Lets see if we can find out where increasing the number of neighbors becomes counterproductive.
# We'll look at what happens when we increase the neighbors to 10
X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8, random_state=42)

for n in neighbors[10:]:
        X_train.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)
        X_test.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)

X_train = neighbor_agg(X_train, neighbors[:10])
X_test = neighbor_agg(X_test, neighbors[:10])

std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)

nb = GaussianNB()
fit = nb.fit(X_train_scaled, y_train)
predict = nb.predict(X_test_scaled)
nb_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

reg = LogisticRegression()
fit = reg.fit(X_train_scaled, y_train)
predict = reg.predict(X_test_scaled)
log_confusion = confusion_matrix(y_test, predict)
log_confusion
precision_score(y_test, predict)
recall_score(y_test, predict)

rf = RandomForestClassifier()
fit = rf.fit(X_train, y_train)
predict = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

# Naive bayes, logistic regression, and random forest seem pretty close to their scores with
# 2 neighbors

# Lets examine a pairplot shaded based on residuals of our random forest model. Maybe we can see 
# exactly what our model has trouble predicting

X_test['residuals'] = predict - y_test
X_test['gent'] = y_test

sns.pairplot(X_test, hue = 'residuals')

# It seems that the model has trouble predicting black majority neighborhoods where education in 
# surrounding neighborhoods is low. Lets see how the model does predicting white majority neighborhoods
# where education in surrounding neighborhoods is greater than .25

X_test.drop(columns=['residuals', 'gent'], inplace=True)

rf_pred_aug = rf.predict(X_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)])
rf_confusion = confusion_matrix(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], rf_pred_aug)

precision_score(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], rf_pred_aug)
recall_score(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], rf_pred_aug)

# It seems recall stayed about the same, but precision increased ~7%! Lets take another look at a pairplot.
# There also appeared to be a relationship between predictability and the land area of neighboring tracts. 
# Lets impose a minimum n_land_area and see what happens.

rf_pred_aug = rf.predict(X_test[X_test.n_land_area > 5e6])
rf_confusion = confusion_matrix(y_test[X_test.n_land_area > 5e6], rf_pred_aug)

precision_score(y_test[X_test.n_land_area > 5e6], rf_pred_aug)
recall_score(y_test[X_test.n_land_area > 5e6], rf_pred_aug)

# Looks like we got an increase in precision there as well! Lets apply both sets of restrictions and see 
# what happens.

rf_pred_aug = rf.predict(X_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))])
rf_confusion = confusion_matrix(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], rf_pred_aug)

precision_score(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], rf_pred_aug)
recall_score(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], rf_pred_aug)

# Precision increased to ~76%! Lets pickle this random forest model.

# with open('/home/mason/Metis/sat-gentrification/rf_gent.pickle', 'wb') as to_write:
#    pickle.dump(rf, to_write)

with open ('/home/mason/Metis/sat-gentrification/rf_gent.pickle', 'rb') as read_file:
    rf = pickle.load(read_file)

# Tuning an XGBoost model with 10 neighboring tracts

X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8, random_state=42)

for n in neighbors[10:]:
        X_train.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)
        X_test.drop(columns=[n+'_ed_total', n+'_trans_solo', n+'_trans_pool', n+'_trans_public', n+'_trans_walk', n+'_trans_remote',
                    n+'_race_white', n+'_race_black', n+'_per_cap_inc', n+'_med_age', n+'_population', n+'_land_area'], inplace=True)

X_train = neighbor_agg(X_train, neighbors[:10])
X_test = neighbor_agg(X_test, neighbors[:10])

y_train.value_counts()

learning_rate = [round(x,2) for x in np.linspace(start = .01, stop = .6, num = 60)]
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_depth = range(3,10,1)
child_weight = range(1,6,2)
gamma = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,2]
subsample = [.6, .7, .8, .9, 1]
col_sample = [.6, .7, .8, .9, 1]

# Tuning the learning_rate:
xgb_tune = XGBClassifier(n_estimators = 100,max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'learning_rate':learning_rate},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_learning_rate = xgb_grid.best_params_['learning_rate']
print("Best learning_rate:", best_learning_rate)

# Tuning the n_estimators:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, max_depth = 3, min_child_weight = 1, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'n_estimators': n_estimators},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_n = xgb_grid.best_params_['n_estimators']
print("Best n_estimators:", best_n)

# Tuning max_depth and min_child_weight:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, subsample = .8, colsample_bytree = 1,gamma = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'max_depth': max_depth, 'min_child_weight': child_weight},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_depth = xgb_grid.best_params_['max_depth']
best_weight = xgb_grid.best_params_['min_child_weight']
print("Best max_depth:", best_depth)
print("Best min_child_weight:", best_weight)

# Tuning gamma:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = .8, colsample_bytree = 1, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'gamma': gamma},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_gamma = xgb_grid.best_params_['gamma']
print("Best gamma:", best_gamma)

# Tuning subsample and colsample_bytree:
xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Rigorously tuning subsample and colsample_bytree:
subsample = [best_subsample-.02, best_subsample - .01, best_subsample, best_subsample +.01, best_subsample + .02]
col_sample = [best_col_sample-.02, best_col_sample - .01, best_col_sample, best_col_sample+.01, best_col_sample+ .02]

xgb_tune = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, gamma = best_gamma, n_jobs = -1)
xgb_grid = GridSearchCV(estimator=xgb_tune, param_grid = {'subsample': subsample, 'colsample_bytree': col_sample},cv=5, scoring='roc_auc', verbose = 0, n_jobs = -1)
xgb_grid.fit(X_train,y_train)
best_subsample = xgb_grid.best_params_['subsample']
best_col_sample = xgb_grid.best_params_['colsample_bytree']
print("Best subsample:", best_subsample)
print("Best colsample_bytree:", best_col_sample)

# Lets evaluate our final xgboost model using 5-fold cv / mean roc_auc
xgb = XGBClassifier(learning_rate = best_learning_rate, n_estimators = best_n, max_depth = best_depth, min_child_weight = best_weight, subsample = best_subsample, colsample_bytree = best_col_sample, gamma = best_gamma, n_jobs = -1)
cv_scores = cross_validate(xgb, X_train,y_train, cv=5, scoring=('roc_auc'), verbose = 0, n_jobs = -1)
a = np.mean(cv_scores['test_score'])
print("Outcomes from the Best XGBoost Classification Model:")
print("Average Test AUC:", a.round(5))

# Lets see how it holds when validated using our 80/20 train test split
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
xgb_confusion = confusion_matrix(y_test, xgb_pred)
xgb_confusion
print('XGB Precision:', precision_score(y_test, xgb_pred))
print('XGB Recall:', recall_score(y_test, xgb_pred))

# ~62% precision and ~32% recall. It appears that our xgboost model may be overfitting more compared
# to our random forest model. Not surprising since we only have ~4400 rows.

# Lets see how xgboost performs under the same restrictions we imposed on the random forest model

xgb_pred_aug = xgb.predict(X_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)])
xgb_confusion = confusion_matrix(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], xgb_pred_aug)

precision_score(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], xgb_pred_aug)
recall_score(y_test[(X_test.race_white > .5) | (X_test.n_ed_total > .25)], xgb_pred_aug)

xgb_pred_aug = xgb.predict(X_test[X_test.n_land_area > 5e6])
xgb_confusion = confusion_matrix(y_test[X_test.n_land_area > 5e6], xgb_pred_aug)

precision_score(y_test[X_test.n_land_area > 5e6], xgb_pred_aug)
recall_score(y_test[X_test.n_land_area > 5e6], xgb_pred_aug)

xgb_pred_aug = xgb.predict(X_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))])
xgb_confusion = confusion_matrix(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], xgb_pred_aug)

precision_score(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], xgb_pred_aug)
recall_score(y_test[(X_test.n_land_area > 5e6) & ((X_test.race_white > .5) | (X_test.n_ed_total > .25))], xgb_pred_aug)

# Looks like we got similar results as far as improvement, but scores are still lower than random forest across the board

with open('/home/mason/Metis/sat-gentrification/xgb_gent.pickle', 'wb') as to_write:
    pickle.dump(xgb, to_write)

# Now lets try to reduce our feature space using Lasso, and see if that improves our model

auc = {} # a dictionary to store mean roc_auc scores for each alpha value
for alpha in tqdm(np.arange(.0005,.05,.002)):
    
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train_scaled, y_train)
    
    # extracting non-zeroed out features from lasso model
    features = []
    for ix, column in enumerate(X_train.columns):
        if lasso.coef_[ix] != 0:
            features.append(column)
            
    std = StandardScaler()
    X_train_scaled_lasso = std.fit_transform(X_train[features])
    
    # Using random forest and 5-fold cross-validation to determine avg roc_auc score for non-zeroed features
    cv_scores = cross_validate(RandomForestClassifier(), X_train_scaled_lasso, y_train, cv=5, scoring=('roc_auc'), n_jobs=-1)
    auc[str(alpha)] = np.mean(cv_scores['test_score']) 

print("Outcomes from the Best Random Forest Model:")
max_auc = max(auc.values())
print("Maximum Average AUC:", max_auc.round(5))

# Identifying the optimal alpha based on mean cv roc_auc score 
for key, val in auc.items():
    if val == max_auc:
        print("The optimal alpha:", key)
        optimal_alpha = float(key)

lasso = Lasso(alpha=optimal_alpha)
lasso.fit(X_train_scaled, y_train)

# extracting non-zeroed features from lasso model with best alpha, and feeding into random forest
features = []
for ix, column in enumerate(X_train.columns):
    if lasso.coef_[ix] != 0:
        features.append(column)

rf = RandomForestClassifier()
rf.fit(X_train[features], y_train)
predict = rf.predict(X_test[features])
rf_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

# reducing our feature space does not seem to improve on our best random forest model, so we'll keep the
# original

# For the sake of being thorough, lets build an artificial neural network

X_train, X_test, y_train, y_test = train_test_split(df_model, gent, train_size=.8, random_state=42)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)

X_train.shape[1:]

# We'll start with a simple 3 layer model: 1 input, 1 hidden, and 1 output

input_layer = keras.layers.Input(shape=X_train.shape[1:])
hidden_layer = keras.layers.Dense(32, activation='relu')(input_layer) 
output_layer = keras.layers.Dense(2, activation='sigmoid')(hidden_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=30)

loss_and_metrics = model.evaluate(X_test_scaled, y_test)
print('\nLoss and Accuracy:\n', loss_and_metrics)


classes = model.predict(X_test_scaled)
#proba = model4.predict_proba(X_test, batch_size=32)

net_predict = [x.argmax() for x in classes] # convert probabilities to binary class predictions with .5 threshold
net_confusion = confusion_matrix(y_test, np.array(net_predict))
print('Neural Net Precision:', precision_score(y_test, net_predict))
print('Neural Net Recall:', recall_score(y_test, net_predict))


def make_model(base=20, add=0, drop=0, depth=10, batchnorm=False, act_funct='relu'):
    '''
    

    Parameters
    ----------
    base : integer, optional
        Multiplied by depth to determine units/nodes in each layer. The default is 20.
    add : integer, optional
        Number of units added to base number of units in each layer. The default is 0.
    drop : float, optional
        Dropout regularization parameter. Number can be 0-1, inclusive. If equal to zero,
        there will be no dropout layer(s). The default is 0.
    depth : integer, optional
        Number of dense layers. The default is 10.
    batchnorm : boolean, optional
        Specifies whether or not to include a BatchNormalization layer (layer that normalizes
        it's inputs) after each Dense layer. The default is False.
    act_funct : string, optional
        Activation function for all dense layers. The default is 'relu'.

    Returns
    -------
    model : object
        A configured TensorFlow.keras Model.

    '''

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

# Lets create a single test model using our make_model function

test_model = make_model(base=10, add=0, depth=2, drop=0.01, batchnorm=True)
test_model.summary()

test_model.fit(
    X_train_scaled, y_train, epochs=200, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[
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

print('Neural Net Precision:', precision_score(y_test, net_predict))
print('Neural Net Recall:', recall_score(y_test, net_predict))

# Looks like our function works. Lets apply the scikit_learn wrapper to keras,
# and use GridSearchCV to tune

from keras.wrappers import scikit_learn as k_sklearn
from sklearn import model_selection

keras_model = k_sklearn.KerasClassifier(make_model) # scikit_learn keras wrapper classification model

# Setting up the GridSearch
validator = model_selection.GridSearchCV(
    keras_model, param_grid={
        'base': [10, 20],
        'depth': [1, 2, 3],
        'drop': [0, 0.01, .001],
        'batchnorm': [False, True],
        'act_funct': ['relu', 'tanh']
    }, scoring='roc_auc', n_jobs=-1, cv=5, verbose=0)

# Run the GridSearch
validator.fit(
    X_train_scaled, y_train, epochs=200, verbose=0, callbacks=[
        keras.callbacks.EarlyStopping(
            patience=15,
            verbose=0,
        )])

validator.best_score_
validator.best_params_
best_model = validator.best_estimator_

# Make Predictions
classes = best_model.predict(X_test_scaled)
#proba = model4.predict_proba(X_test, batch_size=32)
print('\nClass Predictions:\n', classes)

net_confusion = confusion_matrix(y_test, classes)

print('Neural Net Precision:', precision_score(y_test, classes))
print('Neural Net Recall:', recall_score(y_test, classes))

# We got our worst predictions yet, with ~40% precision and recall.
# This is not surprising, since neural nets need a lot of data to train
# and have a tendency to overfit. Since our best model was our random 
# forest, lets visualize our results.

with open ('/home/mason/Metis/sat-gentrification/rf_gent.pickle', 'rb') as read_file:
    rf = pickle.load(read_file)

predict = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test, predict)
precision_score(y_test, predict)
recall_score(y_test, predict)

df_map = df_map.loc[y_test.index.values]
df_map['pred'] = predict

# plotting neighborhoods in the test set
base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(100, 50))
df_map.plot(ax=base, color='bisque', linewidth=0)


base = geodata10.plot(facecolor='grey',linewidth=0, figsize=(150, 75))
base2 = df_map[df_map.gent == 1].plot(ax=base, color='bisque', linewidth=0) # plots gentrified neighborhoods in the test set
df_map[(df_map.gent==1) & (df_map.pred==1)].plot(ax=base2, color='midnightblue', linewidth=0) # colors correctly predicted neighborhoods blue 
