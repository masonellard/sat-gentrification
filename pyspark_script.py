#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:33:27 2021

@author: mason
"""

import numpy as np
from pyspark.sql.functions import split, explode, col, lower, sort_array, expr
from pyspark import SQLContext
import pyspark
from pyspark import SparkContext


spark = pyspark.sql.SparkSession.builder.config('spark.driver.memory', '3g').config('spark.executor.memory', '3g').getOrCreate()

census = spark.read.csv('hdfs:///data/census*', header='true', inferSchema='true')
census.count()

census.printSchema()
census = census.withColumn('GEO_ID', split('GEO_ID', 'US')[1])

census.show(5)

census = census.na.drop()

census = census.filter(census.B19013_001E > 0)

census = census.withColumn('ed_total', sum([census.B15002_032E,
                                   census.B15002_033E,
                                   census.B15002_034E,
                                   census.B15002_035E,
                                   census.B15002_015E,
                                   census.B15002_016E,
                                   census.B15002_017E,
                                   census.B15002_018E])/sum([census.B15002_019E,census.B15002_002E]))

census = census.withColumn('ed_fem', sum([census.B15002_032E,
                                 census.B15002_033E,
                                 census.B15002_034E,
                                 census.B15002_035E]) / census.B15002_019E)

census = census.withColumn('ed_male', sum([census.B15002_015E,
                                  census.B15002_016E,
                                  census.B15002_017E,
                                  census.B15002_018E]) / census.B15002_002E)

census = census.withColumn('trans_solo', census.B08301_003E/census.B08301_001E)

census = census.withColumn('trans_pool', census.B08301_004E/census.B08301_001E)

census = census.withColumn('trans_public', census.B08301_010E/census.B08301_001E)

census = census.withColumn('trans_walk', census.B08301_019E/census.B08301_001E)

census = census.withColumn('trans_remote', census.B08301_021E/census.B08301_001E)

census = census.withColumn('perc_white', census.B02001_002E/census.B02001_001E)

census = census.withColumn('perc_black', census.B02001_003E/census.B02001_001E)

census = census.withColumn('med_hous_inc', census.B19013_001E)

census = census.withColumn('per_cap_inc', census.B19301_001E)

census = census.withColumn('med_age', census.B01002_001E)

census = census.withColumn('population', census.B01003_001E)

census.printSchema()

census = census.select(['GEO_ID', 'NAME', 'state', 'county', 'tract', 'year', 'ed_total', 'ed_fem', 
              'ed_male', 'trans_solo', 'trans_pool', 'trans_public', 'trans_walk', 'trans_remote',
              'perc_white', 'perc_black', 'med_hous_inc', 'per_cap_inc', 'med_age', 'population'])

census.printSchema()

percentile = census.filter(census.year <= 2015).groupby(['state','year']).agg(expr('percentile(med_hous_inc, array(0.10))')[0].alias('%10'))
percentile.show()

for idx, row in enumerate(percentile.collect()):
    temp =  census.filter(census.state == row['state']).filter(census.year == row['year']).filter(census.med_hous_inc <= row['%10'])    
    if idx == 0:
        df_elig = temp
    else:
        df_elig = df_elig.union(temp)


census.printSchema()

df_elig.printSchema()


census.count()

percentile = census.groupby(['state','year']).agg(expr('percentile(med_hous_inc, array(0.20))')[0].alias('%20'))
percentile.show()

percentile_ar = np.array(percentile.collect())


def gentrified(state, year, geo_id, per_cap_inc):
    try:
        tract_summary = census_ar[(census_ar[:, 5] > year) & (census_ar[:, 5] <= year+3) & (census_ar[:, 0] == geo_id)][:, [5, 16, 17]]
        max_income = max(tract_summary[:, 2])
        if ((max_income - per_cap_inc) / per_cap_inc) >= .2:
            return int(1)
        else:
            for row in tract_summary:
                if row[1] >= percentile_ar[(percentile_ar[:, 0] == state) & (percentile_ar[:, 1] == year)][:, 2].item():
                    return int(1)
                else:
                    return int(0)
    except:
        return int(0)

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

states = [1,
 2,
 4,
 5,
 6,
 8,
 9,
 10,
 11,
 12,
 13,
 15,
 16,
 17,
 18,
 19,
 20,
 21,
 22,
 23,
 24,
 25,
 26,
 27,
 28,
 29,
 30,
 31,
 32,
 33,
 34,
 35,
 36,
 37,
 38,
 39,
 40,
 41,
 42,
 44,
 45,
 46,
 47,
 48,
 49,
 50,
 51,
 53,
 54,
 55,
 56]

df_list = []
for state in states:
    census_ar = np.array(census.filter(census.state == state).collect())
    temp = df_elig.filter(df_elig.state == state)
    labeler = udf(gentrified, IntegerType())
    df_list.append(temp.withColumn('gent', labeler('state', 'year', 'GEO_ID', 'per_cap_inc')))

sc = spark.sparkContext
df_model = spark.createDataFrame(sc.union([df.rdd for df in df_list]))

df_model.printSchema()

df_model.count()

df_model.filter(df_model.gent == 1).count()

geo = spark.read.option('header', True).csv('hdfs:///data/geo*')


geo.printSchema()

df_model.printSchema()

df_model = df_model.join(geo.select(['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']), df_model.GEO_ID == geo.GEOID10, how='left')

df_model.printSchema()

census.printSchema()

census = census.join(geo.select(['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'ALAND10']), census.GEO_ID == geo.GEOID10, how='left')

census.count()

census.printSchema()

df_model.count()

census = census.na.drop()
df_model = df_model.na.drop()

from pyspark.sql.types import *

census = census.withColumn('INTPTLAT10', census.INTPTLAT10.cast(FloatType()))
census = census.withColumn('INTPTLON10', census.INTPTLON10.cast(FloatType()))
census = census.withColumn('ALAND10', census.ALAND10.cast(IntegerType()))


df_model = df_model.withColumn('INTPTLAT10', df_model.INTPTLAT10.cast(FloatType()))
df_model = df_model.withColumn('INTPTLON10', df_model.INTPTLON10.cast(FloatType()))
df_model = df_model.withColumn('ALAND10', df_model.ALAND10.cast(IntegerType()))

df_model.printSchema()

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

census.printSchema()

geo_array = np.array(census.select(['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']).collect())

type(str(geo_array[0,0]))

float(geo_array[0, 1])

int(geo_array[0, 3])

def neighbors(geo_id, lat, lon, year):
    
    distances = []
    
    for geo_id2, lat2, lon2, year2 in geo_array:
        if int(year2) == year:
            distances.append((geo_id2, tract_dist(lat, lon, float(lat2), float(lon2))))
            
    distances.sort(key = lambda x: x[1])
    return str(distances[n+1][0])
    

num_neighbors = 10

list(range(num_neighbors))

for n in range(num_neighbors):

    df_list = []
    for state in states:
        geo_array = np.array(census.filter(census.state == state).select(['GEOID10', 'INTPTLAT10', 'INTPTLON10', 'year']).collect())
        temp = df_model.filter(df_model.state == state)
        neighbor_function = udf(neighbors, StringType())
        df_list.append(temp.withColumn('neighbor'+str(n), neighbor_function('GEO_ID', 'INTPTLAT10', 'INTPTLON10', 'year')))

        df_model = spark.createDataFrame(sc.union([df.rdd for df in df_list]))

df_model.printSchema()


df_model.show()

df_model.count()


df_merge = census.select(['GEO_ID',
                   'year',
                   'ed_total',
                   'trans_solo',
                   'trans_pool',
                   'trans_public',
                   'trans_walk',
                   'trans_remote',
                   'perc_white',
                   'perc_black',
                   'med_hous_inc',
                   'per_cap_inc',
                   'med_age',
                   'population',
                   'ALAND10'])

for n in range(num_neighbors):
    
    dfj = df_merge.select(col('GEO_ID').alias(str(n)+'_GEO_ID'),
                               col('year').alias(str(n)+'_year'),
                               col('ed_total').alias(str(n)+'_ed_total'),
                               col('trans_solo').alias(str(n)+'_trans_solo'),
                               col('trans_pool').alias(str(n)+'_trans_pool'),
                               col('trans_public').alias(str(n)+'_trans_public'),
                               col('trans_walk').alias(str(n)+'_trans_walk'),
                               col('trans_remote').alias(str(n)+'_trans_remote'),
                               col('perc_white').alias(str(n)+'_perc_white'),
                               col('perc_black').alias(str(n)+'_perc_black'),
                               col('med_hous_inc').alias(str(n)+'_med_house_inc'),
                               col('per_cap_inc').alias(str(n)+'_per_cap_inc'),
                               col('med_age').alias(str(n)+'_med_age'),
                               col('population').alias(str(n)+'_population'),
                               col('ALAND10').alias(str(n)+'_land_area'))
    
    df_model = df_model.join(dfj, (df_model['neighbor'+str(n)] == dfj[str(n)+'_GEO_ID']) & (df_model.year == dfj[str(n)+'_year']), how='left')
    

df_model.printSchema()

df_model.toPandas().to_csv('national_df.csv')

