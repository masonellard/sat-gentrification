# Predicting Gentrification in California Neighborhoods

I used California census tract (neighborhood) data from 2010 - 2018 to try and predict whether the neighborhoods in the bottom 10th percentile for median household income from 2010-2015 would go on to be 'gentrified' in the 3 years following the year they were analyzed.

# Data

**Source:**
* 5-Year American Community Survey Data for California at the census tract level from 2010-2018
* TIGER/Line Geodata shapefile for California census tracts (based on 2010 census)

**Features:**
* ed_total - Percent of population over 25 with at least a Bachelors degree
* trans_solo - Percent of labor force that drives alone to work
* trans_pool - Percent of labor force that carpools to work
* trans_public - Percent of labor force that takes public transportation to work
* trans_walk - Percent of labor force that walks to work
* trans_remote - Percent of labor force that works remotely
* race_white - Percent of population that is white
* race_black - Percent of population that is black
* med_house_inc - Median household income
* per_cap_inc - Per capita income
* med_age - Median age
* population - total population
* ALAND10 - total land area
* neighbor tracts - All of the aforementioned features were also included for the top n closest neighborhoods for each tract

## Tools

**Software Tools:**
* NumPy
* Pandas
* GeoPandas
* scikit-learn
* Keras
* Matplotlib
* Seaborn

**Algorithms:**
* Gaussian Naive Bayes
* Logistic Regression
* Random Forest
* XGBoost
* Neural Network
