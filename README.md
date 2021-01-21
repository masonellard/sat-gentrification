# Predicting Gentrification in California Neighborhoods

I used California census tract (neighborhood) data from 2010 - 2018 to try and predict whether the neighborhoods in the bottom 10th percentile for median household income from 2010-2015 would go on to be 'gentrified' in the 3 years following the year they were analyzed.

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

## Data

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

## Results

I was able to achieve 67 to 76 percent precision (in some cases) with my best random forest model. Recall remained low - in the 30 to 40 percent range. Overall, when I predicted gentrification to occur, I was most often correct, however my model was not capable of making many predictions relative to how many neighborhoods would actually go on to be gentrified. Interestingly, I was able to improve precision by ~9 percent by limiting my dataset to white-majority neighborhoods in which the combined land area of the 10 surrounding neighborhoods was at least 5 million, and the ed_total feature for those surrounding neighborhoods was at least .25. While it could be interesting to explore why the model is better able to predict white-majority neighborhoods, this condition should obviously not be imposed on the model in practice since that may lead to racial discrimination in how resources are allocated to neighborhoods in order to prevent negative effects of gentrification. Data extraction, cleaning, engineering, and modeling can be found in (https://github.com/masonellard/sat-gentrification/blob/main/gentrification_CA.py)[gentrification_CA.py].
