# Cortexia Darkzones

SIT Academy Capstone project, Cortexia Darkzones

Zurich, Switzelrand

*by Valeriia Rutskaia and Dominik Bacher*

## Journal
### Day 1 - Monday, 4. July 2022
Valeriia Rutskaia
- Data investigation
- Identified that there can be no litter or many types of litter on one place
- Discovered that value.vehicle_mode and speed are 0

Dominik Bacher
- EDA to understand the data
- Identified 60% of data as having no litter
- Created basic Plotly scattermap of Basel


### Day 2 - Tuesday, 5. July 2022
Valeriia Rutskaia
- Data cleaning
- Aggregating the data on edge_id and date

Dominik Bacher
- OSM exploration and familiarization
- Ran a first prediciton model (it was bad, we need more features).
- Created a color coded map with different types of edges (highway, footway, etc.)


### Day 3 - Wednesday, 6. July 2022
Valeriia Rutskaia
- Resolve duplicate problem of edge_id directionality (a,b is the same as b,a).
- Feature extracting: year, month, day, holiday it, weekend or working day.
- Building graphs

Dominik Bacher
- Adapted the geojson file to be interpreted by OSM
- Downloaded OSM feautures of Basel (restaurants, businesses, stores, etc.)


### Day 4 - Thursday, 7. July 2022
Valeriia Rutskaia
- Adapted the code for new data 
- Different aggregation with suitcase_id (by sum(max, mean) of the litter). 
- Creating dark zones

Dominik Bacher
- Cleaning the 2nd dataset we got
- Added categorical feautures
- Ran KNN and PyCaret, we need more features
- Creating dummy rows for the dark zones


### Day 5 - Friday, 8. July 2022
Valeriia Rutskaia
- Built models for linear regression and KNeighborsRegressor.

Dominik Bacher
- Added weather data as a feature
- Feauture tunning to run Logistic Regression and KNN prediciton models
- Merging of both old and new datasets (NO IMPROVEMENT IN PREDICITON SCORES)


### Day 6 - Monday, 11. July 2022
Valeriia Rutskaia
- Added feature for holidays and 2 days after the holiday 
- Discovered exponential behavior of total litter per edge 
- Built poisson ression model with D-square score 0.58. (SO FAR OUR BEST BET FOR THE FINAL MODEL)

Dominik Bacher
- Added latitude and longitude features based on edges.geojson
- Started working on adding count of points on interest from OSM (park, atm, etc.) as a feature


### Day 7 - Tuesday, 12. July 2022
Valeriia Rutskaia
- Built distribution of different cases with sum mean and max by suitcase_id
- Total litter_sum gives clusters
- Poisson distribution shows the best score on total_sum

Dominik Bacher
- Finished adding points of interest from OSM to the data


### Day 8 - Wednesday, 13. July 2022
Valeriia Rutskaia
- Tried to predict some litters with existing Poisson model
- Graphs for each litter type
- Started SHAPELY, it didn't work with greadsearch

Dominik Bacher
- Improved import functions for easy import of files on Colab
- Added length of edge segment, from coordinates to meters
- XGBoost feature importance


### Day 9 - Thursday, 14. July 2022
Valeriia Rutskaia
- Feature Importance by adding features which increase score

Dominik Bacher
- Implemented all the features into one dataframe
- Created function to obtain feature importance


### Day 10 - Friday, 15. July 2022
Valeriia Rutskaia
- Feature Importance by adding features which increase score
- Each litter performance

Dominik Bacher
- Created the script to create the dataframe with the darkzones


### Day 11 - Monday, 18. July 2022
Valeriia Rutskaia
- Each litter performance

Dominik Bacher
- Final notebook with ML model fitting and predictor


### Day 12 - Tuesday, 19. July 2022
Valeriia Rutskaia
- Feature importance through size of coefficients
- Gradient Boosting Regression Trees for Poisson regression

Dominik Bacher
- Final notebook with ML model fitting and predictor
- The user can specify one or more litters to predict from a list


### Day 13 - Wednesday, 20. July 2022
Valeriia Rutskaia
- Models Comparisson with GreadSearch
- Permutation Feature Importance
- Graph for the feature importance

Dominik Bacher
- Created notebook to map the desired edges using Open Street Maps and folium


### Day 14 - Thursday, 21. July 2022
Valeriia Rutskaia
- Trying to calculate mean relative error


Dominik Bacher
- Created graph and formula to obtain and visualize the relative error of each prediction


### Day 15 - Friday, 22. July 2022
Valeriia Rutskaia
- ...

Dominik Bacher
- Trying to further improve the model


### Day 16 to 19 - Monday 25. to Thurday 28. July 2022
Valeriia Rutskaia
- Calculations for one tested route
- Project Description
- Preparation for the Final Presentation

Dominik Bacher
- Final adjustments to the scripts
- Document the functions
- Create the slides for the presentation


### Day 20 - Frieday, 29. July 2022
Final Presentation
