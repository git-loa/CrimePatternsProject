# CrimePatternsProject

Erdos Institute Data Science Project on Crime Patterns and Predictors

## Authors

- Deepesh Singhal
- Yuxin Lin
- Feride Kose
- Leonard Afeke

## Description
The CrimePatternsProject investigates violent crime patterns across the 58 counties of California from 1985 to 2023.

We analyze crime statistics, demographics, economic indicators, housing, education, and government expenditures to find out which factors are the most predictive of crime rates.

We divide the Counties into three groups: Urban, Suburban and Rural and construct separate models for the three categories.

### Overview

**_Stakeholders:_**
1) Local and state government officials and policy makers
2) Community organizations and neighborhood leaders
3) Researchers and public safety analysts
4) Criminal justice and public policy institutions

**_Key performance indicators:_**
Our models are evaluated using mean squared error (MSE) and R². For R², there are two common out-of-sample definitions: one using the mean of the training target values ( \bar{y_train} ) and one using the mean of the test target values ( \bar{y_test} ). We use the version based on \bar{y_train} to compute out-of-sample R². This definition compares the model’s performance against a baseline that always predicts the mean value from the training set. This recent paper https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2216252 justifies this definition of R².

We use three kinds of cross-validation to test our models:
1) Regular 5-fold cross validation.
2) Leave one out cross validation accross counties, where we train our model on all but one counties and then test it on the county that was left out.
3) Time-series cross-validation, we train on data upto 2018 and use that to predict on data from 2019 onwards.

### Datasets
1) Crime statistics (1985 to 2023): "Crimes and Clearances (including Arson)" https://openjustice.doj.ca.gov/data
2) Population (1970 to 2023): https://dof.ca.gov/forecasting/demographics/estimates/
3) Median Age (2010 to 2023): https://data.census.gov/table/ACSST5Y2023.S0101?q=age&g=040XX00US06$0500000
4) Religious demographics (2010 and 2020): https://www.thearda.com/us-religion/census/congregational-membership?y=2020&y2=0&t=0&c=06001
5) Median Household Income (2009 to 2023): https://www.census.gov/data-tools/demo/saipe/#/?s_state=06&s_county=&s_district=&s_geography=county&s_measures=mhi&map_yearSelector=2023&x_tableYears=2023,2022,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021
6) Poverty rate (2009 to 2023): https://www.census.gov/data-tools/demo/saipe/#/?s_state=06&s_county=&s_district=&s_geography=county&s_measures=aa&map_yearSelector=2023&x_tableYears=2023,2022,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021
7) Unemployment rate (1990 to 2023): https://www.bls.gov/lau/tables.htm#cntyaa
8) California CPI (1955 to 2023): https://www.dir.ca.gov/oprl/cpi/entireccpi.pdf
9) Home ownership and rent burden (2010 to 2023): https://data.census.gov/table/ACSDP5Y2023.DP04?q=home+ownership&g=040XX00US06$0500000
10) Median House Value (2010 to 2023): https://data.census.gov/table/ACSDT5Y2023.B25077?q=house+price&g=040XX00US06$0500000
11) Government expenditure (2003 to 2023): https://counties.bythenumbers.sco.ca.gov/#!/year/default
12) Health Insurance (2010 to 2023): https://data.census.gov/table/ACSDP5Y2023.DP03?t=Health:Health+Insurance&g=040XX00US06$0500000
13) Education level (2010 to 2023): https://data.census.gov/table/ACSST5Y2023.S1401?t=School+Enrollment&g=040XX00US06$0500000
14) Educational Attainment (2010 to 2023): https://data.census.gov/table/ACSST5Y2023.S1501?t=School+Enrollment&g=040XX00US06$0500000

### Features
Demographic:
1) Population: The total number of people residing in a county.
2) Population Density: The county's population divided by its area.
3) Median Age: The middle value of the ages of the county’s population.
4) Adherent_rate: The total number of religious adherents in a county divided by its total population.
5) Religion_diversity : The number of religions in a county that are followed by at least 2% of its religious adherents.

Economic:
6) Inflation Adjusted Income: The median household income divided by the Consumer Price Index (CPI).
7) Poverty Rate: The percentage of the population living below the poverty line.
8) Unemployment Rate: The percentage of the labor force that is unemployed.

Housing:
9) House Affordability: The ratio of the median house value to the median household income.
10) Mobile Home Ratio: The ratio of the number of mobile homes to the total number of housing units.
11) Home Ownership Rate: The proportion of occupied housing units that are owner-occupied.
12) Rent Burden: The percentage of people paying more than 35% of their income in rent.
13) Vacancy Rate: The ratio of vacant houses to total housing units.
14) Number of Persons per Household: The average number of individuals living in each household.
15) Renter Ratio: The proportion of population that is renting.

Education:
16) Dropout Rate: One minus the proportion of children attending school.
17) Public School Rate: The ratio of children enrolled in public schools to the total number of children.
18) No High School Rate: The fraction of the adult population without a high school diploma relative to the total adult population.

Health:
19) Uninsured Rate: The proportion of people without health insurance.

Government Expenditure:
20) Adj Police Budget: The police budget adjusted by the CPI and normalized by the county's population.
21) Adj Judiciary Budget: The judiciary budget adjusted for CPI and population factors.
22) Adj Prison Budget: The prison budget adjusted similarly.
23) Adj Education Budget: The education budget adjusted by CPI and county population.
24) Adj Welfare Budget: The welfare budget adjusted for economic factors.
25) Adj Mental Health Budget: The mental health budget adjusted for CPI and population.
26) Adj Rehab Budget: The rehabilitation budget adjusted accordingly.
27) Adj Health Budget: The health budget adjusted for CPI and population measures.

Other:
28) County Type: A categorical variable indicating whether a county is urban, suburban, or rural.
29) Clearance Rate: The ratio of cases solved to the total number of cases.

# Data cleaning and imputation
Many of the datasets have data from a certain year onwards (often from 2010). The dataset of government expenditure is missing all data for San Fransisco County.
We consider two strategies for dealing with missing data:
1) Drop rows (corresponding to certain (County, Year)) for which some entries are missing. This works better for Urban and Rural models.
2) Fix a county and feature, so the values become a time series. Train a simple linear regression of year vs feature on available values and use this simple linear regression to fill in missing values.

### Our model:
We use Multiple linear regression along with Ridge regularization and Principal Component Analysis to prevent overfitting.
We perform DFS to iteratievly remove features that are not contributing to the cross validation R2 score and thus arrive at a subset of the features that performs well.

We use the form log(y) = Ridge(features), this because:
1) it leads to a more accurate model (as compared to y=Ridge(features)).
2) This form will never predict a negative crime-rate for any value of the features.

# Results
The Urban model performs the best and is able to generalize accross counties and accross years. It shows that the dropout rate is by far the most predictive of crime rates.

## Contributing

This project uses pre-commit to ensure code formatting is consistent before
committing. The currently-used hooks are:

- lack for _.py or _.ipynb files.
- prettier (with --prose-wrap always for markdown)

To set up the hooks on your local machine, install pre-commit, then run
pre-commit install to install the formatters that will run before each commit.
