# CrimePatternsProject

**Erdos Institute Data Science Project on Crime Patterns and Predictors**

## Authors

- Deepesh Singhal
- Yuxin Lin
- Feride Kose
- Leonard Afeke

## Description
The CrimePatternsProject investigates violent crime patterns across the 58 counties of California from 1985 to 2023.

We analyze crime statistics, demographics, economic indicators, housing, education, and government expenditures to find out which factors are the most predictive of crime rates.

We divide the Counties into three groups: Urban, Suburban and Rural and construct separate models for the three categories.

### Stakeholders:
1. Local and state government officials and policy makers.
2. Community organizations and neighborhood leaders.
3. Researchers and public safety analysts.
4. Criminal justice and public policy institutions.

### Key performance indicators:
Our models are evaluated using mean squared error (MSE) and R². For R², there are two common out-of-sample definitions, one uses the mean of the test target values ytest , and one uses the mean of the training target values ytrain . While sklearn uses the one with ytest ,
we use the version based on ytrain  to compute out-of-sample R². To avoid confusion, we refer to this as the MR2 score (modified R2). This definition compares the model’s performance against a baseline that always predicts the mean value from the training set 
MR2=1-MSE(model)/MSE(ytrain).
This recent paper also https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2216252 justifies this definition of R².



### Cross‑Validation Schemes
We use three kinds of cross-validation to test our models:
1) Regular 5-fold cross validation.
2) Leave one out cross validation across counties, where we train our model on all but one county and then test it on the county that was left out.
3) Time-series cross-validation, we train on data up to 2018 and use that to predict on data from 2019 onwards.

## Datasets
1) Crime statistics (1985 to 2023): "Crimes and Clearances (including Arson)" https://openjustice.doj.ca.gov/data
2) Population (1970 to 2023): https://dof.ca.gov/forecasting/demographics/estimates/
3) Median Age (2010 to 2023): https://data.census.gov/table/ACSST5Y2023.S0101?q=age&g=040XX00US06$0500000
4) Religious demographics ():
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
15) Religion Data (2010 and 2020):
https://www.thearda.com/us-religion/census/congregational-membership?y=2020&y2=0&t=0&c=06001

## Features
Our target variable is crime rate: The ratio of the number of cases of violent crimes in the county divided by the population of that county.

-**Demographic:**
1. Population: The total number of people residing in a county.
2. Population Density: The county's population divided by its area.
3. Median Age: The middle value of the ages of the county’s population.
4. Adherence rate: The proportion of the population that is religious.
5. Religious diversity: The number of religions followed by at least 2% of the population.

-**Economic:**
6. Inflation Adjusted Income: The median household income divided by the Consumer Price Index (CPI).
7. Poverty Rate: The percentage of the population living below the poverty line.
8. Unemployment Rate: The percentage of the labor force that is unemployed.

-**Housing:**
9. House Affordability: The ratio of the median house value to the median household income.
10. Mobile Home Ratio: The ratio of the number of mobile homes to the total number of housing units.
11. Home Ownership Rate: The proportion of occupied housing units that are owner-occupied.
12. Rent Burden: The percentage of people paying more than 35% of their income in rent.
13. Vacancy Rate: The ratio of vacant houses to total housing units.
14. Number of Persons per Household: The average number of individuals living in each household.
15. Renter Ratio: The proportion of the population that is renting.

-**Education:**
16. Dropout Rate: The proportion of 15 to 17 year olds not attending school.
17. Public School Rate: The ratio of children enrolled in public schools to the total number of children.
18. High School Rate: The fraction of the adult population with a high school diploma relative to the total adult population.

-**Health:**
19. Uninsured Rate: The proportion of people without health insurance.

-**Government Expenditure:**
20. Adj Police Budget: The police budget adjusted by the CPI and normalized by the county's population.
21. Adj Judiciary Budget: The judiciary budget adjusted for CPI and population factors.
22. Adj Prison Budget: The prison budget adjusted similarly.
23. Adj Education Budget: The education budget adjusted by CPI and county population.
24. Adj Welfare Budget: The welfare budget adjusted for economic factors.
25. Adj Mental Health Budget: The mental health budget adjusted for CPI and population.
26. Adj Rehab Budget: The rehabilitation budget adjusted accordingly.
27. Adj Health Budget: The health budget adjusted for CPI and population measures.
28. Security_vs_Social: The ratio of security spending to social spending.
29. Social_vs_Security: The ratio of social spending to security spending.

-**Other:**
30. County Type: A categorical variable indicating whether a county is urban, suburban, or rural.
31. Clearance Rate: The ratio of cases solved to the total number of cases.

# Data cleaning and imputation
Many of the datasets have data from a certain year onwards (often from 2010). The dataset of government expenditure is missing all data for San Francisco County.
We consider two strategies for dealing with missing data:
1. **Row deletion** for missing values (works best for Urban/Rural)  
2. **Time‑series imputation:** fit a simple linear regression (feature vs. year) per county and fill missing values  

## Models used for analysis:
We considered multiple linear regression, random forests and XGBoost. We chose multiple linear regression as our final model since:
1. Interpretability for policy insights  
2. Comparable out‑of‑sample performance  
3. Stronger generalization across counties and over time  

### Our model:
We use Multiple linear regression along with Ridge regularization and Principal Component Analysis to prevent overfitting.
We use poly feature of degree two to create cross term features (for example: Uninsured Rate-population density)
We perform DFS to iteratively remove features that are not contributing to the cross validation R2 score and thus arrive at a subset of the features that performs well.

We use the form log(y) = Ridge(features), this because:
1. it leads to a more accurate model (as compared to y=Ridge(features)).
2. This form will never predict a negative crime-rate for any value of the features.

## Results
The Urban model performs the best and is able to generalize across counties and across years.
The top 4 most important features are:
1. Security vs Social. This is the ratio of security spending to social spending, it indicates that security spending is more helpful to reducing crime rate than social spending.
2. Clearance rate. This is the ratio of cases that are solved by the police. So having an effective police department that solves cases is important to having a lower crime rate.
3. Adjusted income. This is the median household income adjusted for inflation. So higher income levels are helpful in reducing crime rate.
4. Dropout rate. This is the proportion of children aged 15 to 17 that are not going to school. This indicates that children not going to school increases the crime rate.

The Suburban and Rural models have a relatively low R^2 score (0.4 and 0.3 for cross counties validation). We suspect that the data quality is not so good for these counties.

## Contributing

This project uses pre-commit to ensure code formatting is consistent before
committing. The currently-used hooks are:

- lack for _.py or _.ipynb files.
- prettier (with --prose-wrap always for markdown)

To set up the hooks on your local machine, install pre-commit, then run
pre-commit install to install the formatters that will run before each commit.

