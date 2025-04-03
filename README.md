# CrimePatternsProject

Erdos Institute Data Science Project on Crime Patterns and Predictors

## Authors

- Deepesh Singhal
- Yuxin Lin
- Feride Kose
- Leonard Afeke

## Description

### Overview

**_Stakeholder:_**

**_Key performance indicators:_**

### Datasets
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

### Approach

## Contributing

This project uses pre-commit to ensure code formatting is consistent before
committing. The currently-used hooks are:

- lack for _.py or _.ipynb files.
- prettier (with --prose-wrap always for markdown)

To set up the hooks on your local machine, install pre-commit, then run
pre-commit install to install the formatters that will run before each commit.
