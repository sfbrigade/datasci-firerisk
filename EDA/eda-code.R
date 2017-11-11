##    Exploratory Data Analysis
##    San Francisco Fire Risk Project

library(data.table)
library(tidyverse)
library(lubridate)

### Load Data

taxmap.data <- fread('data/tax_map.csv')                          ## From raw_with_match folder
taxrolls.data <- fread('data/Historic_Secured_Property_Tax_Rolls.csv') 

fire.incidents.data <- fread('data/matched_Fire_Incidents.csv')   ## From raw_with_matched folder


### Data Processing

# Join tax data with fire incident data
# Use tax data from FY 2015
# Use fire incident data from 2011 through 2015

temp_tax <- taxrolls.data %>%
  inner_join(taxmap.data, by = c('Block and Lot Number' = 'Tax_ID'))

temp_tax_2015 <- filter(temp_tax, `Closed Roll Fiscal Year` == 2015)

temp_incident <- fire.incidents.data %>%
  mutate(`Incident Date` = ymd(`Incident Date`)) %>%
  filter(`Incident Date` >= "2011-01-01" & `Incident Date` <= "2015-12-31")

joined_incident <- temp_tax_2015 %>%
  left_join(temp_incident, by = c('EAS BaseID Matched' = 'EAS'))


# Calculate count of incidents, group by Block and Lot Number and Year Built
# Assume age > 0 or age < 200 are outliers due to collection error

collapsed_incident <- joined_incident %>%
  group_by(`Block and Lot Number`, `Year Property Built`) %>%
  summarize(count_incident = sum(!is.na(`Incident Number`))) %>%
  mutate(age = 2017 - `Year Property Built`) %>%
  filter(age > 0 & age < 200)

# Scatterplot: x = age, y = count of incidents

ggplot(data = collapsed_incident, mapping = aes(x = age, y = count_incident)) + 
  geom_point(alpha = 1/2)