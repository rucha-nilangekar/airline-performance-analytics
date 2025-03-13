rm(list = ls()) # remove all existing objects in the environment
gc() # garbage collection


setwd("C:/Users/nilan/OneDrive - Cal State Fullerton/Attachments/Fall_2023/577_capstone")

# 'read.csv' reads in csv file
dat = read.csv(
  'airline_2m.csv',
  head = T,
  stringsAsFactors = F,
  na.strings = ''
)

# 'dim' gives the dimensions of the data: # of rows, # of columns
dim(dat)
str(dat)

# remove the last 48 columns because they’re empty
dat1 <- dat[,-c((ncol(dat) - 47):ncol(dat))]

# check
dim(dat1)

# remove years before 2010
dat2 <- dat1[dat1$Year >= 2010,]

# check
dim(dat2)

# remove preliminary multicollinearity variables
library(dplyr)
dat2 <-
  dat2 %>% select(
    -OriginAirportSeqID,
    -OriginCityMarketID,
    -OriginCityName,
    -OriginStateFips,
    -OriginStateName,
    -OriginWac,
    -DestAirportSeqID,
    -DestCityMarketID,
    -DestCityName,
    -DestStateFips,
    -DestStateName,
    -DestWac,
    -WheelsOff,
    -WheelsOn,
    -Diverted,
    -AirTime,
    -Flights
  )

# replace null cancellation code for non-null value
dat2$CancellationCode <-
  ifelse(is.na(dat2$CancellationCode), '0', dat2$CancellationCode)

# # replace null reason for delay for non- null value
dat2$CarrierDelay     <-
  ifelse(is.na(dat2$CarrierDelay), 'Not Av', dat2$CarrierDelay)
dat2$WeatherDelay     <-
  ifelse(is.na(dat2$WeatherDelay), 'Not Av', dat2$WeatherDelay)
dat2$NASDelay         <-
  ifelse(is.na(dat2$NASDelay), 'Not Av', dat2$NASDelay)
dat2$SecurityDelay    <-
  ifelse(is.na(dat2$SecurityDelay), 'Not Av', dat2$SecurityDelay)
dat2$LateAircraftDelay <-
  ifelse(is.na(dat2$LateAircraftDelay),
         'Not Av',
         dat2$LateAircraftDelay)

# check missing ##
matrix.na = is.na(dat2)
pmiss = colMeans(matrix.na) # proportion of missing for each column
nmiss = rowMeans(matrix.na) # proportion of missing for each row
plot(pmiss) # a few columns with high proportion of missing. we want to exclude them.

print(round(pmiss, digits = 2))

# sort by date
dat2 <- dat2[order(dat2$FlightDate),]

# exporting the file
write.csv(
  dat2,
  'C:/Users/nilan/OneDrive - Cal State Fullerton/Attachments/Fall_2023/577_capstone/airline_cleaned_csv.csv',
  row.names = FALSE
)

#----------------------------------------------------------------------------------------------------
# Adding a column 'Airline Name' to the dataset

dataset = read.csv(
  'airline_cleaned_csv.csv',
  head = T,
  stringsAsFactors = F,
  na.strings = ''
)
airline_code = read.csv(
  'Airline_codes_to_names_mapping.csv',
  head = T,
  stringsAsFactors = F,
  na.strings = ''
)


dim(dataset)
View(dataset)
str(dataset)

#Airline_codes dataset cleaning
airlineCodes_cleaned <- na.omit(airline_code)
is_duplicate <-
  duplicated(airlineCodes_cleaned$Reporting_Airline) #Did this becoz there were duplicate codes. So removed the second occurence and just kept the first one.
airline_Data <- airlineCodes_cleaned[!is_duplicate,]


#Inner join and rearrangement of column order
library(dplyr)

joined_airlines_3 <-
  inner_join(dataset, airline_Data, by = 'Reporting_Airline')
Final_dataset <-
  joined_airlines_3 %>% select(
    Year,
    Quarter,
    Month,
    DayofMonth,
    DayOfWeek,
    FlightDate,
    Reporting_Airline,
    Airline.Name,
    everything()
  ) #Changed the order of the columns. Moved the airline names column next to the codes.

#Write the final dataset
file_path <-
  "C:/Users/nilan/OneDrive - Cal State Fullerton/Attachments/Fall_2023/577_capstone/DATASET_capstone.csv"
write.csv(Final_dataset, file = file_path, row.names = FALSE)
