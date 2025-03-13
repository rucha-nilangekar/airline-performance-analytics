
rm(list = ls())
setwd("C:\\Users\\RuchaN1507\\OneDrive - Cal State Fullerton\\Attachments\\Fall_2023\\577_capstone")

dataset_final= read.csv('Final_dataset_capstone.csv', head=T, stringsAsFactors=F, na.strings='')

#Create another dataframe with just delayminutes and dayofweek
#data_statcheck <- dataset_final[!is.na(dataset_final$DepDelayMinutes) & dataset_final$DepDelayMinutes != 0, c("DepDelayMinutes", "DayOfWeek")]
data_statcheck <- dataset_final[!is.na(dataset_final$DepDelayMinutes) & dataset_final$DepDelayMinutes >= 15, c("DepDelayMinutes", "DayOfWeek")]
data_statcheck_1 <- dataset_final[!is.na(dataset_final$ArrDelayMinutes) & dataset_final$ArrDelayMinutes >= 15, c("ArrDelayMinutes", "DayOfWeek")]

data_statcheck <- data_statcheck[!grepl("NA", data_statcheck$DepDelayMinutes), ]
data_statcheck$DepDelayMinutes <- as.numeric(trimws(data_statcheck$DepDelayMinutes))

data_statcheck_1 <- data_statcheck_1[!grepl("NA", data_statcheck_1$ArrDelayMinutes), ]
data_statcheck_1$ArrDelayMinutes <- as.numeric(trimws(data_statcheck_1$ArrDelayMinutes))

#Convert DayOfWeek to factor
data_statcheck$DayOfWeek <- as.factor(data_statcheck$DayOfWeek)
data_statcheck$DayOfWeek <- ordered(data_statcheck$DayOfWeek, levels = 1:7)


data_statcheck_1$DayOfWeek <- as.factor(data_statcheck_1$DayOfWeek)
data_statcheck_1$DayOfWeek <- ordered(data_statcheck_1$DayOfWeek, levels = 1:7)

####################################Departure Delay Minutes Check############################################
---------------------------------------------------------------------------------------------------------------
#Welch's test 1 (Monday vs the rest of the days )
  
monday_group <- (data_statcheck[data_statcheck$DayOfWeek == 1, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 1, 'DepDelayMinutes'])

# Performing Welch's t-test
mon_ttest_result <- t.test(monday_group, rest_of_days_group)

# Printing t-test results
print(mon_ttest_result)

---------------------------------------------------------------------------------------------------------------
#Welch's t-test 2 (Tuesday vs rest of the days)
tuesday_group <- (data_statcheck[data_statcheck$DayOfWeek == 2, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 2, 'DepDelayMinutes'])

# Performing Welch's t-test
tue_ttest_result <- t.test(tuesday_group, rest_of_days_group)

# Printing t-test results
print(tue_ttest_result)

---------------------------------------------------------------------------------------------------------------
#Welch's t-test 3 (Wednesday vs rest of the days)
  
wednesday_group <- (data_statcheck[data_statcheck$DayOfWeek == 3, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 3, 'DepDelayMinutes'])

# Performing Welch's t-test
wed_ttest_result <- t.test(wednesday_group, rest_of_days_group)

# Printing t-test results
print(wed_ttest_result)
  
----------------------------------------------------------------------------------------------------------------
# welch's t-test (Thursday vs rest of the days)
  
thursday_group <- (data_statcheck[data_statcheck$DayOfWeek == 4, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 4, 'DepDelayMinutes'])

# Performing Welch's t-test
thurs_ttest_result <- t.test(thursday_group, rest_of_days_group)

# Printing t-test results
print(thurs_ttest_result)

----------------------------------------------------------------------------------------------------------------
#Welch's t-test (Friday vs rest of the days)
  
friday_group <- (data_statcheck[data_statcheck$DayOfWeek == 5, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 5, 'DepDelayMinutes'])

# Performing Welch's t-test
fri_ttest_result <- t.test(friday_group, rest_of_days_group)

# Printing t-test results
print(fri_ttest_result)

-------------------------------------------------------------------------------------------------------------------
# Welch's t-test (Saturday vs rest of the days)
  
saturday_group <- (data_statcheck[data_statcheck$DayOfWeek == 6, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 6, 'DepDelayMinutes'])

# Performing Welch's t-test
sat_ttest_result <- t.test(saturday_group, rest_of_days_group)

# Printing t-test results
print(sat_ttest_result)

----------------------------------------------------------------------------------------------------------------------
# Welch's t-test (Sunday vs rest of the days)

sunday_group <- (data_statcheck[data_statcheck$DayOfWeek == 7, 'DepDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck$DayOfWeek != 7, 'DepDelayMinutes'])

# Performing Welch's t-test
sun_ttest_result <- t.test(sunday_group, rest_of_days_group)

# Printing t-test results
print(sun_ttest_result)

-------------------------------------------------------------------------------------------------------------------

###############################Arrival Delay Minutes Check#########################################
  
#Welch's test 1 (Monday vs the rest of the days )
  
monday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 1, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 1, 'ArrDelayMinutes'])

# Performing Welch's t-test
mon_ttest_result_1 <- t.test(monday_group_1, rest_of_days_group_1)

# Printing t-test results
print(mon_ttest_result_1)

---------------------------------------------------------------------------------------------------------------
  #Welch's t-test 2 (Tuesday vs rest of the days)
tuesday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 2, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 2, 'ArrDelayMinutes'])

# Performing Welch's t-test
tue_ttest_result_1 <- t.test(tuesday_group_1, rest_of_days_group_1)

# Printing t-test results
print(tue_ttest_result_1)

---------------------------------------------------------------------------------------------------------------
  #Welch's t-test 3 (Wednesday vs rest of the days)
  
wednesday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 3, 'ArrDelayMinutes'])
rest_of_days_group <- (data_statcheck[data_statcheck_1$DayOfWeek != 3, 'ArrDelayMinutes'])

# Performing Welch's t-test
wed_ttest_result_1 <- t.test(wednesday_group_1, rest_of_days_group_1)

# Printing t-test results
print(wed_ttest_result_1)

----------------------------------------------------------------------------------------------------------------
  # welch's t-test (Thursday vs rest of the days)
  
thursday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 4, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 4, 'ArrDelayMinutes'])

# Performing Welch's t-test
thurs_ttest_result_1 <- t.test(thursday_group_1, rest_of_days_group_1)

# Printing t-test results
print(thurs_ttest_result_1)

----------------------------------------------------------------------------------------------------------------
  #Welch's t-test (Friday vs rest of the days)
  
friday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 5, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 5, 'ArrDelayMinutes'])

# Performing Welch's t-test
fri_ttest_result_1 <- t.test(friday_group_1, rest_of_days_group_1)

# Printing t-test results
print(fri_ttest_result_1)

-------------------------------------------------------------------------------------------------------------------
  # Welch's t-test (Saturday vs rest of the days)
  
saturday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 6, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 6, 'ArrDelayMinutes'])

# Performing Welch's t-test
sat_ttest_result_1 <- t.test(saturday_group_1, rest_of_days_group_1)

# Printing t-test results
print(sat_ttest_result_1)

----------------------------------------------------------------------------------------------------------------------
  # Welch's t-test (Sunday vs rest of the days)
  
sunday_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek == 7, 'ArrDelayMinutes'])
rest_of_days_group_1 <- (data_statcheck_1[data_statcheck_1$DayOfWeek != 7, 'ArrDelayMinutes'])

# Performing Welch's t-test
sun_ttest_result_1 <- t.test(sunday_group_1, rest_of_days_group_1)

# Printing t-test results
print(sun_ttest_result)



  
  
  
  
  
  
  
  
  




















