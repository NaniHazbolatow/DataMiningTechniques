import pandas as pd
import numpy as np
import openpyxl as px
import re

# load in the basic data
basic_data = pd.read_excel('./Group Assignment 1/basic_data.xlsx')

# how many observations are there
len(basic_data) # 245

# what are the attributes here?
str(basic_data)

# generating summary statistics of t
basic_data.describe()

# TODO

# Task 2 Classification

# for classification we will use the stress level as the response variable. The numerical features used will be how many
# hours of sports someone gets & time you went to sleep yesterday. Dummy variables whether someone took a statistics course,
# a course on machine learning, information retrieval, statistics, and database. Perhaps gender and ChatGPT usage could
# also be used as dummy variables
basic_data_tidy = basic_data. \
    rename(columns={'What programme are you in?': 'program',
                    'When is your birthday (date)?': 'birthday_date',
                    'Have you taken a course on machine learning?':'machine_learning',
                    'Have you taken a course on information retrieval?': 'information_retrieval',
                    'Have you taken a course on statistics?': 'statistics',
                    'Have you taken a course on databases?': 'databases',
                    'What is your gender?': 'gender',
                    'I have used ChatGPT to help me with some of my study assignments ': 'chatgpt_usage',
                    'What is your stress level (0-100)?': 'stress_level',
                    'How many hours per week do you do sports (in whole hours)? ': 'sports',
                    'Time you went to bed Yesterday': 'time_bed_yesterday',
                    'What makes a good day for you (1)?': 'good_day1',
                    'What makes a good day for you (2)?': 'good_day2'})

# verify the columns were renamed
basic_data_tidy.columns

# Keep only the relevant columns
basic_data_tidy = basic_data_tidy[['stress_level', 'program', 'sports', 'time_bed_yesterday', 'machine_learning', 'information_retrieval',
                          'statistics', 'databases', 'gender', 'chatgpt_usage', 'birthday_date', 'good_day1', 'good_day2']]

# I will clean the data up a bit now. Taking a look at the answers for the program someone is in
set(basic_data_tidy['program'])

# Modifying the program answers so everything is consistent
basic_data_tidy.loc[:, 'program'] = basic_data_tidy['program'].apply(
    lambda x: 'Econometrics' if pd.notnull(x) and ('econometrics' in x.lower() or 'eor' in x.lower())
    else 'Computer Science' if pd.notnull(x) and ('cs' in x.lower() or 'computer science' in x.lower() or 'comp sci' in x.lower())
    else 'Artificial Intelligence' if pd.notnull(x) and ('ai' in x.lower() or 'arti' in x.lower())
    else 'Computational Science' if pd.notnull(x) and ('computational' in x.lower())
    else 'Biomedical Science' if pd.notnull(x) and ('bio' in x.lower())
    else 'Finance' if pd.notnull(x) and ('finance' in x.lower())
    else 'Human Language Technology' if pd.notnull(x) and ('human language' in x.lower())
    else 'Big Data Engineering' if pd.notnull(x) and ('data engineer' in x.lower())
    else 'Green IT' if pd.notnull(x) and ('green' in x.lower())
    else 'Unknown' if pd.notnull(x) and ('ba' in x.lower() or '1234' in x.lower() or 'fintech' in x.lower() or 'master' in x.lower())
    else 'Security' if pd.notnull(x) and ('npn' in x.lower() or 'security' in x.lower())
    else x)

# verifying that worked
set(basic_data_tidy['program'])

# investigating the machine learning answers
set(basic_data_tidy['machine_learning']) # no, unknown, yes

# Adjust pandas display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# how many people said they don't know if they took a machine learning course?
basic_data_tidy[basic_data_tidy['machine_learning'] == 'unknown'] # only 2 people

# If these people don't know if they took a course, we can assume they didn't. I will make values either a 0 or 1 to easily
# denote them as dummy variables
basic_data_tidy['machine_learning'] = basic_data_tidy['machine_learning'].apply(
    lambda x: 1 if x == 'yes' else 0
)

# verifying that worked
set(basic_data_tidy['machine_learning']) # 0 or 1

# investigating stress level
basic_data_tidy['stress_level']

# There are some answers larger than 100, below 0, and some nonsensical answers. I will deal with this in a few ways. First,
# all values larger than 100 will be assigned 100. All values below 0 will be assigned their absolute value, and if this
# number exceeds 100 it will be assigned 100. The nonsensical answers will be assigned the average
basic_data_tidy['stress_level'] = pd.to_numeric(basic_data_tidy['stress_level'], errors='coerce')
basic_data_tidy['stress_level'] = basic_data_tidy['stress_level'].apply(
    lambda x: abs(x) if x < 0 else x
)

# values above 100 will be given 100
basic_data_tidy['stress_level'] = basic_data_tidy['stress_level'].apply(
    lambda x: 100 if x > 100 else x
)

# how many NA values are there?
basic_data_tidy[basic_data_tidy['stress_level'].isna()] # 5

# Computing the average
average_stress = basic_data_tidy['stress_level'].mean()

# what is the average stress?
print(average_stress) # 47.803

# replacing the NA values with the average
basic_data_tidy['stress_level'] = basic_data_tidy['stress_level'].fillna(average_stress)

# Calculate the range of the 'stress_level' column
print(basic_data_tidy['stress_level'].min(), basic_data_tidy['stress_level'].max()) # 0, 100

# now I move to sports
set(basic_data_tidy['sports'])

# there are some nonsensical answers here as well. The highest value that I think realistically could be true is 23. Some
# people put 'hours' in their answer. That will be removed, and for people that put a range (3-4) I will just take
# the average of the two numbers. Someone also wrote zero and that will have to be replaced

# Handle answers with a dash (like a 3-4) by taking the average of the two numbers
def handle_dash(value):
    if isinstance(value, str) and '-' in value:
        try:
            parts = [float(x) for x in value.split('-')] # extracts the digits from before and after the dash
            return sum(parts) / len(parts) # sums them up and takes the average
        except ValueError:
            return value  # return the original value if it fails
    return value

# making the adjustment
basic_data_tidy['sports'] = basic_data_tidy['sports'].apply(handle_dash)

# the zero will be adjusted
basic_data_tidy['sports'] = basic_data_tidy['sports'].apply(
    lambda x: 0 if x == 'zero' else x
)

# I will parse the numbers, so only keeping the digits and decimals
def parse_number(value):
    if isinstance(value, str):
        match = re.search(r'-?\d+(\.\d+)?', value)  # extract digits with decimals
        return float(match.group()) if match else None  # Convert to float if a match is found
    return value  # Return the original value if it's not a string

# someone put down &&&&
basic_data_tidy['sports'] = basic_data_tidy['sports'].apply(
    lambda x: np.nan if x == '&&&&' else x
)

# only display values to the second decimal point
pd.set_option('display.float_format', '{:.2f}'.format)
basic_data_tidy['sports'] = pd.to_numeric(basic_data_tidy['sports'].apply(parse_number).round(2))

# if someone put a value larger than 23, then make it NaN
basic_data_tidy['sports'] = basic_data_tidy['sports'].apply(
    lambda x: np.nan if x > 23 else x
)

# compute the average
average_sports = basic_data_tidy['sports'].mean() # 5.54 hours per week it seems

# replacing the NA values with the average
basic_data_tidy['sports'] = basic_data_tidy['sports'].fillna(average_sports)

# checking it out
basic_data_tidy['sports']

# time you went to bed yesterday
basic_data_tidy['time_bed_yesterday']

# My idea is to instead convert this column into the number of hours someone slept. I will do this by assuming everyone
# wakes up at 7:30. I will convert the : to periods to make the conversion easier, and I will get rid of the am/pm stuff.
# Someone also wrote 'around midnight'.
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: str(x).replace(':', '.').replace('am', '').replace('pm', ''). \
        replace('around midnight', '0.00').replace('AD', "").replace(' AM', ""). \
        replace('u', ".").replace(' PM', "").replace(' x)', "").replace('h', "."). \
        replace('-', ".").replace(r'Midnig.t', '0').replace('AM', "")
)

# any strings where 12 is in the decimal?
basic_data_tidy[basic_data_tidy['time_bed_yesterday'].str.contains(r'\.12', na=False)] # 0, nice

# All 12s will be replaced with a 0 for ease of computation
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: str(x).replace('12', '0')
)

# any strings where 23 is in the decimal?
basic_data_tidy[basic_data_tidy['time_bed_yesterday'].str.contains(r'\.23', na=False)]['time_bed_yesterday'] # 1, not good

# Replace 23 and 11 with -1 only if it is not in the decimal
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: re.sub(r'\b(23|11)(?=\.\d+)', '-1', str(x)) if pd.notnull(x) else x
)

# verifying it
set(basic_data_tidy['time_bed_yesterday'])

# replacing some stuff. That person who put a crazy number will be replaced with a 0 for midnight.
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].replace('2300', '-1').replace('0200', '2'). \
    replace('1743502757', '0').apply(parse_number)

# Replace 22 and 10 with -2 only if it is not in the decimal
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: re.sub(r'\b(22|10)(?=\.\d+)', '-2', str(x)) if pd.notnull(x) else x
)

# Replace 21 and 9 with -3 only if it is not in the decimal
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: re.sub(r'\b(21|9)(?=\.\d+)', '-3', str(x)) if pd.notnull(x) else x
)

# glancing at how it looks like so far
set(basic_data_tidy['time_bed_yesterday'])

# making some manual adjustments
basic_data_tidy['time_bed_yesterday'] = basic_data_tidy['time_bed_yesterday'].replace('11.0', '-1').replace('23.0', '-1')

# glancing at how it looks like so far
set(basic_data_tidy['time_bed_yesterday']) # looks good now

# Note: I'm assuming that someone who says they went to sleep at 8 means 8am. Same with 7 being 7am. It's college students
# that put this down, so... In that case I think the time they wake up should be 8:30, or 8.5. I need to modify
# the decimal values to convert them from minutes to fraction values

# Function to convert decimal minutes to fractional hours
def convert_decimal_time_to_hour_fraction(time_val):
    # Make sure it's a float
    try:
        time_val = float(time_val)
    except ValueError:
        return None  # or raise an error, depending on your use case

    hours = int(time_val)
    minutes_decimal = time_val - hours
    minutes = round(minutes_decimal * 100)
    return hours + (minutes / 60)

# do the conversion
basic_data_tidy['time_bed_yesterday'] = pd.to_numeric(basic_data_tidy['time_bed_yesterday'].apply(convert_decimal_time_to_hour_fraction))

# verify it looks good
set(basic_data_tidy['time_bed_yesterday'])

# I think we now assume people get up at 8. so I will do 8 - the values
basic_data_tidy['hours_slept_yesterday'] = basic_data_tidy['time_bed_yesterday'].apply(
    lambda x: 8 - x if pd.notnull(x) else x
)

# taking a look
basic_data_tidy['hours_slept_yesterday'].sort_values()

# Some people are getting more than 8 hours of sleep. My floormates say this is totally normal, so I will take it
# Here are the remaining columns
set(basic_data_tidy['information_retrieval']) # 0, 1, unknown
set(basic_data_tidy['statistics'] # mu should be 1, sigma which should be 0, and unknown
set(basic_data_tidy['databases']) # ja which should be 1, nee which should be 0, and unknown
set(basic_data_tidy['birthday_date']) # some people don't have years...
set(basic_data_tidy['gender']) # so many made up labels...
set(basic_data_tidy['chatgpt_usage']) # no, yes, not willing to say. We all know everyone uses it :)
set(basic_data_tidy['good_day1']) # so many answers
set(basic_data_tidy['good_day2']) # so many answers



