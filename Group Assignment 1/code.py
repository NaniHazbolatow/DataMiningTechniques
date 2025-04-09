import pandas as pd
import numpy as np
import openpyxl as px

# load in the basic data
basic_data = pd.read_excel('/Users/valentindonchev/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Data Mining/DataMiningTechniques/Group Assignment 1/basic_data.xlsx')

# how many observations are there
len(basic_data) # 245

# what are the attributes here?
str(basic_data)


# generating summary statistics of t
basic_data.describe()










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
                          'statistics', 'databases', 'gender', 'chatgpt_usage', 'birthday_date', 'time_bed_yesterday',
                          'good_day1', 'good_day2']]

# I will clean the data up a bit now. Taking a look at the answers for the program someone is in
set(basic_data_tidy['program'])

# I will adjust the names of the programs to be more readable, creating common categories
clean_data = basic_data_tidy.copy()

clean_data['program'] = clean_data['program'].apply(
    lambda x: 'Econometrics' if pd.notnull(x) and ('econometrics' in x.lower() or 'eor' in x.lower())
    else 'Computer Science' if pd.notnull(x) and ('cs' in x.lower() or 'computer science' in x.lower() or 'comp sci' in x.lower())
    else 'Artificial Intelligence' if pd.notnull(x) and ('ai' in x.lower() or 'arti' in x.lower())
    else 'Computational Science' if pd.notnull(x) and ('computational' in x.lower())
    else 'Biomedical Science' if pd.notnull(x) and ('bio' in x.lower())
    else 'Finance' if pd.notnull(x) and ('finance' in x.lower())
    else 'Human Language Technology' if pd.notnull(x) and ('human language' in x.lower())
    else 'Big Data Engineering' if pd.notnull(x) and ('data engineer' in x.lower())
    else 'Green IT' if pd.notnull(x) and ('green' in x.lower())
    else 'Unknown' if pd.notnull(x) and ('data engineer' in x.lower())
    else x
)

set(clean_data['program'])






# look into optuna



basic_data_tidy





