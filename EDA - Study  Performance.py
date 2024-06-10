#!/usr/bin/env python
# coding: utf-8

# # Exploratory  Data  Analysis :-
# 
# ##     Student_performance
# 
# ###  Level : Begineer

# # Import  libraries:- 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import plotly.graph_objs as go


# # Import  Datasets :-

# In[2]:


study = pd.read_csv(r"Desktop\study_performance.csv")


# ## Showing the Dataset:-

# In[3]:


study


# In[5]:


study.head()


# In[6]:


study.tail()


# In[35]:


study.sample(5)


# ## Conclusion :- The dataset is not biased..

# ## Summary of  Datasets :-
# 
# About Dataset
# 
# Problem Statement:
# This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.
# 
# Content
# 
# This data set consists of the marks secured by the students in various subjects.
# 
# gender : sex of students -> (Male/female)
# 
# race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
# 
# parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,- high school)
# 
# lunch : having lunch before test (standard or free/reduced)
# 
# test preparation course : complete or not complete before test
# 
# math score
# 
# reading score
# 
# writing score
# 
# Inspiration:
# 
# To understand the influence of the parent's background, test preparation etc on students' performance

# ## General view of dataset :-

# In[7]:


type(study)


# In[9]:


study.shape


# In[10]:


study.columns


# In[12]:


total_records = np.product(study.shape)
total_records


# In[19]:


study.nunique()


# In[20]:


study.describe()


# ##  Checking  the null values :-

# In[25]:


study.isnull().sum()


# In[36]:


study.isnull().any()


# In[37]:


study.isnull().sum().values.sum()


# ## Checking  the  Duplicated  rows:-

# In[28]:


study.duplicated().sum()


# In[21]:


study.info(


# # Conclusion :-  
# 
# 1- It having no null values.
# 
# 2- It having no duplicated row.
# 
# 3- In this, we have 5 columns which have "object dtype", so it's convert into "category dtype"

# In[45]:


study = study.astype({"gender":"category"})
study = study.astype({"race_ethnicity":"category"})
study = study.astype({"parental_level_of_education":"category"})
study = study.astype({"lunch":"category"})
study = study.astype({"test_preparation_course":"category"})


# In[46]:


study.info()


# # Visualizing  the  dataset :-

# ## Types of columns:-
# 
# 1- NUMERICAL :- math_score, reading_score, writing_score
# 
# 2- CATEGROICAL :- gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course
# 
# 3- MIXED :- None column.

# ## Univariate anaysis :-

# In[37]:


study["gender"].value_counts().plot.pie(autopct="%1.2f%%")


# In[42]:


sns.countplot(study["race_ethnicity"])


# In[34]:


study["parental_level_of_education"].value_counts().plot.pie(autopct="%1.2f%%")


# In[91]:


sns.countplot(study["test_preparation_course"])


# # Bivariate  Analysis :-

# In[46]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
sns.barplot(x=study["test_preparation_course"], y=study["math_score"], ci=False)

plt.subplot(3,1,2)
sns.barplot(x=study["test_preparation_course"], y=study["reading_score"], ci=False)

plt.subplot(3,1,3)
sns.barplot(x=study["test_preparation_course"], y=study["writing_score"], ci=False)

plt.show()


# In[49]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
sns.barplot(x=study["gender"], y=study["math_score"], ci=False)

plt.subplot(3,1,2)
sns.barplot(x=study["gender"], y=study["reading_score"], ci=False)

plt.subplot(3,1,3)
sns.barplot(x=study["gender"], y=study["writing_score"], ci=False)


# In[50]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
sns.barplot(x=study["race_ethnicity"], y= study["math_score"],ci=False)

plt.subplot(3,1,2)
sns.barplot(x=study["race_ethnicity"], y= study["reading_score"],ci=False)

plt.subplot(3,1,3)
sns.barplot(x=study["race_ethnicity"], y= study["writing_score"],ci=False)


# In[55]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
plt.barh(study["parental_level_of_education"],study["math_score"], color="skyblue")

plt.subplot(3,1,2)
plt.barh(study["parental_level_of_education"],study["writing_score"], color="green")

plt.subplot(3,1,3)
plt.barh(study["parental_level_of_education"],study["reading_score"], color="red")


# In[56]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
sns.barplot(x=study["lunch"], y = study["math_score"], ci=False)

plt.subplot(3,1,2)
sns.barplot(x=study["lunch"], y = study["reading_score"], ci=False)

plt.subplot(3,1,3)
sns.barplot(x=study["lunch"], y = study["writing_score"], ci=False)


# In[58]:


plt.figure(figsize=(7,15))
plt.subplots_adjust()

plt.subplot(3,1,1)
sns.boxplot(study['math_score'])

plt.subplot(3,1,2)
sns.boxplot(study['reading_score'])

plt.subplot(3,1,3)
sns.boxplot(study['writing_score'])


# ## Checking the skewness :-

# In[59]:


print(study["math_score"].skew())
print(study["reading_score"].skew())
print(study["writing_score"].skew())


# ## Conclusion :- 
# 
# ### 1-  All these 3 have very slightly negative skewness (Left-skewed)
# 
# ### 2- All these 3 columns have outliers..

# # Multi-variate  Analysis :

# In[9]:


sns.barplot(x=study["gender"], y=study["math_score"], hue=study["lunch"], ci=False)


# In[15]:


sns.pairplot(data=study)


# # Correlation :-

# In[21]:


df = study.corr()
df


# In[28]:


sns.heatmap(df, annot=True, linecolor="black", linewidths=1, cmap="Greens")


# # QUESTIONs :
# 
# How the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

# # Overall  Observation  of  this  dataset  is:
# 
# 1- In this, the Female students are more than the male students.
# 
# 2- THE TEST SCORE IS AFFECTED BY THE OTHER'S VARIABLE ARE :
# 
# ==>GENDER: The males students got the highest marks in maths while the Female students got the high score in reading and writing.
# 
# ==> ETHNICITY: .The groupC student is the most and group A is the least. and the Group E student got the highest score.
# 
# ==> PARENTAL LEVEL OF EDUCATION: Most parents have some college degree and the masters degree parents are the least. While The parents who have masters degree theirs child got least marks in MATHS.
# 
# For reading and writing, their is no relation between parents degree and student's score.
# 
# ==> LUNCH: The female students who get the standatd lunch got high score.
# 
# ==> TEST PREPARATION COURSE: The students who completed their preparation before test getting the high score.
# 
# 
# 

#                                                                                                     AUTHOR:
#                                                                                                      NIDHI SHARMA
#                                                                                                      
