# Exploring data arrays with NumPy

# Lets start by looking at some simple data in a list format
# Suppose a college takes a sample of student grades for a data science class.
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
"""print(data)"""

# The data above has been loaded into a Python list structure, which is a good data type for general data manipulation, but not optimized for numeric analysis. For that, we're going to use the NumPy package, which includes specific data types and functions for working with Numbers in Python.
import numpy as np

grades = np.array(data)
"""print(grades)"""

# Compare how the list and a NumPy array behave when we use them in an expression that multiplies them by 2.
"""print (type(data),'x 2:', data * 2)
print('\n')
print (type(grades),'x 2:', grades * 2)"""

# the class type for the numpy array above is a numpy.ndarray. The nd indicates that this is a structure that can consists of multiple dimensions (it can have n dimensions). Our specific instance has a single dimension of student grades.
"""print(grades.shape)"""

# Look at the first record in the numpy array
"""print(grades[0])"""

# You can apply aggregations across the elements in the array, so let's find the simple average grade (in other words, the mean grade value).
"""print(grades.mean())"""

# Let's add a second set of data for the same students, this time recording the typical number of hours per week they devoted to studying.
# Define an array of study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]
# Create a 2D array (an array of arrays)
student_data = np.array([study_hours,grades])
# display the array
"""print(student_data)"""

# Now the data consists of a 2-dimensional array - an array of arrays. Let's look at its shape.
"""print(student_data.shape)"""

# To navigate this structure, you need to specify the position of each element in the hierarchy. So to find the first value in the first array (which contains the study hours data), you can use the following code.
"""print(student_data[0][0])"""

# Now you have a multidimensional array containing both the student's study time and grade information, which you can use to compare data. For example, how does the mean study time compare to the mean grade?
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

"""print(f'Average study hours: {avg_study} \nAverage grades : {avg_grade}')"""

# Run the following cell to import the Pandas library and create a DataFrame with three columns. The first column is a list of student names, and the second and third columns are the NumPy arrays containing the study time and grade data.
import pandas as pd

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                                     'StudyHours':student_data[0],
                                     'Grade': student_data[1]})

"""print(df_students)"""

# You can use the DataFrame's loc method to retrieve data for a specific index value, like this.
"""print(df_students.loc[5])"""

# You can also get the data at a range of index values, like this:
"""print(df_students[0:5])"""

# In addition to being able to use the loc method to find rows based on the index, you can use the iloc method to find rows based on their ordinal position in the DataFrame (regardless of the index):
"""print(df_students.iloc[0:5])"""

# iloc identifies data values in a DataFrame by position, which extends beyond rows to columns. So for example, you can use it to find the values for the columns in positions 1 and 2 in row 0, like this:
"""print(df_students.iloc[0,[1,2]])"""

# loc is used to locate data items based on index values rather than positions. In the absence of an explicit index column, the rows in our dataframe are indexed as integer values, but the columns are identified by name:
"""print(df_students.loc[0,'Grade'])"""

# You can use the loc method to find indexed rows based on a filtering expression that references named columns other than the index, like this:
"""print(df_students.loc[df_students['Name'] == 'Aisha'])"""

# Actually, you don't need to explicitly use the loc method to do this - you can simply apply a DataFrame filtering expression, like this:
"""print(df_students[df_students['Name']=='Aisha'])"""

# And for good measure, you can achieve the same results by using the DataFrame's query method, like this:
"""print(df_students.query('Name=="Aisha"'))"""

# You can specify the column name as a named index value (as in the df_students['Name'] examples we've seen so far), or you can use the column as a property of the DataFrame, like this:
"""print(df_students[df_students.Name == 'Aisha'])"""

# Loading a DataFrame from a file
df_students = pd.read_csv('data/grades.csv',delimiter=',',header='infer')
"""print(df_students.head())"""

# Handling missing values
# One of the most common issues data scientists need to deal with is incomplete or missing data. So how would we know that the DataFrame contains missing values? You can use the isnull method to identify which individual values are null, like this:
"""print(df_students.isnull())"""

# Of course, with a larger DataFrame, it would be inefficient to review all of the rows and columns individually; so we can get the sum of missing values for each column, like this:
"""print(df_students.isnull().sum())"""

# we can filter the dataframe to include only rows where any of the columns (axis 1 of the DataFrame) are null.
"""print(df_students[df_students.isnull().any(axis=1)])"""

# One common approach is to impute replacement values. For example, if the number of study hours is missing, we could just assume that the student studied for an average amount of time and replace the missing value with the mean study hours. To do this, we can use the fillna method, like this:
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
"""print(df_students)"""

# Alternatively, it might be important to ensure that you only use data you know to be absolutely correct; so you can drop rows or columns that contains null values by using the dropna method. In this case, we'll remove rows (axis 0 of the DataFrame) where any of the columns contain null values.
df_students = df_students.dropna(axis=0, how='any')
"""print(df_students)"""

# Now that we've cleaned up the missing values, we're ready to explore the data in the DataFrame. Let's start by comparing the mean study hours and grades.
# Get the mean study hours using to column name as an index
mean_study = df_students['StudyHours'].mean()
# Get the mean grade using the column name as a property (just to make the point!)
mean_grade = df_students['Grade'].mean()
# Print the mean study hours and mean grade
"""print(f'Average Study Hours are: {mean_study} \nAverage Grade is: {mean_grade}')"""

# OK, let's filter the DataFrame to find only the students who studied for more than the average amount of time.
# Get students who studied for the mean or more hours
"""print(df_students[df_students.StudyHours > mean_study])"""

# For example, let's find the average grade for students who undertook more than the average amount of study time.
# What was their mean grade?
print(df_students[df_students.StudyHours > mean_study].Grade.mean())

# Let's assume that the passing grade for the course is 60.
# We can use that information to add a new column to the DataFrame, indicating whether or not each student passed.
# First, we'll create a Pandas Series containing the pass/fail indicator (True or False), and then we'll concatenate that series as a new column (axis 1) in the DataFrame.
passes = pd.Series(df_students['Grade'] > 60)
df_students = pd.concat([df_students,passes.rename("Pass")], axis=1)

"""print(df_students)"""

# DataFrames are amazingly versatile, and make it easy to manipulate data. Most DataFrame operations return a new copy of the DataFrame; so if you want to modify a DataFrame but keep the existing variable, you need to assign the result of the operation to the existing variable. For example, the following code sorts the student data into descending order of Grade, and assigns the resulting sorted DataFrame to the original df_students variable.
# Create a DataFrame with the data sorted by Grade (descending)
df_students = df_students.sort_values('Grade', ascending=False)

"""print(df_students)"""

# Visualizing data with Matplotlib
# Ensure plots are displayed inline in the notebook
%matplotlib inline
from matplotlib import pyplot as plt
# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade)
# Display the plot
plt.show()

# Well, that worked; but the chart could use some improvements to make it clearer what we're looking at.
# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')
# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)
# Display the plot
"""plt.show()"""

# A plot is technically contained with a Figure. In the previous examples, the figure was created implicitly for you; but you can create it explicitly. For example, the following code creates a figure with a specific size.
# Create a Figure
fig = plt.figure(figsize=(8,3))
# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')
# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)
# Show the figure
"""plt.show()"""

# A figure can contain multiple subplots, each on its own axis.
# For example, the following code creates a figure with two subplots - one is a bar chart showing student grades, and the other is a pie chart comparing the number of passing grades to non-passing grades.
# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))
# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)
# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())
# Add a title to the Figure
fig.suptitle('Student Data')
# Show the figure
"""fig.show()"""

# Until now, you've used methods of the Matplotlib.pyplot object to plot charts. However, Matplotlib is so foundational to graphics in Python that many packages, including Pandas, provide methods that abstract the underlying Matplotlib functions and simplify plotting. For example, the DataFrame provides its own methods for plotting data, as shown in the following example to plot a bar chart of study hours.
df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))

# Descriptive statistics and data distribution
# When examining a variable (for example a sample of student grades), data scientists are particularly interested in its distribution (in other words, how are all the different grade values spread across the sample). The starting point for this exploration is often to visualize the data as a histogram, and see how frequently each value for the variable occurs.
# Get the variable to examine
var_data = df_students['Grade']
# Create a Figure
fig = plt.figure(figsize=(10,4))
# Plot a histogram
plt.hist(var_data)
# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Show the figure
"""fig.show()"""

# To understand the distribution better, we can examine so-called measures of central tendency; which is a fancy way of describing statistics that represent the "middle" of the data. The goal of this is to try to find a "typical" value. Common ways to define the middle of the data include:
# Let's calculate these values, along with the minimum and maximum values for comparison, and show them on the histogram.
# Get the variable to examine
var = df_students['Grade']
# Get statistics
min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]
print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                        mean_val,
                                                                                        med_val,
                                                                                        mod_val,
                                                                                        max_val))

# Create a Figure
fig = plt.figure(figsize=(10,4))
# Plot a histogram
plt.hist(var)
# Add lines for the statistics
plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)
# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Show the figure
"""fig.show()"""

# Another way to visualize the distribution of a variable is to use a box plot (sometimes called a box-and-whiskers plot). Let's create one for the grade data.
# Get the variable to examine
var = df_students['Grade']
# Create a Figure
fig = plt.figure(figsize=(10,4))
# Plot a histogram
plt.boxplot(var)
# Add titles and labels
plt.title('Data Distribution')
# Show the figure
"""fig.show()"""

# It's often useful to combine histograms and box plots, with the box plot's orientation changed to align it with the histogram (in some ways, it can be helpful to think of the histogram as a "front elevation" view of the distribution, and the box plot as a "plan" view of the distribution from above.)
# Create a function that we can re-use
def show_distribution(var_data):
    from matplotlib import pyplot as plt
    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))
    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))
    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')
    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)
    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')
    # Add a title to the Figure
    fig.suptitle('Data Distribution')
    # Show the figure
    fig.show()
# Get the variable to examine
col = df_students['Grade']
# Call the function
"""show_distribution(col)"""

# To explore this distribution in more detail, you need to understand that statistics is fundamentally about taking samples of data and using probability functions to extrapolate information about the full population of data. For example, the student data consists of 22 samples, and for each sample there is a grade value. You can think of each sample grade as a variable that's been randomly selected from the set of all grades awarded for this course. With enough of these random variables, you can calculate something called a probability density function, which estimates the distribution of grades for the full population.
# The Pandas DataFrame class provides a helpful plot function to show this density.

def show_density(var_data):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10,4))
    # Plot density
    var_data.plot.density()
    # Add titles and labels
    plt.title('Data Density')
    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)
    # Show the figure
    plt.show()
# Get the density of Grade
col = df_students['Grade']
"""show_density(col)"""

# As expected from the histogram of the sample, the density shows the characteristic 'bell curve" of what statisticians call a normal distribution with the mean and mode at the center and symmetric tails.
# Now let's take a look at the distribution of the study hours data.
# Get the variable to examine
col = df_students['StudyHours']
# Call the function
"""show_distribution(col)"""

# Outliers can occur for many reasons. Maybe a student meant to record "10" hours of study time, but entered "1" and missed the "0". Or maybe the student was abnormally lazy when it comes to studying! Either way, it's a statistical anomaly that doesn't represent a typical student. Let's see what the distribution looks like without it.
# Get the variable to examine
col = df_students[df_students.StudyHours>1]['StudyHours']
# Call the function
"""show_distribution(col)"""

# Let's look at the density for this distribution.
# Get the density of StudyHours
"""show_density(col)"""

# So now we have a good idea where the middle of the grade and study hours data distributions are. However, there's another aspect of the distributions we should examine: how much variability is there in the data?
# Typical statistics that measure variability in the data include:
    # Range: The difference between the maximum and minimum. There's no built-in function for this, but it's easy to calculate using the min and max functions.
    # Variance: The average of the squared difference from the mean. You can use the built-in var function to find this.
    # Standard Deviation: The square root of the variance. You can use the built-in std function to find this.
for col_name in ['Grade','StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
# When working with a normal distribution, the standard deviation works with the particular characteristics of a normal distribution to provide even greater insight. Run the cell below to see the relationship between standard deviations and the data in the normal distribution.
import scipy.stats as stats
# Get the Grade column
col = df_students['Grade']
# get the density
density = stats.gaussian_kde(col)
# Plot the density
col.plot.density()
# Get the mean and standard deviation
s = col.std()
m = col.mean()
# Annotate 1 stdev
x1 = [m-s, m+s]
y1 = density(x1)
plt.plot(x1,y1, color='magenta')
plt.annotate('1 std (68.26%)', (x1[1],y1[1]))
# Annotate 2 stdevs
x2 = [m-(s*2), m+(s*2)]
y2 = density(x2)
plt.plot(x2,y2, color='green')
plt.annotate('2 std (95.45%)', (x2[1],y2[1]))
# Annotate 3 stdevs
x3 = [m-(s*3), m+(s*3)]
y3 = density(x3)
plt.plot(x3,y3, color='orange')
plt.annotate('3 std (99.73%)', (x3[1],y3[1]))
# Show the location of the mean
plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)
plt.axis('off')
"""plt.show()"""

# The descriptive statistics we've used to understand the distribution of the student data variables are the basis of statistical analysis; and because they're such an important part of exploring your data, there's a built-in Describe method of the DataFrame object that returns the main descriptive statistics for all numeric columns.
print(df_students.describe())

# Comparing data
# Now that you know something about the statistical distribution of the data in your dataset, you're ready to examine your data to identify any apparent relationships between variables.
# First of all, let's get rid of any rows that contain outliers so that we have a sample that is representative of a typical class of students. We identified that the StudyHours column contains some outliers with extremely low values, so we'll remove those rows.
df_sample = df_students[df_students['StudyHours']>1]
print(df_sample)

# Comparing numeric and categorical variables
# The data includes two numeric variables (StudyHours and Grade) and two categorical variables (Name and Pass). Let's start by comparing the numeric StudyHours column to the categorical Pass column to see if there's an apparent relationship between the number of hours studied and a passing grade.
# To make this comparison, let's create box plots showing the distribution of StudyHours for each possible Pass value (true and false).
df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))

# Comparing numeric variables
# Now let's compare two numeric variables. We'll start by creating a bar chart that shows both grade and study hours.
# Create a bar plot of name vs grade and study hours
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))


