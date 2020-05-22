#Plotting percentage of women who received an engineering degree over years

#importing bokeh and pandas
from bokeh.plotting import figure
from bokeh.io import output_file, show
import pandas as pd

#prepare some data
df = pd.read_csv("http://pythonhow.com/data/bachelors.csv")
x = df["Year"]
#y = df["Engineering"]
y = df["Agriculture"]

#prepare the output file
output_file("Line_from_bachelors.html")

#create a figure object
f = figure()

#create line plot
f.line(x, y)

#write the plot in the figure object
show(f)
