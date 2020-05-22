#Plotting periodic table elements

#Importing libraries
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.sampledata.periodic_table import elements
from bokeh.models import Range1d, PanTool, ResetTool, HoverTool, ColumnDataSource, LabelSet
import pandas

#Remove rows with NaN values and then map standard states to colors
elements.dropna(inplace=True) #if inplace is not set to True the changes are not written to the dataframe
colormap = {'gas':'green','liquid':'blue','solid':'red'}
elements['color'] = [colormap[x] for x in elements['standard state']]
elements['size'] = elements['van der Waals radius'] / 10

#Create three ColumnDataSources for elements of unique standard states
gas = ColumnDataSource(elements[elements['standard state']=='gas'])
liquid = ColumnDataSource(elements[elements['standard state']=='liquid'])
solid = ColumnDataSource(elements[elements['standard state']=='solid'])

#Define the output file path
output_file("elements_gridplot.html")

#Create the figure object
f1=figure()

#adding glyphs
f1.circle(x="atomic radius", y="boiling point", size='size',
         fill_alpha=0.2, color="color", legend='Gas', source=gas)

f2=figure()
f2.circle(x="atomic radius", y="boiling point", size='size',
         fill_alpha=0.2, color="color", legend='Liquid', source=liquid)

f3=figure()
f3.circle(x="atomic radius",  y="boiling point", size='size',
         fill_alpha=0.2, color="color", legend='Solid', source=solid)

f=gridplot([[f1,f2], [None,f3]])

#Save and show the figure
show(f)
