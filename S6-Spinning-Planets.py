#Orbiting planets

#importing libraries
from math import cos, sin, radians
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

#Setting up the figure
p = figure(x_range=(-3, 3), y_range=(-3, 3))

#Drawing static glyphs
mercury_orbit=p.circle(x=0, y=0, radius=0.3, line_color='violet',line_alpha=0.5,fill_color=None, line_width=2)
venus_orbit=p.circle(x=0, y=0, radius=0.5, line_color='orange',line_alpha=0.5,fill_color=None, line_width=2)
earth_orbit=p.circle(x=0, y=0, radius=0.5, line_color='blue',line_alpha=0.5,fill_color=None, line_width=2)
mars_orbit=p.circle(x=0, y=0, radius=0.7, line_color='red',line_alpha=0.5,fill_color=None, line_width=2)
jupiter_orbit=p.circle(x=0, y=0, radius=1.4, line_color='green',line_alpha=0.5,fill_color=None, line_width=2)
saturn_orbit=p.circle(x=0, y=0, radius=1.0, line_color='gold',line_alpha=0.5,fill_color=None, line_width=2)
uranus_orbit=p.circle(x=0, y=0, radius=0.8, line_color='purple',line_alpha=0.5,fill_color=None, line_width=2)
neptune_orbit=p.circle(x=0, y=0, radius=1.2, line_color='turquoise',line_alpha=0.5,fill_color=None, line_width=2)
sun=p.circle(x=0, y=0, radius=0.2, line_color=None, fill_color="yellow", fill_alpha=0.5)

#Creating columndatasources for the moving circles
mercury_source = ColumnDataSource(data=dict(x_mercury=[mercury_orbit.glyph.radius*cos(radians(0))], y_mercury=[mercury_orbit.glyph.radius*sin(radians(0))]))
venus_source = ColumnDataSource(data=dict(x_venus=[venus_orbit.glyph.radius*cos(radians(0))], y_venus=[venus_orbit.glyph.radius*sin(radians(0))]))
earth_source = ColumnDataSource(data=dict(x_earth=[earth_orbit.glyph.radius*cos(radians(0))], y_earth=[earth_orbit.glyph.radius*sin(radians(0))]))
mars_source = ColumnDataSource(data=dict(x_mars=[mars_orbit.glyph.radius*cos(radians(0))], y_mars=[mars_orbit.glyph.radius*sin(radians(0))]))
jupiter_source = ColumnDataSource(data=dict(x_jupiter=[jupiter_orbit.glyph.radius*cos(radians(0))], y_jupiter=[jupiter_orbit.glyph.radius*sin(radians(0))]))
saturn_source = ColumnDataSource(data=dict(x_saturn=[saturn_orbit.glyph.radius*cos(radians(0))], y_saturn=[saturn_orbit.glyph.radius*sin(radians(0))]))
uranus_source = ColumnDataSource(data=dict(x_uranus=[uranus_orbit.glyph.radius*cos(radians(0))], y_uranus=[uranus_orbit.glyph.radius*sin(radians(0))]))
neptune_source = ColumnDataSource(data=dict(x_neptune=[neptune_orbit.glyph.radius*cos(radians(0))], y_neptune=[neptune_orbit.glyph.radius*sin(radians(0))]))



#Drawing the moving glyphs
mercury=p.circle(x='x_mercury', y='y_mercury', size=9, fill_color='violet', line_color=None, fill_alpha=0.6, source=mercury_source)
venus=p.circle(x='x_venus', y='y_venus', size=10, fill_color='orange', line_color=None, fill_alpha=0.6, source=venus_source)
earth=p.circle(x='x_earth', y='y_earth', size=12, fill_color='blue', line_color=None, fill_alpha=0.6, source=earth_source)
mars=p.circle(x='x_mars', y='y_mars', size=9, fill_color='red', line_color=None, fill_alpha=0.6, source=mars_source)
jupiter=p.circle(x='x_jupiter', y='y_jupiter', size=20, fill_color='green', line_color=None, fill_alpha=0.6, source=jupiter_source)
saturn=p.circle(x='x_saturn', y='y_saturn', size=18, fill_color='gold', line_color=None, fill_alpha=0.6, source=saturn_source)
uranus=p.circle(x='x_uranus', y='y_uranus', size=15, fill_color='purple', line_color=None, fill_alpha=0.6, source=uranus_source)
neptune=p.circle(x='x_jupiter', y='y_jupiter', size=16, fill_color='turquoise', line_color=None, fill_alpha=0.6, source=neptune_source)

#we  will generate x and y coordinates every 0.1 seconds out of angles starting from an angle of 0 for both earth and mars
i_earth=0
i_mars=0
i_mercury=0
i_venus=0
i_jupiter=0
i_saturn=0
i_uranus=0
i_neptune=0

#the update function will genjupitere coordinates
def update():
    global i_earth,i_mars,i_mercury,i_venus,i_jupiter,i_saturn, i_uranus,i_neptune #this tells the function to use global variables declared outside the function
    i_earth=i_earth + 2 #we will increase the angle of earth by 2 in function call
    i_mars=i_mars + 1
    i_mercury=i_mercury + 3
    i_venus=i_venus + 2
    i_jupiter=i_jupiter + 1
    i_saturn= i_saturn + 1
    i_uranus=i_uranus + 1
    i_neptune=i_neptune + 1
    new_mercury_data = dict(x_mercury=[mercury_orbit.glyph.radius*cos(radians(i_mercury))],y_mercury=[mercury_orbit.glyph.radius*sin(radians(i_mercury))])
    new_venus_data = dict(x_venus=[venus_orbit.glyph.radius*cos(radians(i_venus))],y_venus=[venus_orbit.glyph.radius*sin(radians(i_venus))])
    new_earth_data = dict(x_earth=[earth_orbit.glyph.radius*cos(radians(i_earth))],y_earth=[earth_orbit.glyph.radius*sin(radians(i_earth))])
    new_mars_data = dict(x_mars=[mars_orbit.glyph.radius*cos(radians(i_mars))],y_mars=[mars_orbit.glyph.radius*sin(radians(i_mars))])
    new_jupiter_data = dict(x_jupiter=[jupiter_orbit.glyph.radius*cos(radians(i_jupiter))],y_jupiter=[jupiter_orbit.glyph.radius*sin(radians(i_jupiter))])
    new_saturn_data = dict(x_saturn=[saturn_orbit.glyph.radius*cos(radians(i_saturn))],y_saturn=[saturn_orbit.glyph.radius*sin(radians(i_saturn))])
    new_uranus_data = dict(x_uranus=[uranus_orbit.glyph.radius*cos(radians(i_uranus))],y_uranus=[uranus_orbit.glyph.radius*sin(radians(i_uranus))])
    new_neptune_data = dict(x_neptune=[neptune_orbit.glyph.radius*cos(radians(i_neptune))],y_neptune=[neptune_orbit.glyph.radius*sin(radians(i_neptune))])
    mercury_source.stream(new_mercury_data,rollover=1)
    venus_source.stream(new_venus_data,rollover=1)
    earth_source.stream(new_earth_data,rollover=1)
    mars_source.stream(new_mars_data,rollover=1)
    jupiter_source.stream(new_jupiter_data,rollover=1)
    saturn_source.stream(new_saturn_data,rollover=1)
    uranus_source.stream(new_uranus_data,rollover=1)
    neptune_source.stream(new_neptune_data,rollover=1)

    #just printing the data in the terminal
    print(mercury_source.data)
    print(venus_source.data)
    print(earth_source.data)
    print(mars_source.data)
    print(jupiter_source.data)
    print(saturn_source.data)
    print(uranus_source.data)
    print(neptune_source.data)


#adding periodic callback and the plot to curdoc
curdoc().add_periodic_callback(update, 100)
curdoc().add_root(p)
