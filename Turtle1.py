import turtle

loadWindow = turtle.Screen()
turtle.colormode(255)

turtle.speed(0)

sides = 5

def shape(size, sides):
    for i in range(sides):
        turtle.forward(size)
        turtle.left(360 / sides)
'''
for i in range(100):
  turtle.circle(3*i)
  turtle.circle(-3*i)
  turtle.left(i)
  turtle.color(5*i, 5*i, 5*i)
'''

for i in range(100):
    shape(5 * i, sides)
    turtle.left(i)

    
turtle.exitonclick()
