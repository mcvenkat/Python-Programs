import turtle

hz = turtle.Turtle()
hz.speed(10)
#hz.shape("cat")
hz.color('green', 'yellow')

cl = ['red', 'green', 'blue']
hz.speed(50)

def drawArt(angle, d, x, y):
  hz.up()
  hz.goto(x, y)
  hz.down()
  c=0
  for i in range(1, 400):
    hz.pencolor(cl[c])
    hz.forward(d)
    hz.left(angle)
    d = d - 2
    c = c + 1
    if(c > 2):
      c = 0

drawArt(200, 98, 0, 0)

drawArt(50, 140, -100, 0)
