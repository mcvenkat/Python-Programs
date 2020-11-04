import turtle

vk = turtle.Turtle()
vk.getscreen().bgcolor("#994444")

def star(turtle, size):
    if size <= 10:
        return
    else:
        turtle.begin_fill()
        for i in range(5):
            turtle.forward(size)
            star(vk, size/3)
            turtle.left(216)
        turtle.end_fill()

star(vk, 360)
turtle.done()
