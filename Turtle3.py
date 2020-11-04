import turtle
import random
import math

vk = turtle.Turtle()
vk.color("red", "blue")
vk.speed(10)

for i in range(1, 1000):
  vk.forward(random.randint(int(math.sqrt(i)), int(math.sqrt(i + 99))))
  vk.left(i%90)

turtle.exitonclick()
