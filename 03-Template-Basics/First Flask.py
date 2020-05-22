from flask import Flask, render_template
app = Flask(__name__)
@app.route('/') #127.0.0.1:5000

def index():
    return render_template('first flask.html')
#   return  '<h1>Hello VK!</h1>'

#@app.route('/information') #127.0.0.1:5000/information

#def info():
#    return '<h1> Vk is learning Flask </h1>'

#@app.route('/vk/<name>') #127.0.0.1:5000/vk/info

#def func(name):
#    return '<h1> This is the first dynamic routed page for {}</h1>'.format(name)



if __name__ == '__main__':
  app.run()
