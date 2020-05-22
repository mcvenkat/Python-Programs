# Set up your imports here!
# import ...
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')

def index():
    # Welcome Page
    return '<h1> This is the generic page for puppies </h1> '# Create a generic welcome page.

@app.route('/puppy/<name>') # Fill this in!

def puppylatin(name):

    pupname = ' '
    if name[ -1] == 'y':
        pupname = name[: -1] + 'iful'
    else:
        pupname = name + 'y'

    return '<h1> Your puppylatin name is :{}'.format(pupname)

if __name__ == '__main__':
    app.run()
