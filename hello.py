from flask import Flask

app = Flask(__name__)

@app.route('/home')
def home():
    return " Welcome to Sentiment Analysis <h1>Test Tweet <h1>"

if __name__== '__main__':
    app.run()