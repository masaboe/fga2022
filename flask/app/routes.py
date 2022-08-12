from flask import render_template
from app import app


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Abu Salam'}
    posts = [
        {
            'author': {'username': 'Jono'},
            'body': 'Hari yang cerah di Semarang!'
        },
        {
            'author': {'username': 'Susi'},
            'body': 'Enaknya minum Es Kopi Gula Aren!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)
