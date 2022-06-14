from flask import Flask, render_template, redirect, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emoji/', methods=['POST'])
def emoji():
    if request.method != 'POST':
        return redirect('/')
    # код генерации эмодзи
    # эмодзи нужно сохранить в папку static
    filename = 'emoji.webp'
    return render_template('emoji.html', filename=filename)

app.run(debug=True)