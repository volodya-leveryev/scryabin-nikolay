from os.path import join
from shutil import move
from urllib.parse import quote, unquote

from flask import Flask, render_template, redirect, request

from generate2 import IMG_EXTENSIONS, main

app = Flask(__name__)


@app.route('/')
def index():
    error = unquote(request.args.get('error', ''))
    return render_template('index.html', error=error)


@app.route('/emoji/', methods=['POST'])
def emoji():
    # Извлекаем первый файл
    file = next(iter(request.files.values()))
    filename = file.filename.lower()
    for ext in IMG_EXTENSIONS:
        if file.filename.endswith(ext):
            src = join('data', 'GG', filename)
            dst = join('experiments', 'name', 'results', 'pred', filename)
            break
    else:
        error = quote('Invalid file format')
        return redirect(f'/?error={error}')

    # Сохраняем куда нужно
    file.save(src)
    
    # Запускаем генерацию
    main()
    
    # Перемещаем исходный файл и результат
    move(src, join('static', 'inp_' + filename))
    move(dst, join('static', 'out_' + filename))
    return render_template('emoji.html', filename=filename)


app.run(debug=True)
