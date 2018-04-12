import os
from json import load
from urllib2 import urlopen

from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename

from sim_imgs_flat_model import search_sim_images
import i2iList


app = Flask(__name__)

DIR_FEAVCT_MODEL = 'FeaVCT_cc21000_flat'


@app.route('/upload')
def upload() :
    pubIP = load(urlopen('https://api.ipify.org/?format=json'))['ip']
    print pubIP
    return render_template('upload.html', pubIP=pubIP)

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader() :
    global DIR_FEAVCT_MODEL

    if request.method == 'POST':
        f = request.files['file']
        fn = secure_filename(f.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(fpath)
       
        imgFNs = search_sim_images(DIR_FEAVCT_MODEL, fpath)
        return render_template('sim_images.html', uploadFN=fn, localFNs=imgFNs)

@app.route('/i2i/<filename>')
def i2i(filename) :
#--DEBUG        imgFNs = ['1001369_L.jpg', '1006938_L.jpg', '1007180_L.jpg'] 
    inBasename = os.path.splitext(filename)[0]
    imgFNs = i2iList.i2iList(inBasename, 'FeaVCT_cc21000_flat/image_i2i.tsv')
    imgFNs = [ bn + '.jpg' for bn in imgFNs ]
    return render_template('i2i.html', curFN=filename, localFNs=imgFNs)

@app.route('/sfd_uploads/<filename>')
def send_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sfd_local/<filename>')
def send_local_file(filename):
    return send_from_directory('image2100000000', filename)

@app.route('/')
def hello_world() :
    return 'Hello Flask'


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.debug = True
    app.run(host='0.0.0.0')

