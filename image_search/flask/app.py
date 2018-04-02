import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename

from sim_classified_imgs import search_sim_images


app = Flask(__name__)

@app.route('/upload')
def upload() :
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader() :
    if request.method == 'POST':
        f = request.files['file']
        fn = secure_filename(f.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(fpath)
       
        imgFNs = search_sim_images(fpath)
###        imgFNs = ['1001369_L.jpg', '1006938_L.jpg', '1007180_L.jpg'] 
        return render_template('sim_images.html', uploadFN=fn, localFNs=imgFNs)

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

