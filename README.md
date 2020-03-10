## TITAN
Typing-free Image and Tag Augmented Navigation

## Environment
### [Miniconda](https://conda.io/miniconda.html)
We use Miniconda to handle the compatibility between the python packages and make the virtual pyhton env.
* [download](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
* [install](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#)
* [getting started with conda](https://conda.io/docs/user-guide/getting-started.html#)

### Install packages
```
conda install django==1.11.10
conda install -c conda-forge djangorestframework
conda install keras
conda install numpy
conda install opencv
conda install scikit-image
conda install elasticsearch==5.4
conda install elasticsearch-dsl==5.3
```

### [Install keras-gpu](https://anaconda.org/anaconda/keras-gpu)
* able to run keras on GPU
```
conda install keras-gpu
```
* verify
```
import keras
keras.backend.tensorflow_backend._get_available_gpus()
```

## Start services
There are 2 services need to start:
1. Django - the web server for image search 
2. Elasticsearch - the search engine for tag search

```
# Image search
sudo su -l titan 
cd TITAN/image_search/fileupload/
nohup python manage.py runserver 0.0.0.0:8000 &

# tag navigation
sudo su -l elk
cd elk/elasticsearch/
./bin/elasticsearch -d
```

### 



