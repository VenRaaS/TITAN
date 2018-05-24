# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
import os
import urlparse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser 
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings
from . import settings as app_settings
from wsgiref.util import FileWrapper

from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .serializers import FileSerializer

from rest_framework.renderers import TemplateHTMLRenderer
from sim_imgs_flat_model import search_sim_images


def simple_upload(request):
    if request.method == 'POST' and request.FILES['file']:
        f = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(f.name, f)

        uploaded_file_url = fs.url(filename)
        uploaded_file_path = fs.path(filename)
        print uploaded_file_url
   
        imgFNs = search_sim_images(uploaded_file_path, 
                                    (app_settings.imgFea1Ds, app_settings.imgBNs),
                                    app_settings.cnn_model)
        print app_settings.FEATURE_IMAGE_URL
        recImgUrls = map(lambda fn: urlparse.urljoin(app_settings.FEATURE_IMAGE_URL, fn), imgFNs)
        print recImgUrls
         
        uploaded_img = { 'url': uploaded_file_url } 
        return render(request, 'upload_app/recomd.html', {'uploaded_img': uploaded_img, 'recImgUrls': recImgUrls})

    return render(request, 'upload_app/upload_form.html')


class FileView(APIView):
    parser_classes = (FileUploadParser,)

    def post(self, request, filename, format='jpg'):
        up_f = request.data['file']
    
        fpath = os.path.join(settings.MEDIA_ROOT, filename)
        with open(fpath, 'wb+') as dest:
            for chunk in up_f:
                dest.write(chunk)
        
        with open(fpath, 'rb') as src:
            resp = HttpResponse(FileWrapper(src), content_type="image/jpeg")
        
        resp['Content-Disposition'] = 'attachment; filepath={}'.format(os.path.join(settings.MEDIA_URL, filename))
        return resp


class Recomd(APIView):
    parser_classes = (FileUploadParser,)

    def post(self, request, filename, format='jpg'):
        up_f = request.data['file']
    
        fpath = os.path.join(settings.MEDIA_ROOT, filename)
        with open(fpath, 'wb+') as dest:
            for chunk in up_f:
                dest.write(chunk)

        img = { 'url': os.path.join(settings.MEDIA_URL, filename) } 

        return render(request, 'upload_app/recomd.html', {'img': img})
