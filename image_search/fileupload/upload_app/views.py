# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
import os
import urlparse
import uuid
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser 
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings
from . import settings as app_settings
from wsgiref.util import FileWrapper

from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .serializers import FileSerializer

from rest_framework.renderers import TemplateHTMLRenderer
from sim_imgs_flat_model import search_sim_images, rotate_image_basedon_exif 
from util import query_by_gids


def simple_upload(request):
    if request.method == 'POST' and request.FILES['file']:
        f = request.FILES['file']
        fs = FileSystemStorage()
        localFN = fs.save(f.name, f)

        uploaded_file_url = fs.url(localFN)
        uploaded_file_path = fs.path(localFN)
        print uploaded_file_url
   
        rotate_image_basedon_exif(uploaded_file_path)
        imgFNs = search_sim_images(uploaded_file_path, 
                                    (app_settings.imgFea1Ds_list, app_settings.normImgFea1Ds_list, app_settings.imgBNs_list),
                                    app_settings.cnn_model)
        print app_settings.FEATURE_IMAGE_URL
        recImgUrls = map(lambda fn: urlparse.urljoin(app_settings.FEATURE_IMAGE_URL, fn), imgFNs)
        print recImgUrls
         
        uploaded_img = { 'url': uploaded_file_url } 
        return render(request, 'upload_app/recomd.html', {'uploaded_img': uploaded_img, 'recImgUrls': recImgUrls})

    return render(request, 'upload_app/upload_form.html')


class Recomd(APIView):
#    parser_classes = (FileUploadParser,)

    def post(self, request, filename, format='jpg'):
#        up_f = request.data['file']
        up_f = request.FILES['file']

        fs = FileSystemStorage()
        localFN = fs.save(filename, up_f)

        uploaded_file_url = fs.url(localFN)
        print uploaded_file_url
        uploaded_file_path = fs.path(localFN)
        print uploaded_file_path

        rotate_image_basedon_exif(uploaded_file_path)
        imgFNs = search_sim_images(uploaded_file_path, 
                                    (app_settings.imgFea1Ds_list, app_settings.normImgFea1Ds_list, app_settings.imgBNs_list),
                                    app_settings.cnn_model)
        print imgFNs

        gids = [ fn.split('_')[0] for fn in imgFNs ]
        gid2props = query_by_gids(gids)
        print gid2props.keys()
        
        recomd_list = []
        for fn in imgFNs:
            gid = fn.split('_')[0]
            recomd_list.append(
                {
                    'id': gid,
                    'name': gid2props[gid]['name'] if gid in gid2props else '',
                    'sale_price': gid2props[gid]['sale_price'] if gid in gid2props else '',
                    'goods_img_url': urlparse.urljoin(app_settings.FEATURE_IMAGE_URL, fn),
                    'goods_page_url': 'https://www.momoshop.com.tw/goods/GoodsDetail.jsp?i_code={}'.format(gid)
                })
        
        recomd_id = str(uuid.uuid4()).split('-')[-1]

        resp = JsonResponse(
                {
                    'recomd_id': recomd_id,
                    'uploaded_file_url': uploaded_file_url,
                    'recomd_list': recomd_list
                })

        return resp


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


