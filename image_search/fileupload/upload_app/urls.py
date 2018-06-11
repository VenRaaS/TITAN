from django.conf.urls import url
from django.views.static import serve
from django.conf import settings

from .views import FileView, Recomd, simple_upload


urlpatterns = [
    url(r'^upload/$', simple_upload),
    url(r'^api/image/rank/(?P<filename>[^/]+)$', Recomd.as_view()),
    url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT})
]
