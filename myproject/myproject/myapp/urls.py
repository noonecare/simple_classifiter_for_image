# -*- coding: utf-8 -*-
from django.conf.urls import url
from myproject.myapp.views import list, classify

urlpatterns = [
    url(r'^list/$', list, name='list'),
    url(r"^classify/$", classify, name='classify')
]



