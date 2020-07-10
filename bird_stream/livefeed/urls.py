# -*- coding: utf-8 -*-
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.livefe, name='livefeed-home'),
]

