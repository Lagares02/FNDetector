from django.urls import path
from . import views

app_name="get_news"

urlpatterns = [
    path('', views.index, name="index"),
    path('get_news/', views.news, name="get_news"),
    path('valid/', views.valid_new, name="valid"),
    path('verificar/', views.verificar, name='verificar'),
]