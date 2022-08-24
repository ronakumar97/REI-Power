from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('csv/', views.create_csv, name='create_csv')
]
