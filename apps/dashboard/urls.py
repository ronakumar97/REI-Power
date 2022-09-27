

from django.urls import path, re_path
from apps.dashboard import views

urlpatterns = [ 
    # The home page
    path('dashboard/', views.index, name='dashboard'),
    path('', views.index, name='dashboard'),
    path("create_csv/", views.create_csv, name='csv'),
    path("download/", views.download, name= 'download'),
]
