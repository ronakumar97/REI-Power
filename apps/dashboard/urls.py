

from django.urls import path, re_path
from apps.dashboard import views

urlpatterns = [

    # The home page
    path('', views.index, name='dashboard'),
    path("create_csv/", views.create_csv, name='csv'),
 
]
