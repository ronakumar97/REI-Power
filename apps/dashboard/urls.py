

from django.urls import path, re_path
from apps.dashboard import views

urlpatterns = [ 
    # The home page
    path('home/', views.index, name='home'),
    path('', views.index, name='home'),
    path("create_csv/", views.create_csv, name='csv'),
    path("create_csv/<str:filetype>", views.create_csv, name='csv'),
    path("download/", views.download, name= 'download'),
    path("filterdata/", views.filterdata, name= 'filterdata'),
    path("charts/<str:file>", views.charts, name= 'charts'),
    path("downloadcsv/", views.downloadcsv, name= 'downloadcsv'),
    
]
