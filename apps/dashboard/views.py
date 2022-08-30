
import pandas as pd
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.shortcuts import render
import json
#from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt
import csv


def index(request):
    context = {}
    context['segment'] = 'dashboard.html'
    html_template = loader.get_template('home/dashboard.html')
    return HttpResponse(html_template.render(context, request))


@csrf_exempt
def create_csv(request):
    
    if request.method == 'POST':
        csv_file = request.FILES["csv_file"]
        df = pd.read_csv(csv_file)
        # df = pd.read_csv('C:/Users/lifec/Desktop/sensor_data.csv')
        df = df.reset_index(drop=True)

        result = []

        for index, row in df.iterrows():
            if (row['A'] == row['B']):
                result.append('T')
            else:
                result.append('F')

        df['Result'] = result
        result = df.to_html(index=False)
        print(df)
        request.session['result'] = df.to_json(orient="records")
        html_template = loader.get_template('home/dashboard.html')
        return HttpResponse(html_template.render({"html":result, "dataframe":df}, request))
                   
    else:
        df = pd.DataFrame()
        if request.session.get('result'):
            result =  request.session.get('result')
            df = pd.DataFrame(json.loads(result))
            print(df)
        response = HttpResponse(df.to_csv(index=False),content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="sensor_data_results.csv"'
        return response
          