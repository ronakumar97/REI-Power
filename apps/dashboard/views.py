
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
import numpy as np


def index(request):
    context = {}
    context['segment'] = 'dashboard.html'
    html_template = loader.get_template('home/dashboard.html')
    return HttpResponse(html_template.render(context, request))

def is_free_cooling_operation(CP2CHOAT, CP2CH1M5, CP2CH2M10):
    try:
        if ((float(CP2CHOAT) < 60) and (CP2CH1M5 or CP2CH2M10)):
            return True
        return False
    except:
        return np.NaN

def chiller_water_temp_diff(CCHWST, CCHWRT, CP2CH1M5, CP2CH2M10):
    try:
        if(abs(float(CCHWST)- float(CCHWRT)) < 5 and (CP2CH1M5 or CP2CH2M10)):
            return True
        return False
    except:
        return np.NaN
def condensor_water_temp_diff(CDWST, CDWRT, CP2CH1M5, CP2CH2M10):
    try:
        if(abs(float(CDWST) - float(CDWRT)) < 5 and (CP2CH1M5 or CP2CH2M10)):
            return True
        return False
    except:
        return np.NaN
def condensor_water_reset_temp(CDWRT, CP2CHOAT):
    try:
        if(float(CDWRT) > (float(CP2CHOAT) + 7)):
            return True
        return False
    except:
        return np.NaN

def fault_rule_implementation(data_file,mapping_file):
    df = pd.read_csv(mapping_file, header=None)
    column_mappings = {}
    for index, row in df.iterrows():
        row[0] = row[0].split(':')[0]
        column_mappings[row[0]] = row[1]

    df = pd.read_csv(data_file, sep="\t")
    df = df.iloc[:, :-1]
    df = df.rename(columns={'<>Date': 'Date'})
    df = df.rename(columns=column_mappings)

    df = df[:-1]

    df.replace({'OFF': 0, 'ON': 1}, inplace=True)

    return df

@csrf_exempt
def create_csv(request):
    
    if request.method == 'POST':

        # TODO: Make two buttons (one for uploading the column mapping and other for uploading the data in the CSV)
       
        data_file = request.FILES["csv_file"]
        mapping_file = request.FILES["Mapping_file"]
            
        df = fault_rule_implementation(data_file,mapping_file)

        df['is_free_cooling_operation_results'] = df.apply(
            lambda row: is_free_cooling_operation(row['CP2.CHOAT'], row['CP2.CH1.M5'], row['CP2.CH2.M10']), axis=1)
        df['chiller_water_temp_diff_results'] = df.apply(
            lambda row: chiller_water_temp_diff(row['CCHWST'], row['CCHWRT'], row['CP2.CH1.M5'], row['CP2.CH2.M10']),
            axis=1)
        df['condensor_water_temp_diff_results'] = df.apply(
            lambda row: condensor_water_temp_diff(row['CDWST'], row['CDWRT'], row['CP2.CH1.M5'], row['CP2.CH2.M10']),
            axis=1)
        df['condensor_water_reset_temp_results'] = df.apply(
            lambda row: condensor_water_reset_temp(row['CDWRT'], row['CP2.CHOAT']), axis=1)

        # TODO: Dropdown to select which faults to choose
         # result = df.to_html(index=False)
        df['DateTime'] = df[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    
        filter_col=['DateTime','is_free_cooling_operation_results','chiller_water_temp_diff_results','condensor_water_temp_diff_results','condensor_water_reset_temp_results']
        df = df[filter_col].T
        new_header = df.iloc[0] 
        df = df[1:]
        df.columns = new_header 

        filter_col_name=['Free cooling operation results','Chiller water temp diff results','Condensor water temp diff results','Condensor water reset temp results']
        df.insert(loc=0, column='Fault Rules', value=filter_col_name)
       
        request.session['result'] = df.to_json(orient="records")
        html_template = loader.get_template('Dashboard/Index.html')
        return HttpResponse(html_template.render({"dataframe":df}, request))
                   
    else:
        df = pd.DataFrame()
        if request.session.get('result'):
            result =  request.session.get('result')
            df = pd.DataFrame(json.loads(result))
            # print(df)
        response = HttpResponse(df.to_csv(index=False),content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="sensor_data_results.csv"'
        return response


          