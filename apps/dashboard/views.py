
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

def is_free_cooling_operation(df):
    is_free_cooling_operation_results = []

    for index, row in df.iterrows():
        try:
            if((float(row['CP2.CHOAT']) > 60) and (row['CP2.CH1.M5'] or row['CP2.CH2.M10'])):
                is_free_cooling_operation_results.append(1)
            else:
                is_free_cooling_operation_results.append(0)
        except:
            is_free_cooling_operation_results.append(-1)
            continue

    results_df = df[['CP2.CHOAT', 'CP2.CH1.M5', 'CP2.CH2.M10']]
    results_df['is_free_cooling_operation_results'] = is_free_cooling_operation_results

    return results_df

def chiller_water_temp_diff(df):
    chiller_water_temp_diff_results = []

    for index, row in df.iterrows():
        try:
            if (abs(row['CDWST'], row['CDWRT']) < 5 and (row['CP2CH1M5'] or row['CP2CH2M10'])):
                chiller_water_temp_diff_results.append(1)
            else:
                chiller_water_temp_diff_results.append(0)
        except:
            chiller_water_temp_diff_results.append(-1)
            continue

    results_df = df[['CDWST', 'CDWRT', 'CP2CH1M5', 'CP2.CH2.M10']]
    results_df['chiller_water_temp_diff_results'] = chiller_water_temp_diff_results

    return results_df
def condensor_water_temp_diff(df):
    condensor_water_temp_diff_results = []

    for index, row in df.iterrows():
        try:
            if(abs(row['CCHWST'], row['CCHWRT']) < 5 and (row['CP2CH1M5'] or row['CP2CH2M10'])):
                condensor_water_temp_diff_results.append(1)
            else:
                condensor_water_temp_diff_results.append(0)
        except:
            condensor_water_temp_diff_results.append(-1)
            continue

    results_df = df[['CCHWST', 'CCHWRT', 'CP2CH1M5', 'CP2.CH2.M10']]
    results_df['condensor_water_temp_diff_results'] = condensor_water_temp_diff_results

    return results_df

def condensor_water_reset_temp(df):
    condensor_water_return_temp_results = []

    for index, row in df.iterrows():
        try:
            if(row['CDWRT'] > (row['CP2.CHOAT'] + 7)):
                condensor_water_return_temp_results.append(1)
            else:
                condensor_water_return_temp_results.append(0)
        except:
            condensor_water_return_temp_results.append(-1)
            continue

    results_df = df[['CDWRT', 'CP2.CHOAT']]
    results_df['condensor_water_return_temp_results'] = condensor_water_return_temp_results

    return results_df

def fault_rule_implementation():
    df = pd.read_csv('/home/ubuntu/PycharmProjects/rei_power/column_mapping.csv', header=None)
    column_mappings = {}
    for index, row in df.iterrows():
        row[0] = row[0].split(':')[0]
        column_mappings[row[0]] = row[1]

    df = pd.read_csv('/home/ubuntu/PycharmProjects/rei_power/CP2 COOLING_data.csv', sep="\t")
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
        csv_file = request.FILES["csv_file"]
        df = pd.read_csv(csv_file)
        df = df.reset_index(drop=True)

        df = fault_rule_implementation()

        is_free_cooling_operation_results_df = is_free_cooling_operation(df)
        chiller_water_temp_diff_results_df = chiller_water_temp_diff(df)
        condensor_water_temp_diff_results_df = chiller_water_temp_diff(df)
        condensor_water_reset_temp_results_df = condensor_water_reset_temp(df)

        # TODO: Dropdown to select which faults to choose

        result = is_free_cooling_operation_results_df.to_html(index=False)
        # result = chiller_water_temp_diff.to_html(index=False)
        # result = condensor_water_temp_diff_results_df.to_html(index=False)
        # result = condensor_water_reset_temp_results_df.to_html(index=False)

        print(df)
        request.session['result'] = is_free_cooling_operation_results_df.to_json(orient="records")
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
          