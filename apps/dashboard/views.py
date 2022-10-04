
from contextlib import redirect_stderr
from http.client import HTTPResponse
import re
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
import datetime
from plotly import express as px
from django.shortcuts import redirect
from json import dumps

UNOCCUPIED_HOURS_LIST = [23, 24, 0, 1, 2, 3, 4, 5]
DATES_SET_1 = {}
DATES_SET_2 = {}
DATES_SET_3 = {}
DATES_SET_4 = {}

def index(request):
    context = {}
    context['segment'] = 'home.html'
    html_template = loader.get_template('dashboard/home.html')
    return HttpResponse(html_template.render(context, request))

# For CP1 Cooling
def low_delta_t_chiller(CHWS, CHWR, SQ1_CP1_DF_CH1_KW, SQ1_CP1_DF_CH2_KW, SQ1_CP1_DF_CH3_KW, SQ1_CP1_DF_CH4_KW):
    try:
        if(abs(float(CHWS)- float(CHWR)) < 5 and (float(SQ1_CP1_DF_CH1_KW) > 10 or float(SQ1_CP1_DF_CH2_KW) > 10 or float(SQ1_CP1_DF_CH3_KW) > 10 or float(SQ1_CP1_DF_CH4_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def chiller_operating_during_unoccupied_hours_1(time, SQ1_CP1_DF_OAT, SQ1_CP1_DF_CH1_KW):
    try:
        if(int(time.split("\\:")[0]) in UNOCCUPIED_HOURS_LIST and float(SQ1_CP1_DF_OAT) > 60 and (float(SQ1_CP1_DF_CH1_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def chiller_operating_during_unoccupied_hours_2(time, SQ1_CP1_DF_OAT, SQ1_CP1_DF_CH2_KW):
    try:
        if(int(time.split("\\:")[0]) in UNOCCUPIED_HOURS_LIST and float(SQ1_CP1_DF_OAT) > 60 and (float(SQ1_CP1_DF_CH2_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def chiller_operating_during_unoccupied_hours_3(time, SQ1_CP1_DF_OAT, SQ1_CP1_DF_CH3_KW):
    try:
        if(int(time.split("\\:")[0]) in UNOCCUPIED_HOURS_LIST and float(SQ1_CP1_DF_OAT) > 60 and (float(SQ1_CP1_DF_CH3_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def chiller_operating_during_unoccupied_hours_4(time, SQ1_CP1_DF_OAT, SQ1_CP1_DF_CH4_KW):
    try:
        if(int(time.split("\\:")[0]) in UNOCCUPIED_HOURS_LIST and float(SQ1_CP1_DF_OAT) > 60 and (float(SQ1_CP1_DF_CH4_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def individual_chiller_efficiency_1(SQ1_CP1_DF_CH1_KWT, SQ1_CP1_DF_CH1_KW):
    try:
        if(float(SQ1_CP1_DF_CH1_KWT) > 0.6 and (float(SQ1_CP1_DF_CH1_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def individual_chiller_efficiency_2(SQ1_CP1_DF_CH2_KWT, SQ1_CP1_DF_CH2_KW):
    try:
        if(float(SQ1_CP1_DF_CH2_KWT) > 0.6 and (float(SQ1_CP1_DF_CH2_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def individual_chiller_efficiency_3(SQ1_CP1_DF_CH3_KWT, SQ1_CP1_DF_CH3_KW):
    try:
        if(float(SQ1_CP1_DF_CH3_KWT) > 0.6 and (float(SQ1_CP1_DF_CH3_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def individual_chiller_efficiency_4(SQ1_CP1_DF_CH4_KWT, SQ1_CP1_DF_CH4_KW):
    try:
        if(float(SQ1_CP1_DF_CH4_KWT) > 0.6 and (float(SQ1_CP1_DF_CH4_KW) > 10)):
            return True
        return False
    except:
        return np.NaN

def too_many_starts_1(date, SQ1_CP1_DF_CH1_KW):
    try:
        if(date in DATES_SET_1):
            if(float(SQ1_CP1_DF_CH1_KW) < 10):
                DATES_SET_1[date][0] += 1
            else:
                DATES_SET_1[date][1] += 1
        else:
            DATES_SET_1[date] = [0,0]
    except:
        return np.NaN

def too_many_starts_2(date, SQ1_CP1_DF_CH2_KW):
    try:
        if(date in DATES_SET_2):
            if(float(SQ1_CP1_DF_CH2_KW) < 10):
                DATES_SET_2[date][0] += 1
            else:
                DATES_SET_2[date][1] += 1
        else:
            DATES_SET_2[date] = [0,0]
    except:
        return np.NaN

def too_many_starts_3(date, SQ1_CP1_DF_CH3_KW):
    try:
        if(date in DATES_SET_3):
            if(float(SQ1_CP1_DF_CH3_KW) < 10):
                DATES_SET_3[date][0] += 1
            else:
                DATES_SET_3[date][1] += 1
        else:
            DATES_SET_3[date] = [0,0]
    except:
        return np.NaN

def too_many_starts_4(date, SQ1_CP1_DF_CH4_KW):
    try:
        if(date in DATES_SET_4):
            if(float(SQ1_CP1_DF_CH4_KW) < 10):
                DATES_SET_4[date][0] += 1
            else:
                DATES_SET_4[date][1] += 1
        else:
            DATES_SET_4[date] = [0,0]
    except:
        return np.NaN

def too_many_starts(df):
    pass

# For CP2 Cooling
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

def fault_rule_implementation(data_file,mapping_file,filetype):
    df = pd.read_csv(mapping_file, header=None)
    
    column_mappings = {}
    for index, row in df.iterrows():
        row[0] = row[0].split(':')[0]
        column_mappings[row[0]] = row[1]
    if(filetype == "CP1"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_csv(data_file, sep="\t")
    df = df.iloc[:, :-1]
    df = df.rename(columns={'<>Date': 'Date'})
    df = df.rename(columns=column_mappings)

    df = df[:-1]

    df.replace({'OFF': 0, 'ON': 1}, inplace=True)
    print(df.head())
    return df

@csrf_exempt
def create_csv(request):
    if request.method == 'POST':
        data_file = request.FILES["csv_file"]
        mapping_file = request.FILES["Mapping_file"]
        

        df = fault_rule_implementation(data_file, mapping_file,request.POST.get('filetype'))
        
        request.session['df'] = df.to_json(orient="records")

        if(request.POST.get('filetype') == 'CP1'):
            df['low_delta_t_chiller'] = df.apply(
                lambda row: low_delta_t_chiller(row['1CHWS'], row['1CHWR'], row['SQ1_CP1_DF_CH1_KW'], row['SQ1_CP1_DF_CH2_KW'], row['SQ1_CP1_DF_CH3_KW'], row['SQ1_CP1_DF_CH4_KW']), axis=1)

            df['chiller_operating_during_unoccupied_hours_1'] = df.apply(
                lambda row: chiller_operating_during_unoccupied_hours_1(row['Time'], row['SQ1_CP1_DF_OAT'], row['SQ1_CP1_DF_CH1_KW']), axis=1)
            df['chiller_operating_during_unoccupied_hours_2'] = df.apply(
                lambda row: chiller_operating_during_unoccupied_hours_2(row['Time'], row['SQ1_CP1_DF_OAT'], row['SQ1_CP1_DF_CH2_KW']), axis=1)
            df['chiller_operating_during_unoccupied_hours_3'] = df.apply(
                lambda row: chiller_operating_during_unoccupied_hours_3(row['Time'], row['SQ1_CP1_DF_OAT'], row['SQ1_CP1_DF_CH3_KW']), axis=1)
            df['chiller_operating_during_unoccupied_hours_4'] = df.apply(
                lambda row: chiller_operating_during_unoccupied_hours_4(row['Time'], row['SQ1_CP1_DF_OAT'], row['SQ1_CP1_DF_CH4_KW']), axis=1)

            df['individual_chiller_efficiency_1'] = df.apply(
                lambda row: individual_chiller_efficiency_1(row['SQ1_CP1_DF_CH1_KWT'], row['SQ1_CP1_DF_CH1_KW']), axis=1)
            df['individual_chiller_efficiency_2'] = df.apply(
                lambda row: individual_chiller_efficiency_2(row['SQ1_CP1_DF_CH2_KWT'], row['SQ1_CP1_DF_CH2_KW']), axis=1)
            df['individual_chiller_efficiency_3'] = df.apply(
                lambda row: individual_chiller_efficiency_3(row['SQ1_CP1_DF_CH3_KWT'], row['SQ1_CP1_DF_CH3_KW']), axis=1)
            df['individual_chiller_efficiency_4'] = df.apply(
                lambda row: individual_chiller_efficiency_4(row['SQ1_CP1_DF_CH4_KWT'], row['SQ1_CP1_DF_CH4_KW']), axis=1)

            df.apply(lambda row: too_many_starts_1(str(row['Date']), row['SQ1_CP1_DF_CH1_KW']), axis=1)
            df.apply(lambda row: too_many_starts_2(str(row['Date']), row['SQ1_CP1_DF_CH2_KW']), axis=1)
            df.apply(lambda row: too_many_starts_3(str(row['Date']), row['SQ1_CP1_DF_CH3_KW']), axis=1)
            df.apply(lambda row: too_many_starts_4(str(row['Date']), row['SQ1_CP1_DF_CH4_KW']), axis=1)
            too_many_starts()

            # TODO: Dropdown to select which faults to choose
            # result = df.to_html(index=False)
            df['DateTime'] = df[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)

            filter_col = ['DateTime',
                          'low_delta_t_chiller',
                          'chiller_operating_during_unoccupied_hours_1', 'chiller_operating_during_unoccupied_hours_2', 'chiller_operating_during_unoccupied_hours_3', 'chiller_operating_during_unoccupied_hours_4',
                          'individual_chiller_efficiency_1', 'individual_chiller_efficiency_2', 'individual_chiller_efficiency_3', 'individual_chiller_efficiency_4']

            df = df[filter_col].T
            new_header = df.iloc[0]
            df = df[1:]
            df.columns = new_header

            filter_col_name = ['Low Delta T Chiller',
                          'Chiller Operating During Unoccupied Hours 1', 'Chiller Operating During Unoccupied Hours 2', 'Chiller Operating During Unoccupied Hours 3', 'Chiller Operating During Unoccupied Hours 4',
                          'Individual Chiller Efficiency 1', 'Individual Chiller Efficiency 2', 'Individual Chiller Efficiency 3', 'Individual Chiller Efficiency 4']

            df.insert(loc=0, column='Fault Rules', value=filter_col_name)

            request.session['result'] = df.to_json(orient="records")
            html_template = loader.get_template('Dashboard/Index.html')
            return HttpResponse(html_template.render({"dataframe": df}, request))

        elif(request.POST.get('filetype') == 'CP2'):
           
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
                   

@csrf_exempt
def download(request):
    df = pd.DataFrame()
    if request.session.get('result'):
        result =  request.session.get('result')
        df = pd.DataFrame(json.loads(result))
        
        response = HttpResponse(df.to_csv(index=False),content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename="sensor_data_results.csv"'
        return response

@csrf_exempt
def filterdata(request):
    starttime = pd.to_datetime(request.POST.get('starttime'))
    endtime = pd.to_datetime(request.POST.get('endtime'))
    print(pd.to_datetime(starttime))
    
   
    print(endtime)
    if request.session.get('result'):
        result =  request.session.get('result')
        df = pd.DataFrame(json.loads(result))
        # startdate, starttime = starttime.split('T')
        # enddate, endtime = endtime.split('T')

        # start = startdate + " " + starttime
        # end = enddate + " " + endtime
        df = df.T
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        df['datetime'] = df.index
        df['datetime']= pd.to_datetime(df["datetime"])
        df = df[(df['datetime'] > starttime) &  (df['datetime'] < endtime)]

        print(df)
    
    return "successs"

@csrf_exempt
def charts(request):
    df = pd.DataFrame()
    
    if request.session.get('df'):
            result =  request.session.get('df')
            df = pd.DataFrame(json.loads(result))
            headers = list(df.columns)
            headers.remove('Date')
            headers.remove('Time')
            
            
            
    if request.method == "POST":
        # data_file = request.FILES["csv_file"]
        # mapping_file = request.FILES["Mapping_file"]
        # print("refresh")
        # df = fault_rule_implementation(data_file, mapping_file,request.POST.get('filetype'))
        df['DateTime'] = df[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
        
        xaxis = 'DateTime'
        yaxis = request.POST.getlist('fields')
        if yaxis == '':
            yaxis = headers[0]
        #df = px.data.gapminder().query("period in )
        type = request.POST.get("type")
        if type == 'bar':
            print(xaxis)
            fig = px.bar(df, x=xaxis, y=yaxis,title="")
        elif type == 'scatter':
            fig = px.scatter(df, x=xaxis, y=yaxis,labels={"variable": "Components"},title='')
        else:
            fig = px.line(df, x=xaxis, y=yaxis, labels={"variable": "Components"}, title='')

        graph = fig.to_html(full_html=False, default_height="450px", default_width="100%")

        dataDictionary = {
        'yaxis': yaxis,
        'type': type
        }

        dataJSON = dumps(dataDictionary)
        
        html_template = loader.get_template('Dashboard/charts.html')
        return HttpResponse(html_template.render({"headers":headers, "graph":graph, "data":dataJSON}, request))
    else:
        print('getcall')
        html_template = loader.get_template('Dashboard/charts.html')
        return HttpResponse(html_template.render({"headers":headers}, request)) 
    
    



          