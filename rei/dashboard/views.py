from django.shortcuts import render
import pandas as pd

def index(request):
    return render(request, 'dashboard/index.html')

def create_csv(request):
    df = pd.read_csv('/home/ubuntu/PycharmProjects/rei_power/rei/sensor_data.csv')
    df = df.reset_index()

    result = []

    for index, row in df.iterrows():
        if (row['A'] == row['B']):
            result.append('T')
        else:
            result.append('F')

    df['Result'] = result
    df.to_csv('/home/ubuntu/PycharmProjects/rei_power/rei/sensor_data_results.csv', index=False)

    return render(request, 'dashboard/success.html')
