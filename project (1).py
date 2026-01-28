import math
import os.path

import numpy
import pandas
import copy
import openpyxl
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import lightgbm as ltb
from sklearn.model_selection import train_test_split
_temp_pd = pandas.read_csv('covid_19_clean_data1.csv', parse_dates=True)
import warnings

countries_to_check_list = ['US', 'India']

warnings.filterwarnings("ignore")

all_countries = _temp_pd['Country/Region'].unique()

complete_country_info = {}

if not os.path.isdir('pre_data_analysis'):
    os.mkdir('pre_data_analysis')

if not os.path.isdir('preparing_data_for_model'):
    os.mkdir('preparing_data_for_model')

for each_country in all_countries:
    each_country_info = {'name': each_country, 'raw_data': _temp_pd[_temp_pd["Country/Region"] == each_country]}
    each_country_info['na_stats'] = each_country_info['raw_data'].isna().sum()

    complete_country_info[each_country] = each_country_info


na_stats_list = []
for each_country in complete_country_info.keys():
    _temp_dict = {'country': each_country,
                  'recovered': complete_country_info[each_country]['na_stats']['Recovered'],
                  'confirmed': complete_country_info[each_country]['na_stats']['Confirmed'],
                  'deaths': complete_country_info[each_country]['na_stats']['Deaths']}

    na_stats_list.append(_temp_dict)

_tem_stats_pd = pandas.DataFrame(na_stats_list)

_tem_stats_pd.to_excel('project_na_stats.xlsx', index=False)


def initial_analysis_graphs_temp(country_to_check_list):
    for each_country_to_check in country_to_check_list:
        analyse_adjust_data(each_country_to_check, False, folder_to_save='pre_data_analysis')



def analyse_adjust_data(country, is_diff=False, folder_to_save='.'):
    current_data = copy.deepcopy(complete_country_info[country]['raw_data'])
    current_data.set_index('Date', inplace=True)
    original = 'Original'
    for each_attribute in ['Recovered', 'Deaths', 'Confirmed']:
        if is_diff:
            current_data_trimmed = current_data[each_attribute].diff(periods=1)
            original = 'Adjusted'
        else:
            current_data_trimmed = current_data[each_attribute]
        current_data_trimmed.plot(figsize=(12, 5))
        plt.title(f'{country} {each_attribute} cases graph')
        plt.xlabel('Date')
        plt.ylabel(each_attribute)
        plt.savefig(f'{folder_to_save}/{country}_{each_attribute}_{original}.png')
        plt.close()
    del current_data_trimmed


def initial_analysis_graphs(country_to_check_list):
    for each_country_to_check in country_to_check_list:
        analyse_adjust_data(each_country_to_check, False, folder_to_save='pre_data_analysis')


initial_analysis_graphs(countries_to_check_list)


def adjusting_data(countries_list):
    for each_country_name in countries_list:
        analyse_adjust_data(each_country_name, True, folder_to_save='preparing_data_for_model')


adjusting_data(countries_to_check_list)


def adfuller_test(data,country,attr):
    context = f'{country} {attr}'
    new_data = copy.deepcopy(data)
    # new_data.set_index('Date',inplace=True)
    new_data = new_data.dropna()
    if not os.path.isdir('adfuller_test_results'):
        os.mkdir('adfuller_test_results')
    if not os.path.isdir(f'adfuller_test_results/{country}'):
        os.mkdir(f'adfuller_test_results/{country}')
    _temp_file = open(f'adfuller_test_results/{country}/{context}.txt','w')

    _temp_file.write(f'--Ad fuller test for {context} -- \n')
    return_param_list=['adf','pvalue','usedlag','nobs','critical_values','icbest']
    ad_test_result = adfuller(new_data,autolag='AIC')
    i=0

    if type(ad_test_result) == tuple:
        for each in ad_test_result:

            _temp_file.write(f'{return_param_list[i]} : {each} \n')
            i += 1

    _temp_file.write('--------------------------------------\n')
    _temp_file.close()
    del new_data


def check_all_adfuller_test():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            adfuller_test(complete_country_info[each_entry]['raw_data'][each_attr],each_entry,each_attr)



check_all_adfuller_test()

arima_values = {}
def auto_arima_check(country,attribute):
    current_data = copy.deepcopy(complete_country_info[country]['raw_data'])
    current_data.set_index('Date', inplace=True)
    current_data = current_data[attribute]
    current_data = current_data.dropna()
    arima_stats = pm.auto_arima(current_data)

    if not os.path.isdir("arima_results"):
        os.mkdir('arima_results')
    if not os.path.isdir(f'arima_results/{country}'):
        os.mkdir(f'arima_results/{country}')
    _temp_file = open(f'arima_results/{country}/{country}_{attribute}.txt','w')
    _temp_file.write(str(arima_stats.summary()))
    arima_values[f'{country}_{attribute}'] = str(arima_stats).replace('ARIMA(','').split(')')[0].strip()
    _temp_file.close()
    del current_data


def auto_arima_analysis():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            auto_arima_check(each_entry,each_attr)


auto_arima_analysis()

analyzed_country_predictions = {}
def fit_arima(country,attr):
    current_data = copy.deepcopy(complete_country_info[country]['raw_data'])
    current_data.set_index('Date', inplace=True)
    current_data = current_data[attr]
    current_data = current_data.dropna()
    arima_order = arima_values[f'{country}_{attr}'].split(',')
    _temp_model = ARIMA(current_data,order=(int(arima_order[0]),int(arima_order[1]),int(arima_order[2])))
    _temp_model = _temp_model.fit()
    predictions = _temp_model.predict(start=current_data.shape[0]-30,end=current_data.shape[0],typ='levels')
    analyzed_country_predictions[f'{country}_{attr}_predictions'] = predictions
    analyzed_country_predictions[f'{country}_{attr}_train_data'] = current_data
    del current_data

def all_fit_data_to_arima_model():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            fit_arima(each_entry,each_attr)


all_fit_data_to_arima_model()


def plot_predictions(country,attr):
    train_data = analyzed_country_predictions[f'{country}_{attr}_train_data']
    predicted_data = analyzed_country_predictions[f'{country}_{attr}_predictions']
    # train_data.plot(legend=True)
    predicted_data.plot(legend=True)
    if not os.path.isdir('prediction_graphs'):
        os.mkdir('prediction_graphs')
    if not os.path.isdir(f'prediction_graphs/{country}'):
        os.mkdir(f'prediction_graphs/{country}')
    plt.title(f'Arima model {country} {attr} predictions')
    plt.ylabel(f'{attr}')
    plt.savefig(f'prediction_graphs/{country}/{country}_{attr}.png')
    plt.close()


def all_plot_graphs_predicted():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            plot_predictions(each_entry,each_attr)

all_plot_graphs_predicted()

def calculate_mean_squared_error(country,attr,model):
    train_data = analyzed_country_predictions[f'{country}_{attr}_train_data']
    predicted_data = analyzed_country_predictions[f'{country}_{attr}_predictions']
    print(f'------{country} {attr} cases ------ for model {model}----root mean squared error ')
    print(math.sqrt(mean_squared_error(train_data[train_data.shape[0]-31:],predicted_data)))
    print('----------------------------------------')



def all_predict_mean_squared_error():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            calculate_mean_squared_error(each_entry,each_attr,'ARIMA')

all_predict_mean_squared_error()


def model_lgbm_fit(country,attr):
    x_train = copy.deepcopy(complete_country_info[country]["raw_data"][attr])
    x_train = x_train.dropna()
    # print(x_train)
    x_index = numpy.array(x_train.index).reshape(-1,1)
    x_train = numpy.array(x_train).reshape(-1,1)



    # print(x_train.shape)
    y_test = copy.deepcopy(x_train[-30:])
    _temp_model = ltb.LGBMRegressor()
    _temp_model.fit(x_index,x_train)
    pred_values = _temp_model.predict(x_train[-30:])
    print(f' ----------{country} {attr} cases----- root mean squared - model lgbm regressor')
    print(math.sqrt(mean_squared_error(y_test,pred_values)))
    print('------------------------------------------------')


def all_lgbm_fit_data():
    for each_entry in countries_to_check_list:
        for each_attr in ['Recovered', 'Deaths', 'Confirmed']:
            model_lgbm_fit(each_entry,each_attr)


all_lgbm_fit_data()


def model_lgbm_fit_v2(country):
    data = pandas.DataFrame(complete_country_info[country]['raw_data'],columns=['Date','Confirmed',"Deaths",'Recovered'])
    data.set_index('Date',inplace=True)
    data = data.dropna()

    x_train = pandas.DataFrame(data,columns=['Confirmed','Deaths'])

    print(len(x_train))
    y_train = numpy.array(data['Recovered'])
    print(len(y_train))
    y_actual = numpy.array(data['Recovered'])
    _temp_model = ltb.LGBMRegressor()
    _temp_model.fit(x_train, y_train)
    actual = pandas.DataFrame(complete_country_info[country]['raw_data'],columns=['Date','Confirmed',"Deaths"])
    actual.set_index('Date',inplace=True)
    actual = numpy.array(actual)
    pred_values = _temp_model.predict(actual).reshape(-1,1)
    print(pred_values)
    plt.title(f'{country} -- recovered prediction based on Confirmed and death cases')
    plt.xlabel('Day since records stared')
    plt.ylabel('Recovered')
    plt.plot(y_actual)
    plt.plot(pred_values)
    plt.savefig(f'prediction_graphs/{country}/{country}_recovered_based_on_confirmed_and_death.png')
    plt.show()


def all_lgbm_fit_data_v2():
    for each_entry in countries_to_check_list:
        model_lgbm_fit_v2(each_entry)


all_lgbm_fit_data_v2()

model_lgbm_fit('US','Deaths')



