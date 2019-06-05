import os
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config as cfg
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def _sort_by_date(year_data): 
    month_data = year_data.loc[year_data['month'] == 1]
    sorted_df = month_data.sort_values(by = 'day')
    
    for month in range(2,13):
        month_data = year_data.loc[year_data['month'] == month]
        sorted_month_data = month_data.sort_values(by = 'day')
        sorted_df = pd.concat([sorted_df, sorted_month_data],)
    
    return sorted_df


def _interpolate_weather_data(weather):
    """ 
    Parameters
    --------
    weather: data frame containing missing values 
    
    Returns
    --------
    complete_data: data frame not containing any missing values
    """
    #See description
   

    wind = weather[["wind_mSec"]].interpolate(method='linear')
    temp7h = weather['temp_7h'].fillna(method='bfill')
    temp14h = weather['temp_14h'].fillna(method='bfill')
    temp19h = weather['temp_19h'].fillna(method='bfill')
    hum_missing = weather[["hum_14h","hum_19h","hum_7h"]]
    hum_missing = hum_missing.reset_index()
    hum_missing = hum_missing.interpolate(method = 'piecewise_polynomial')
    hum_missing.set_index(["year","month","week","day"],inplace = True)
    
    #2012 to 2018_01 are ok (just a bit of NaN)
    wind_degrees_only = weather[["wind_degrees"]]
    wind_degrees_2012_until_2018_01 = wind_degrees_only.iloc[0:2223]
    wind_degrees_2012_until_2018_01 = wind_degrees_2012_until_2018_01.interpolate(method='linear')
    
    #Tricky part(find the indices before)---description
    wind_degrees_2018_Feb = wind_degrees_only.iloc[2223:2251]     
    wind_degrees_2018_Feb.iloc[:] = wind_degrees_only.iloc[1858:1886].values
    
    wind_degrees_2018_March = wind_degrees_only.iloc[2251:2282] #no NaN
    
    wind_degrees_2018_April = wind_degrees_only.iloc[2282:2312]
    wind_degrees_2018_April.iloc[:] = wind_degrees_only.iloc[1917:1947].values
    
    wind_degrees_2018_May = wind_degrees_only.iloc[2312:2343]
    wind_degrees_2018_May.iloc[:] = wind_degrees_only.iloc[1947:1978].values
    
    wind_degrees_2018_June_July_August = wind_degrees_only.iloc[2343:] #no NaN
    
    skyCover_and_sun = weather[['skyCover_14h','skyCover_19h','skyCover_7h','sun_hours']].interpolate(method='linear')
    
    
    wind_degrees = pd.concat([wind_degrees_2012_until_2018_01, wind_degrees_2018_Feb, wind_degrees_2018_March,
                              wind_degrees_2018_April,wind_degrees_2018_May, wind_degrees_2018_June_July_August])
   


    complete_data = pd.concat([ weather[['temp_dailyMax','temp_dailyMean',
                                                'temp_dailyMin','temp_minGround','hum_dailyMean','precip']],hum_missing,wind,temp7h,temp14h,temp19h,skyCover_and_sun], axis = 1, sort = True)

    return complete_data


def _handle_outliers(noisy_data):
    """ 
    Parameters
    --------
    noisy_data: data frame that contains outliers
    
    Returns
    --------
    cleaned_data: data frame with outliers
    """

    def out_std(s, nstd=3.0):
        data_mean, data_std = s.mean(), s.std()
        cut_off = data_std * nstd
        lower, upper = data_mean - cut_off, data_mean + cut_off

        return [False if x < lower or x > upper else True for x in s]

    noisy_data = noisy_data[noisy_data.apply(out_std, nstd=1.8)]
    noisy_data.loc[noisy_data.temp_14h >= 40, 'temp_14h'] = np.NaN
    
    noisy_data = noisy_data.reset_index()
    noisy_data = noisy_data.interpolate(method = 'piecewise_polynomial')
    noisy_data.set_index(["year","month","week","day"],inplace = True)
    
    cleaned_data = noisy_data
    return cleaned_data


def _aggregate_weekly(data):
    """ 
    Parameters
    --------
    data: weather data frame
    
    Returns
    --------
    weekly_stats: data frame that contains statistics aggregated on a weekly basis
    """
    
   
    data=data.reset_index()
    data = data.iloc[1:] #Aggregation with a 2012 1 1 is actually week 52 from 2011 (because of the way they count it) so we should not include it in aggregation
    data.set_index(["year","week"],inplace=True)

    data["temp_weeklyMin"] = data.pivot_table('temp_dailyMin', index=["year",'week'],aggfunc=min)
    data["temp_weeklyMax"] = data.pivot_table('temp_dailyMax', index=["year",'week'],aggfunc=np.mean)
    data["temp_weeklyMean"] = data.pivot_table('temp_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["temp_7h_weeklyMedian"] = data.pivot_table('temp_7h', index=["year",'week'],aggfunc=np.median)
    data["temp_14h_weeklyMedian"] = data.pivot_table('temp_14h', index=["year",'week'],aggfunc=np.median)
    data["temp_19h_weeklyMedian"] = data.pivot_table('temp_19h', index=["year",'week'],aggfunc=np.median)
    data["hum_weeklyMean"] = data.pivot_table('hum_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["hum_7h_weeklyMedian"] = data.pivot_table('hum_7h', index=["year",'week'],aggfunc=np.median)
    data["hum_14h_weeklyMedian"] = data.pivot_table('hum_14h', index=["year",'week'],aggfunc=np.median)
    data["hum_19h_weeklyMedian"] = data.pivot_table('hum_19h', index=["year",'week'],aggfunc=np.median)
    data["precip_weeklyMean"] = data.pivot_table('precip', index=["year",'week'],aggfunc=np.mean)
    data["wind_mSec_mean"] = data.pivot_table('wind_mSec', index=["year",'week'],aggfunc=np.mean)
    
    weekly_weather_data = data.drop(["temp_minGround","sun_hours","skyCover_7h","skyCover_19h","skyCover_14h",'temp_dailyMin', 'temp_dailyMax','temp_dailyMean','temp_7h','temp_14h','temp_19h','hum_dailyMean','hum_7h','hum_14h','hum_19h','precip','wind_mSec'], 1)
    ww2012=weekly_weather_data.xs(2012,level='year')
    ww2012=ww2012[~ww2012.index.get_level_values(0).duplicated()]
    ww2013=weekly_weather_data.xs(2013,level='year')
    ww2013=ww2013[~ww2013.index.get_level_values(0).duplicated()]
    ww2014=weekly_weather_data.xs(2014,level='year')
    ww2014=ww2014[~ww2014.index.get_level_values(0).duplicated()]
    ww2015=weekly_weather_data.xs(2015,level='year')
    ww2015=ww2015[~ww2015.index.get_level_values(0).duplicated()]
    ww2016=weekly_weather_data.xs(2016,level='year')
    ww2016=ww2016[~ww2016.index.get_level_values(0).duplicated()]
    ww2017=weekly_weather_data.xs(2017,level='year')
    ww2017=ww2017[~ww2017.index.get_level_values(0).duplicated()]
    ww2018=weekly_weather_data.xs(2018,level='year')
    ww2018=ww2018[~ww2018.index.get_level_values(0).duplicated()]
    weekly_weather_data = pd.concat([ww2012,ww2013,ww2014,ww2015,ww2016,ww2017,ww2018], keys=['2012','2013',"2014","2015","2016","2017","2018"])
    weekly_weather_data.index.names = ['year','week']
    weekly_weather_data.reset_index(inplace=True)
    weekly_weather_data['year'] = pd.to_numeric(weekly_weather_data['year'])
    weekly_weather_data['week'] = pd.to_numeric(weekly_weather_data['week'])
    weekly_weather_data.set_index(["year","week"],inplace=True)
    
    return weekly_weather_data


def _merge_data(weather_df, influenza_df):
    """ 
    Parameters
    --------
    weather_df: weekly weather data frame
    influenza_df: influenza data frame
    
    Returns
    --------
    merged_data: merged data frame that contains both weekly weather observations and prevalence of influence infections
    """
    merged_data = weather_df.join(influenza_df)
    merged_data=merged_data.reset_index()
    merged_data = merged_data.apply(pd.to_numeric)
    merged_data["weekly_infections"]= merged_data["weekly_infections"].interpolate(method = 'linear')
    
    merged_data.iloc[327:,14] = merged_data.iloc[275:297,14].values
    merged_data.set_index("year","week")
    return merged_data


def load_weather_data(datapath):
    """ 
    Load all weather data files and combine them into a single Pandas DataFrame.
    Add a week column and a hierarchical index (year, month, week, day)
    
    Returns
    --------
    weather_data: data frame containing the weather data
    """
    
    years_to_load = ['2012','2013','2014','2015','2016','2017','2018']
    paths = [os.path.join(datapath, 'weather_{}.csv'.format(year)) for year in years_to_load]
    col = list()
    datalist = list()
    week_col = list()
    load_years=list()

    first_year = True
    for filename in paths:
        year_data = pd.read_csv(filename)
        year_data = _sort_by_date(year_data)
        if first_year:
            weather_data = year_data
            first_year = False
        else:
            weather_data = pd.concat([weather_data, year_data], sort = True)

    year, week, day = dt.date(int(years_to_load[0]),1,1).isocalendar()
    count = 8-day
    rows, cols = weather_data.shape
    first_year = int(years_to_load[0])
    
    for i in range(count):
        week_col.append(week)

    if week == 52:
        if dt.date(year, 12, 28).isocalendar()[1] == 53: # every 4-th year, check it- add it!
            week = 53
        else:
            week = 1
            if year == first_year:
                first_year += 1
    elif week == 53:
        week = 1
        if year == first_year:
            first_year += 1        
        week = 1
    else:
        week += 1
        
        
    while len(week_col) < rows:
        for i in range(7):
            if len(week_col)<rows:
                week_col.append(week)

        if week == 52:
            if dt.date(int(first_year),12,28).isocalendar()[1] == 53: #53 woche?
                week = 53
            else:
                week = 1
                first_year += 1
        elif week == 53:
            week = 1 
            first_year += 1
        else:
            week += 1
            
    weather_data.drop("Unnamed: 0",axis=1,inplace=True)
   
    weather_data.insert(loc=0, column='week', value = week_col)
    weather_data.set_index(["year","month","week","day"],inplace=True)
    return weather_data


def load_influenza_data(datapath):
    """ 
    Load and prepare the influenza data file
    
    Returns
    --------
    influenza_data: data frame containing the influenza data
    """
    influenza_data = pd.read_csv(os.path.join(datapath, 'influenza.csv'))
    influenza_data = pd.DataFrame(influenza_data)
    influenza_data = influenza_data[["Neuerkrankungen pro Woche","Jahr","Kalenderwoche"]]
    new_names = {'Neuerkrankungen pro Woche':'weekly_infections',"Jahr":"year","Kalenderwoche":"week"}
    
    influenza_data.rename(index=str, columns=new_names,inplace=True)
    influenza_data['week'] = influenza_data['week'].str.replace('Woche', '')
    influenza_data['week'] = influenza_data['week'].str.replace('.', '')
    influenza_data['year'] = influenza_data['year'].astype(int)
    influenza_data['week'] = influenza_data['week'].astype(int)
    influenza_data.set_index(["year","week"],inplace=True)
    return influenza_data


def _split_xy(data):
    data = data.reset_index()
    year = data["year"]
    week = data["week"]
    data = data.drop(["year","week"],axis=1)
    cols = data.columns

    X = data.drop(["index", "month", "day", "weekly_infections"], axis=1)

    y = data["weekly_infections"].reset_index()
    y.drop(["index"], axis=1, inplace=True)

    return X, y 


def _scaler_fit(data):
    scaler = MinMaxScaler()
    return scaler.fit(data)


def _scaler_transform(scaler, data):
    cols = data.columns
    data = scaler.transform(data)
    return pd.DataFrame(data, columns=cols)


def load_data(datapath):
    weather = load_weather_data(datapath)
    weather = _interpolate_weather_data(weather)
    weather = _handle_outliers(weather)
    weather = _aggregate_weekly(weather)

    flu = load_influenza_data(datapath)

    data = _merge_data(weather, flu)
    data_2018 = data.loc[data['year'] == 2018]
    data = data.loc[data['year'] != 2018]

    X, y = _split_xy(data)
    X_2018, y_2018 = _split_xy(data_2018)

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED)

    # scale dataset
    scaler_x = _scaler_fit(X_train)
    scaler_y = _scaler_fit(y_train)
    X_train, y_train = _scaler_transform(scaler_x, X_train), _scaler_transform(scaler_y, y_train)
    X_test, y_test = _scaler_transform(scaler_x, X_test), _scaler_transform(scaler_y, y_test)
    X_2018, y_2018 = _scaler_transform(scaler_x, X_2018), _scaler_transform(scaler_y, y_2018)

    return X_train, y_train, X_test, y_test, X_2018, y_2018


def pair_plot(data, output_path):
    data = data[["temp_weeklyMin","temp_weeklyMax","temp_weeklyMean","temp_7h_weeklyMedian","hum_weeklyMean",\
                        "hum_7h_weeklyMedian","precip_weeklyMean","wind_mSec_mean","weekly_infections"]].reset_index()

    plt.figure()
    plot = sns.pairplot(data)
    plot.savefig(output_path)
    plt.clf()


def corr_plot(data, output_path):
    data = data.reset_index()
    data.drop(['year',"week"],axis=1,inplace=True)

    # calculate the correlation matrix
    corr = data.corr()

    # plot the heatmap
    plt.figure()
    plot = sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plot.figure.savefig(output_path)
    plt.clf()


def save_dataset(X, y, datapath, suffix=''):
    X_fp = os.path.join(datapath, '_'.join(['X', suffix]) + '.pkl')
    y_fp = os.path.join(datapath, '_'.join(['y', suffix]) + '.pkl')

    pickle.dump(X, open(X_fp, 'wb'))
    pickle.dump(y, open(y_fp, 'wb'))

if __name__ == '__main__':
    datapath = cfg.DATA_PATH

    X_train, y_train, X_test, y_test, X_2018, y_2018 = load_data(datapath)
    save_dataset(X_train, y_train.values.flatten(), datapath, suffix='train')
    save_dataset(X_test, y_test.values.flatten(), datapath, suffix='test')
    save_dataset(X_2018, y_2018.values.flatten(), datapath, suffix='2018')

    #pair_plot(d, os.path.join(datapath, 'pair.png'))
    #corr_plot(d, os.path.join(datapath, 'corr.png'))
