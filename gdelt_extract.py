import random
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import time
import csv

# W1 : 01.10.2016 -> 09.10.2016
# W2 : 15.10.2016 -> 23.10.2016
# W3 : 25.10.2016 -> 02.11.2016
# W4 : 05.11.216  -> 13.11.2016
# W5 : 15.11.2016 -> 23.11.2016

week = 'W1'

times = {
    'W1' : {
        'start': 20161001000000,
        'end'  : 20161007000000
    },
    'W2' : {
        'start': 20161015000000,
        'end'  : 20161022000000
    },
    'W3' : {
        'start': 20161025000000,
        'end'  : 20161101000000
    },
    'W4' : {
        'start': 20161105000000,
        'end'  : 20161112000000
    },
    'W5' : {
        'start': 20161115000000,
        'end'  : 20161122000000
    }
}


start_stamp = times[week]['start']
file_base_str = ".mentions.CSV"
n_files = 768 # indication : ~3k files / month with update every 15 mins // 96 / day
#save_fname = "test_" + str(n_files) + "_files_" + time.strftime('%Y_%m_%d_%H_%M_%S') + ".csv"
#save_fname = "sources_" + time.strftime('%Y_%m_%d_%H_%M_%S') + ".csv"
save_fname = "sources_" + week + ".csv"
gid_key = 'GLOBALEVENTID'
names = [gid_key, 'EventTimeDate', 'MentionTimeDate' ,'SourceName']
news_path = 'dl/mentions/'

def read_news(filename):
    try:
        df = pd.read_csv(news_path + filename, header=None, usecols=[0], names=[gid_key], sep='\t')
    except:
        print("[WARNING] File {} not found. Moving on ...".format(filename))
        df = pd.DataFrame(columns=[gid_key])
    df['counter'] = 1
    return df

def read_source_event(filename):
    try:
        df = pd.read_csv(news_path + filename, header=None, usecols=[0,1,2,4], names=names, sep='\t')
    except:
        print("[WARNING] File {} not found. Moving on ...".format(filename))
        df = pd.DataFrame(columns=names)
    return df

def count_events(df):
    return df.groupby(gid_key, as_index=False).agg('sum')

def save_event_sources_distribution(df, sorted=False):
    if sorted is True:
        df = df.sort_values('counter')['counter']
    df.to_csv(save_fname)
    print("==> Saved result to {}".format(save_fname))

def timestamp_factory(idx, base=start_stamp):
    stmp = pd.to_datetime(base, format="%Y%m%d%H%M%S") + pd.Timedelta(minutes=15*idx)
    return stmp.strftime('%Y%m%d%H%M%S') + file_base_str

def save_ts(ts):
    fname = 'times_' + save_fname
    with open(fname,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['file_count','time'])
        for row in ts:
            csv_out.writerow(row)
    print("==> Saved times result to {}".format(fname))

## MAIN

start_file = start_stamp + file_base_str
times = []
print("Getting file {} ...".format(start_file))
def count_sources():
    base_df = count_events(read_news(start_file))
    print("{} entries processed".format(len(base_df)))

    for i in range(1,n_files):
        start = timer()
        filename = timestamp_factory(i)
        curr_df = read_news(filename)
        print("Getting file {} ...".format(filename))
        base_df = count_events(base_df.merge(curr_df, how='outer', on=[gid_key, 'counter']))
        print("{} entries processed. {} total".format(len(curr_df), len(base_df)))
        end = timer()
        times.append((i,end-start))
        print("Done processing file #{}. time = {}".format(i, end-start))

    save_event_sources_distribution(base_df)
    save_ts(times)

    print("\n-- Done.")
    print("-- Total time : {} seconds.".format(sum([pair[1] for pair in times])))
    print("-- Total entries : {} from {} files.".format(len(base_df), n_files))

def process_sources():
    event_df = pd.read_csv('sources_W5.csv')
    event_list = event_df[event_df['counter']][gid_key]
    base_df = pd.DataFrame()

    for i in range(0,n_files):
        filename = timestamp_factory(i)
        curr_df = read_source_event(filename)
        base_df = pd.concat([base_df, curr_df[curr_df[gid_key].isin(event_list)]])
    save_event_sources_distribution(base_df)

count_sources()
#process_sources()
