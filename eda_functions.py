# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:33:05 2021

@author: jqiu
"""
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import time
import matplotlib.pyplot as plt
import math

# define function for logging
def LOG_EVENTS(str_filename='./logs/eda_pt1.log'):
	# set logging format
	FORMAT = '%(name)s:%(levelname)s:%(asctime)s:%(message)s'
	# get logger
	logger = logging.getLogger(__name__)
	# try making log
	try:
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	except FileNotFoundError:
		os.mkdir('./logs')
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	# change to append
	handler = logging.FileHandler(str_filename, mode='a')
	# set the level to info
	handler.setLevel(logging.INFO)
	# set format
	formatter = logging.Formatter(FORMAT)
	# format the handler
	handler.setFormatter(formatter)
	# add handler
	logger.addHandler(handler)
	# return logger
	return logger

# define function to check out shape of df
def GET_SHAPE(df, logger=None):
	shape = df.shape
	#log it
	if logger:
		logger.warning(f'Data shape is {shape}')

# define function to read chunks of CSV
def CHUNKY_CSV(str_filename, chunksize, logger=None):
	# start timer
	time_start = time.perf_counter()
	# chunk csv
	df = pd.DataFrame()
	for chunk in pd.read_csv(str_filename, chunksize=chunksize, low_memory=False):
		df = pd.concat([df,chunk])
	# log it
	if logger:
		logger.warning(f'Data imported from csv chunks in {(time.perf_counter()-time_start)/60:0.4} min.')
	return df


#df = pd.DataFrame(np.array([[1,2,4,4],[1,1,4,4],[6,7,4,4],[1,1,4,4],[1,1,4,4]]), columns=['a','b','c','c'])
# define function to read json
def JSON_TO_DF(str_filename, logger=None):
	# start timer
	time_start = time.perf_counter()
	# read json file
	df = json.load(open(str_filename, 'r'))
	# store in df
	df = pd.DataFrame.from_dict(df, orient='columns')
	# if we are using a logger
	if logger:
		# log it
		logger.warning(f'Data imported from json in {(time.perf_counter()-time_start)/60:0.4} min.')
	# return
	return df

# define function to read csv
def READ_CSV(str_filename, logger=None):
	#start timer
	time_start = time.perf_counter()
	#read csv
	df = pd.read_csv(str_filename)
	#log it
	if logger:
		logger.warning(f'Data imported from csv in {(time.perf_counter()-time_start)/60:0.4} min.')
	# return
	return df

# define function to write csv
def WRITE_CSV(df, filename='./output/df_raw_eda.csv', logger=None):
	#start timer
	time_start = time.perf_counter()
	df.to_csv(filename, index=False)
	# log it
	if logger:
		logger.warning(f'Data written to csv in {(time.perf_counter()-time_start)/60:0.4} min.')
	
	
# define function to remove columns with no variance 
def DROP_NO_VARIANCE(df, logger=None):
	# create empty list
	list_no_var = []
	# iterate through columns in df
	for col in df.columns:
		# get the series
		series_ = df[col]
		# drop na
		series_.dropna(inplace=True)
		# get count unique
		count_unique = series_.nunique()
		# if count_unique == 1
		if count_unique == 1:
			# append to list
			list_no_var.append(col)
	# drop list_no_var
	df = df.drop(list_no_var, axis=1, inplace=True)
	# log it
	if logger:
		logger.warning(f'list of no-variance columns generated and removed from dataframe')
	return list_no_var, df

# define function to plot % of missing from entire df
def plot_na_overall(df, filename, tpl_figsize=(10,15), logger=None):
	"""
	takes a data frame and returns a pie chart of missing and not missing.
	"""
	# get total number missing
	n_missing = np.sum(df.isnull().sum())
	# get total observations
	n_observations = df.shape[0] * df.shape[1]
	# both into a list
	list_values = [n_missing, n_observations]
	# create axis
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title('Pie Chart of Missing Values')
	ax.pie(x=[n_missing, n_observations], 
	       colors=['y', 'c'],
	       explode=(0, 0.1),
	       labels=['Missing', 'Non-Missing'], 
	       autopct='%1.1f%%')
	# save fig
	plt.savefig(filename, bbox_inches='tight')
	# close plot
	plt.close()
	if logger:
		logger.warning('pie chart for missing % generated')
	# return fig
	return fig

# define function to generate missing proportions df
def GET_PCT_MISSING(df, filename, logger=None):
	# find percent missing for columns
	percent_missing = df.isnull().sum() * 100 / len(df)
	missing_value_df = pd.DataFrame({'column_name': df.columns,
								  'percent_missing': percent_missing})
	if logger:
		logger.warning('missing proportions found')
	# write to csv
	missing_value_df.to_csv(filename, index=False)
	return missing_value_df

# manually search and remove func, ***EXACT MATCH, OR KEY WORD FOR COLUMN NAME***
def SEARCH_N_DESTROY(df, str_name, logger=None):
	# get desired columns into a list
	list_col = []
	# for loop for cols in df
	for col in df:
	    if str_name in col:
	        list_col.append(col)
	# count the list
	count = len(list_col)
	# drop
	df.drop(list_col, axis=1, inplace=True)
	# log
	if logger:
	    logger.warning(f"{count} columns removed with column name key-word '{str_name}'")
	# return
	return df

# define frunciton to drop duplicated and unnecessary columns
def DROP_COLS(df, list_drop_cols, logger=None):
	df = df.drop(list_drop_cols, axis=1)
	count = len(list_drop_cols)
	#log
	if logger:
		logger.warning(f'{count} columns have been removed from the dataframe')
	#return df
	return df

# define function to get rid of all duplicate rows
def RID_DUPES(df, logger=None):
	# get list of dupe rows, keep first unique as false
	list_dup_rows = df.duplicated(keep='first')
	# count the number of dups
	count_dup_rows = sum(list_dup_rows)
	# drop the dup rows
	df = df.drop_duplicates()
	# log it
	logger.warning(f'{count_dup_rows} duplicated rows eliminated')
	# get list dupe cols, keep first unique as false
	list_dup_cols = df.columns.duplicated(keep='first')
	# count number of dup cols
	count_dup_cols = sum(list_dup_cols)
	# drop dup cols
	df = df.loc[:,~df.columns.duplicated()]
	# log it
	if logger:
		logger.warning(f'{count_dup_cols} duplicated cols eliminated')
	# return 
	return df


# list numeric and list non-numeric columns
def GET_NUMERIC(df, logger=None):
	# define non numeric
	list_non_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==False]
	# define numeric
	list_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==True]  
	# log it
	if logger:
		logger.warning('list numeric and list non-numeric generated')
	return list_non_numeric, list_numeric



# define function to compare made/missed payments
def PLOT_BINARY_COMPARISON(ser_bin, str_filename='./output/target_freqplot.png', logger=None):
	# get value counts for each
	ser_val_counts = pd.value_counts(ser_bin)
	# get x
	x = ser_val_counts.index
	# get y
	y = ser_val_counts.values
	# get total
	int_total = len(ser_bin)
	# get pct missed
	flt_pct_missed = (y[1]/int_total)*100
	# get proportion made
	flt_pct_made = (y[0]/int_total)*100
	# create axis
	fig, ax = plt.subplots(figsize=(15, 10))
	# title
	ax.set_title(f'{flt_pct_made:0.4}% = 0, {flt_pct_missed:0.4}% = 1, (N = {int_total})')
	# frequency bar plot
	ax.bar(x, y)
	# ylabel
	ax.set_ylabel('Frequency')
	# xticks
	ax.set_xticks([0, 1])
	# xtick labels
	ax.set_xticklabels(['0','1'])
	# save
	plt.savefig(str_filename, bbox_inches='tight')
	# log it
	if logger: 
		logger.warning(f'target frequency plot saved to {str_filename}')
	# return
	return fig