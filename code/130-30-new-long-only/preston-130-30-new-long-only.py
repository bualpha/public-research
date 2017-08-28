"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from odo import odo
from statsmodels import regression
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data import morningstar
from statsmodels.stats.stattools import jarque_bera
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import Returns, AverageDollarVolume

# Custom Factor 1 : Price to Trailing 12 Month Sales       
class Price_to_TTM_Sales(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, ps):
        out[:] = -ps[-1]

        
# Custom Factor 2 : Price to Trailing 12 Month Cashflow
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [morningstar.valuation_ratios.pcf_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pcf):
        out[:] = -pcf[-1] 

# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]
        
        
# This pulls all necessary data in one step
def Data_Pull():
    
    # create the pipeline for the data pull
    Data_Pipe = Pipeline()
    
    # create SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')

    # Price / TTM Sales
    Data_Pipe.add(Price_to_TTM_Sales(), 'Price / TTM Sales')
    
    # Price / TTM Cashflows
    Data_Pipe.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflow')
        
    return Data_Pipe


# function to filter out unwanted values in the scores
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x   


def standard_frame_compute(df):
    """
    Standardizes the Pipeline API data pull
    using the S&P500's means and standard deviations for
    particular CustomFactors.

    parameters
    ----------
    df: numpy.array
        full result of Data_Pull

    returns
    -------
    numpy.array
        standardized Data_Pull results        
    numpy.array
        index of equities
    """
    
    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)
    
    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()
    
    # store index values
    index = df.index.values
    
    # turn iinto a numpy array for speed
    df = df.as_matrix()
    
    # create an empty vector on which to add standardized values
    df_standard = np.empty(df.shape[0])
    
    for col_SPY, col_full in zip(df_SPY.T, df.T):
        
        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma)) 

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))
        
        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])
        
        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))
     
    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard,0,0)
    
    return (df_standard, index)


def composite_score(df, index):
    """
    Summarize standardized data in a single number.

    parameters
    ----------
    df: numpy.array
        standardized results
    index: numpy.array
        index of equities
        
    returns
    -------
    pandas.Series
        series of summarized, ranked results
    """
    # sum up transformed data
    df_composite = df.sum(axis=0)
    
    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite,index=index)
    
    # sort descending
    df_composite.sort(ascending=False)

    return df_composite

def initialize(context):
    """
    Called once at the start of the algorithm.
    """ 
    # get data from pipeline
    data_pull = Data_Pull()
    attach_pipeline(data_pull,'Data')
    
    # filter out bad stocks for universe
    mask = filter_universe()
    data_pull.set_screen(mask)
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    # Create our dynamic stock selector
    attach_pipeline(make_pipeline(), 'my_pipeline')

def make_pipeline():
    """
    Function to create a pipeline
    """
    pipe = Pipeline()
    
    # Base universe set to the Q1500US
    base_universe = Q1500US()
    
    pipe.add(YOY_Slope(), 'YOY-Slope')
    
    loser_returns = YOY_Slope.percentile_between(YOY_Slope(),0,10,mask=base_universe)
    winner_returns = YOY_Slope.percentile_between(YOY_Slope(),90,100,mask=base_universe)
    
    pipe.set_screen(loser_returns | winner_returns)

    pipe.add(loser_returns, 'loser_returns')
    pipe.add(winner_returns, 'winner_returns')
    
    return pipe

def linreg(X,Y):
    """ 
    Create a linear regression function for the data
    """
    X = sm.add_constant(X)
    model = regression.linear_model.OLS(Y, X).fit()
    a = model.params[0]
    b = model.params[1]
    X = X[:, 1]

    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    return [a,b]

class YOY_Slope(CustomFactor):
    # Get the YOY slope of prices for the Price Momentum Factor
    inputs = [USEquityPricing.close]
    window_length = 272
    
    def compute(self, today, assets, out, prices):
        window_length = 272
        time = [i for i in range(window_length-271)]
        out[:] = linreg(prices[:-271], time)[1]


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
  
    # These are the securities that we are interested in trading each day.
    context.long_secs = context.output[context.output['winner_returns']]
    context.short_secs = context.output[context.output['loser_returns']]
    context.security_list = context.long_secs.index.union(context.short_secs.index).tolist()
    context.security_set = set(context.security_list)

     
def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    context.long_weight = 1.3 / len(context.long_secs)
    context.short_weight = -0.3 / len(context.short_secs)
    pass
 
def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    my_assign_weights(context, data)
    # Order buys for the winners and sales for the losers
    for security in context.security_list:
        if security in context.long_secs.index:
            order_target_percent(security, context.long_weight)
        elif security in context.short_secs.index:
            order_target_percent(security, context.short_weight)
    # Remove previous securities not included in winner or loser anymore
    for security in context.portfolio.positions:
        if security not in context.security_set and data.can_trade(security):
            order_target_percent(security, 0)
    # Log our winner and loser tickers
    log.info("Today's longs: "+", ".join([long_.symbol for long_ in context.long_secs.index]))
    log.info("Today's shorts: "  +", ".join([short_.symbol for short_ in context.short_secs.index]))
    pass


def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    pass
