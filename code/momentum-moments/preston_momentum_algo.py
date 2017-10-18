"""
Preston Yadegar's implementation of the Momentum Moments Research Paper by Barroso and Santa-Clara. This strategy uses Size and Value factors and then ranks the assets based on returns. As a momentum strategy, we long the 'winners' (highest returns) and short the 'losers' (lowest returns). We then adjust our leverage based on the underlying volatility in the past 6 months.
"""
from quantopian.pipeline.factors import AverageDollarVolume, Returns, AnnualizedVolatility
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data import morningstar
import pandas as pd
import numpy as np
import math

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.long_leverage = 0.5
    context.short_leverage = -0.5
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    
class Size(CustomFactor):
    # Custom Factor using the size score from morningstar which calculates the cumulative market cap from 5 years of data
    inputs = [USEquityPricing.close, morningstar.asset_classification.size_score]
    window_length = 1
    def compute(self, today, assets, out, four, size):
        out[:] = size[-1]
        
class Value(CustomFactor):
    # Custom Factor using the value score from morningstar which incorporates EPS, book-to-value, dividends, and revenues
    inputs = [USEquityPricing.close, morningstar.asset_classification.value_score]
    window_length = 1
    def compute(self, today, assets, out, four, value):
        out[:] = value[-1]
# Not sure what four is supposed to represent, remove the name four above and inspect the errors
            
def make_pipeline(context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    pipe = Pipeline()
    
    dollar_volume = AverageDollarVolume(window_length=1)
    pipe.add(dollar_volume, 'dollar_volume')
    
    value = Value(window_length=1)
    size = Size(window_length=1)
    pipe.add(value, 'value')
    pipe.add(size, 'size')
    
    # Subtract 2 month returns from 12 months to get desired values
    recent_returns = Returns(window_length=252)
    other_returns = Returns(window_length=42)
    recent_returns -= other_returns
    
    pipe.add(recent_returns, 'recent_returns')
    
    # Create factors for the size and value to mock the Fama French factors
    
    # Take the lowest and highest value score rankings of stocks
    low_value = value.percentile_between(0, 25)
    high_value = value.percentile_between(75, 100)
    # Take the lowest and highest size score rankings of stocks
    low_size = size.percentile_between(0, 25)
    high_size = size.percentile_between(75, 100)

    high_dollar_volume = dollar_volume.percentile_between(50, 100)
    
    loser_returns = recent_returns.percentile_between(0,10,mask=((low_value | high_size) & high_dollar_volume))
    winner_returns = recent_returns.percentile_between(90,100,mask=((high_value | low_size) & high_dollar_volume))

    pipe.add(recent_returns.rank(mask=high_dollar_volume), 'recent_returns_rank')

    pipe.set_screen(loser_returns | winner_returns)

    pipe.add(loser_returns, 'loser_returns')
    pipe.add(winner_returns, 'winner_returns')
  
    return pipe
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
    # Create lists of the securities to go long and short
    context.long_secs = context.output[context.output['winner_returns']]
    context.short_secs = context.output[context.output['loser_returns']]
    
    # Calculate variance forecast from previous 6 months returns
    spy_prices = data.history(sid(8554), fields="price", bar_count=125, frequency="1d")
    #returns = Returns(window_length=125)
    #spy_returns = returns[sid(8554)]
    calc_returns = compute_volatility(spy_prices)
    var_forecast = 21 * (calc_returns / 126)
    # Calculate leverage weight based on variance estimate
    vol_target = 0.12
    variance_adjusted = vol_target / var_forecast
    context.long_leverage = variance_adjusted / 2
    context.short_leverage = variance_adjusted / -2
    
    # Pass through the combined long/short securities list as one
    context.security_list = context.long_secs.index.union(context.short_secs.index).tolist()
    context.security_set = set(context.security_list)
     
        
def compute_volatility(price_history):  
    # Compute daily returns  
    daily_returns = price_history.pct_change().dropna().values  
    # Compute daily volatility  
    historical_vol_daily = np.std(daily_returns,axis=0)  
    # Convert daily volatility to annual volatility, assuming 252 trading days  
    historical_vol_annually = historical_vol_daily*math.sqrt(252)  
    # Return estimate of annual volatility  
    return 100*historical_vol_annually  
        
def my_assign_weights(context):
    """
    Assign weights to securities that we want to order.
    """
    context.long_weight = context.long_leverage / len(context.long_secs)
    context.short_weight = context.short_leverage / len(context.short_secs)
    pass
 
def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    my_assign_weights(context)
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
    long_val = context.long_weight * len(context.long_secs)
    short_val = context.short_weight * len(context.short_secs)
    record(long_weight=long_val, short_weight=short_val)
    pass




# coding: utf-8

# In[ ]:


"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, Returns, AnnualizedVolatility
import numpy as np
import pandas as pd
import math
#from quantopian.pipeline.filters.morningstar import Q1500US

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.long_leverage = 0.5
    context.short_leverage = -0.5
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(context), 'my_pipeline')
         
def make_pipeline(context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    pipe = Pipeline()
    
    # Base universe set to the Q500US
    # base_universe = Q1500US()
    
    dollar_volume = AverageDollarVolume(window_length=1)
    pipe.add(dollar_volume, 'dollar_volume')
    
    # Subtract 2 month returns from 12 months to get desired values
    recent_returns = Returns(window_length=252)
    other_returns = Returns(window_length=42)
    recent_returns -= other_returns
    
    pipe.add(recent_returns, 'recent_returns')

    high_dollar_volume = dollar_volume.percentile_between(95, 100)
    
    loser_returns = recent_returns.percentile_between(0,10,mask=high_dollar_volume)
    winner_returns = recent_returns.percentile_between(90,100,mask=high_dollar_volume)

    pipe.add(recent_returns.rank(mask=high_dollar_volume), 'recent_returns_rank')

    pipe.set_screen(loser_returns | winner_returns)

    pipe.add(loser_returns, 'loser_returns')
    pipe.add(winner_returns, 'winner_returns')
  
    return pipe
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
    # Create lists of the securities to go long and short
    context.long_secs = context.output[context.output['winner_returns']]
    context.short_secs = context.output[context.output['loser_returns']]
    
    # Calculate variance forecast from previous 6 months returns
    spy_prices = data.history(sid(8554), fields="price", bar_count=125, frequency="1d")
    #returns = Returns(window_length=125)
    #spy_returns = returns[sid(8554)]
    calc_returns = compute_volatility(spy_prices)
    var_forecast = 21 * (calc_returns / 126)
    # Calculate leverage weight based on variance estimate
    vol_target = 0.12
    variance_adjusted = vol_target / var_forecast
    if variance_adjusted > 1:
        context.long_leverage = 0.5
        context.short_leverage = -0.5
    elif variance_adjusted > 0.75:
        context.long_leverage = 0.375
        context.short_leverage = -0.375
    elif variance_adjusted > 0.5:
        context.long_leverage = 0.25
        context.short_leverage = -0.25
    elif variance_adjusted > 0:
        context.long_leverage = 0.125
        context.short_leverage = -0.125
    
    
    # Pass through the combined long/short securities list as one
    context.security_list = context.long_secs.index.union(context.short_secs.index).tolist()
    context.security_set = set(context.security_list)
     
        
def compute_volatility(price_history):  
    # Compute daily returns  
    daily_returns = price_history.pct_change().dropna().values  
    # Compute daily volatility  
    historical_vol_daily = np.std(daily_returns,axis=0)  
    # Convert daily volatility to annual volatility, assuming 252 trading days  
    historical_vol_annually = historical_vol_daily*math.sqrt(252)  
    # Return estimate of annual volatility  
    return 100*historical_vol_annually  
        
def my_assign_weights(context):
    """
    Assign weights to securities that we want to order.
    """
    context.long_weight = context.long_leverage / len(context.long_secs)
    context.short_weight = context.short_leverage / len(context.short_secs)
    pass
 
def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    my_assign_weights(context)
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
    long_val = context.long_weight * len(context.long_secs)
    short_val = context.short_weight * len(context.short_secs)
    record(long_weight=long_val, short_weight=short_val)
    pass

