
# coding: utf-8

# In[ ]:

"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
import datetime
from sqlalchemy import or_
import numpy as np
import pandas as pd
from quantopian.pipeline.factors import Returns


def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    
    # Rebalance every month, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    #schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

class StdDev(CustomFactor):
    def compute(self, today, asset_ids, out, values):
         #Calculates the column-wise standard deviation, ignoring NaNs
            out[:] = np.nanstd(values, axis=0)
   

def my_compute_weights(context):
    """
    Compute ordering weights.
    """
    # Compute even target weights for our long positions and short positions.
    long_weight = 0.5 / len(context.longs)
    short_weight = -0.5 / len(context.shorts)

    return long_weight, short_weight
    pass 
    
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    base_universe = Q1500US()
    
    returns_past_12_months = Returns(window_length=252,mask=base_universe)
    returns_past_2_months = Returns(window_length=42,mask=base_universe)
    returns_relevant = returns_past_12_months-returns_past_2_months
    
    returns_relevant_deciles=returns_relevant.deciles()
    longs = (returns_relevant_deciles.eq(9))
    shorts=(returns_relevant_deciles.eq(0))
    
    securities_to_trade = (longs | shorts)

    return Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'returns_relevant':returns_relevant
        },screen=(securities_to_trade)
        
    )
  
 
    
#Most algorithms save their pipeline Pipeline outputs on context for use in functions other than before_trading_start. For example, an algorithm might compute a set of target portfolio weights as a Factor, store the target weights on context, and then use the stored weights in a rebalance function scheduled to run once per day.


def before_trading_start(context, data):
    """
    Called every month before market open.
    """
    context.output = pipeline_output('my_pipeline')
        # Go long in securities for which the 'longs' value is True
    context.longs = context.output[context.output['longs']].index.tolist()
        # Go short in securities for which the 'shorts' value is True.
    context.shorts = context.output[context.output['shorts']].index.tolist()

    base_universe = Q1500US()
    returns_past_12_months = Returns(window_length=252,mask=base_universe)
    returns_past_2_months = Returns(window_length=42,mask=base_universe)
    returns_relevant = returns_past_12_months-returns_past_2_months
    returns_relevant_deciles=returns_relevant.deciles()
    longs = (returns_relevant_deciles.eq(9))
    shorts=(returns_relevant_deciles.eq(0))   
    WML_returns= context.output['returns_relevant'][context.output['longs']].sum()-context.output['returns_relevant'][context.output['shorts']].sum()                                                      
        
    #std_dev = StdDev(inputs=[WML_returns], window_length=126)
    std_dev=np.std(WML_returns)
        
    if std_dev<0.12:
        context.long_leverage, context.short_leverage = 0.5,-0.5
    else:
        context.long_leverage, context.short_leverage =.5(.12/std_dev),-.5(.12/std_dev)

    # These are the securities that we are interested in trading each day.
#    context.security_list = context.output.index
     
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    
    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)

    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, context.long_leverage/len(context.longs))

    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, context.short_leverage/len(context.shorts))
    
   
def handle_data(context,data):
    """
    Called every minute.
    """
    pass

