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
import numpy as np
import pandas as pd

normal_days = 31
business_days = int(0.69 * normal_days)
weight_lower_quantile=0.10
weight_upper_quantile=0.90

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    attach_pipeline(make_pipeline(), 'ranked stocks')
    
    # Rebalance every month, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_month(), time_rules.market_open(hours=1))
     
    
class Returns(CustomFactor):
    """
    this factor outputs the returns over the period defined by 
    business_days, ending on the previous trading day, for every security.
    """
    window_length = business_days
    inputs = [USEquityPricing.close]
    def compute(self,today,assets,out,price):
        out[:] = (price[-1] - price[0]) / price[0] * 100

class StdDev(CustomFactor):
    def compute(self, today, asset_ids, out, values):
        # Calculates the column-wise standard deviation, ignoring NaNs
        out[:] = np.nanstd(values, axis=0)
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """

    returns = Returns()
    std_dev = StdDev(inputs=[returns], window_length=126)
    # Base universe set to the Q500US
    base_universe = Q1500US()

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest
     
    pipe = Pipeline(
        screen = base_universe,
        columns = {
            'close': yesterday_close,
            'std_dev': std_dev
            #'returns': returns
        }
    )
    return pipe
 
def before_trading_start(context, data):
    """
    Called every month before market open.
    """
    weight_lower_quantile=weight_lower_quantile*(.12/pipe[std_dev])
    weight_upper_quantile=1-weight_lower_quantile
    results = pipeline_output('ranked stocks').dropna()
    lower, upper = results['returns'].quantile([weight_lower_quantile, weight_upper_quantile])
    context.shorts = results[results['returns'] <= lower]
    context.longs = results[results['returns'] >= upper]
    
  
    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
     
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    for stock in context.shorts.index:
       if stock in data:
           order_target_percent(stock, context.shortleverage/                len(context.shorts))

    for stock in context.longs.index:
       if stock in data:
           order_target_percent(stock, context.longleverage /                len(context.longs))
            
    for stock in context.portfolio.positions:
       if stock not in context.longs.index and \
           stock not in context.shorts.index:
           order_target(stock, 0)
   
 
 
def handle_data(context,data):
    """
    Called every minute.
    """
    pass
y
