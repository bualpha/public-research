"""
Implementation of the Risk-Managed Momentum Strategy as proposed in 'Momentum has its moments' by Pedro Barroso and Pedro Santa-Clara (http://docentes.fe.unl.pt/~psc/MomentumMoments.pdf)

"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.filters.morningstar import Q1500US
import numpy as np
import math
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
    
    # Record variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'my_pipeline')
         
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    # Base universe set to the Q500US.
    base_universe = Q1500US()
    
    # Cumulative returns of base universe stocks for months t-12 to t-2
    cumulative_returns = CumulativeReturns(mask=base_universe)
    
    cumulative_returns_decile = cumulative_returns.deciles()
    
    # Filter to select securities to long.
    longs = (cumulative_returns_decile.eq(9))
    
    # Filter to select securities to short.
    shorts = (cumulative_returns_decile.eq(0))
    
    # Filter for all securities that we want to trade.
    securities_to_trade = (shorts | longs)
     
    return Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts
        },
        screen=(securities_to_trade),
    )

class CumulativeReturns(CustomFactor):
    # Default inputs
    inputs = [USEquityPricing.close]
    # Start at month t-12.
    window_length = 365

    # Compute cumulative returns.
    def compute(self, today, assets, out, close):
        moving_cumulative_returns = []
        
        # End at month t-2.
        for i in range(305-1):
            pct_change = (close[i+1] - close[i]) / close[i]
            moving_cumulative_returns.append(pct_change)
            
        out[:] = moving_cumulative_returns[-1]
        
class RealizedVariance(CustomFactor):
    # Default inputs
    inputs = [USEquityPricing.close]
    # Previous six months
    window_length = 180
    
    # Compute realized variance.
    def compute(self, today, assets, out, close):
        sqrd_returns_sum = 0
        
        for i in range(180-1):
            pct_change = (close[i+1] - close[i]) / close[i]
            sqrd_returns_sum += pct_change**2
            
        out[:] = sqrd_returns_sum
     
def my_compute_weights(context):
    """
    Compute ordering weights.
    """
    # Compute target weights for our long positions and short positions.
    long_raw_weight = 0.12 / np.sqrt(RealizedVariance(mask=context.longs)) * context.longs
    
    short_raw_weight = 0.12 / np.sqrt(RealizedVariance(mask=context.shorts)) * context.shorts
    
    long_normalized_weight = ((long_raw_weight / long_raw_weight).abs().sum()) / 2
    
    short_normalized_weight = -1 * ((short_raw_weight / short_raw_weight).abs().sum()) / 2

    return long_normalized_weight, short_normalized_weight
 
def my_rebalance(context, data):
    """
    Rebalance every day 1 hour after market open.
    """
    # Gets our pipeline output every day.
    context.output = pipeline_output('my_pipeline')

    # Go long in securities for which the 'longs' value is True.
    context.longs = context.output[context.output['longs']].index.tolist()

    # Go short in securities for which the 'shorts' value is True.
    context.shorts = context.output[context.output['shorts']].index.tolist()

    # Calculate our target weights.
    long_weights, short_weights = my_compute_weights(context)

    # Place orders for each of our long securities.
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, long_weights[security])
    
    # Place orders for each of our short securities.
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, short_weights[security])
            
def my_record_vars(context, data):
    """
    Record variables at the end of each day.
    """
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    # Record our variables.
    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)
