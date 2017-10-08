"""
Implementation of the Risk-Managed Momentum Strategy as proposed in 'Momentum has its moments' by Pedro Barroso and Pedro Santa-Clara (http://docentes.fe.unl.pt/~psc/MomentumMoments.pdf)
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.filters.morningstar import Q1500US
import numpy as np


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
    pipe = Pipeline()

    # Base universe set to the Q1500US.
    base_universe = Q1500US()

    # Get returns of base universe stocks starting from 12 months ago (252 trading days ago).
    returns_12m = Returns(window_length=252, mask=base_universe)
    # Get returns of base universe stocks starting from 2 months ago (42 trading days ago).
    returns_2m = Returns(window_length=42, mask=base_universe)
    # Get returns of base universe stocks starting from 12 months and ending 2 months ago.
    returns_diff = returns_12m - returns_2m

    # Divide those returns into deciles.
    returns_diff_decile = returns_diff.deciles()

    # Filter to select securities to long
    longs = (returns_diff_decile.eq(9))
    pipe.add(longs, 'longs')

    # Filter to select securities to short
    shorts = (returns_diff_decile.eq(0))
    pipe.add(shorts, 'shorts')

    # Filter for all securities that we want to trade
    securities_to_trade = (longs | shorts)

    pipe.set_screen(securities_to_trade)

    return pipe


def my_compute_momentum(context, data):
    """
    Compute daily momentum (Winners Minus Losers) for the past six months (125 trading days).
    """
    WML_cumulative = []

    num_days = 125

    for i in range(1, num_days+1):
        winner_prices = data.history(context.longs, "price", i+1, "1d")
        winner_returns = winner_prices.pct_change().dropna().values

        loser_prices = data.history(context.shorts, "price", i+1, "1d")
        loser_returns = loser_prices.pct_change().dropna().values

        WML_current = winner_returns.mean() - loser_returns.mean()

        WML_cumulative.append(WML_current)

    return WML_cumulative


def my_compute_weights(context):
    """
    Compute ordering weights.
    """
    long_weight = context.long_leverage / len(context.longs)
    short_weight = context.short_leverage / len(context.shorts)

    return long_weight, short_weight


def my_rebalance(context, data):
    """
    Rebalance every day 1 hour after market open.
    """
    long_weight, short_weight = my_compute_weights(context)

    # Remove securities that no longer constitute momentum.
    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)

    # Place orders for each of our long securities.
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, long_weight)

    # Place orders for each of our short securities.
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, short_weight)


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    # Gets our pipeline output every day.
    context.output = pipeline_output('my_pipeline')

    # Go long in securities for which the 'longs' value is True.
    context.longs = context.output[context.output['longs']].index.tolist()
    # Go short in securities for which the 'shorts' value is True.
    context.shorts = context.output[context.output['shorts']].index.tolist()

    # Compute daily momentum for the past 6 months.
    momentum = my_compute_momentum(context, data)
    # Convert into a NumPy array for computations.
    momentum_np = np.array(momentum)

    # Calculate daily volatility of historical momentum.
    realized_vol_daily = np.std(momentum_np)
    # Annualize
    realized_vol_annual = realized_vol_daily * np.sqrt(252)

    # Target volatility of 12%
    target_vol = 0.12

    ratio = target_vol / realized_vol_annual

    # Scale leverage.
    leverage = 0.5 * ratio
    # Max leverage of 1.1
    if leverage > 0.55:
        leverage = 0.55

    context.long_leverage = leverage
    context.short_leverage = -1 * leverage


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
