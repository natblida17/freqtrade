# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
from freqtrade.persistence import Trade
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.vendor.qtpylib.indicators import heikinashi, tdi, awesome_oscillator, sma
import math
import logging
logger = logging.getLogger(__name__)
class Strategy001(IStrategy):
    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {"0": 0.02, "20": 0.015, "40": 0.014, "60": 0.012, "180": 0.015, }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.7
    # Optimal timeframe for the strategy
    timeframe = '5m'
    # trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    # run "populate_indicators" only for new candle
    process_only_new_candles = True
    startup_candle_count = 96
    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    #adjust trade position
    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4
    adjust_trade_position = True
    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi_14'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)
        return dataframe
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        max_reached_price = trade.max_rate  # Maximum price since the trade was opened
        trailing_percentage = 0.05  # Trailing 4% behind the maximum reached price
        new_stoploss = max_reached_price * (1 - trailing_percentage)
        return max(new_stoploss, self.stoploss)  # Ensure it's not below the initial stop loss
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < -0.03 and (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'
    def informative_pairs(self):
        # Define the informative pairs
        return []
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Convert to Heikin Ashi candles
        heikin_ashi_df = heikinashi(dataframe)
        dataframe['ha_close'] = heikin_ashi_df['close']
        dataframe['ha_open'] = heikin_ashi_df['open']
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        # Add TDI (Traders Dynamic Index)
        tdi_df = tdi(dataframe['close'])
        dataframe['tdi_rsi'] = tdi_df['rsi']
        dataframe['tdi_signal'] = tdi_df['rsi_signal']
        # Add Awesome Oscillator
        dataframe['ao'] = awesome_oscillator(dataframe)
        # Add Simple Moving Average for comparison
        dataframe['sma'] = sma(dataframe['close'], window=14)
        dataframe = self.pump_dump_protection(dataframe, metadata)
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions =[]
        buyonred = (
             qtpylib.crossed_below(dataframe['ema_14'], dataframe['ema20']) &
            (dataframe['ha_close'] < dataframe['ema_14']) &
            (dataframe['ha_open'] > dataframe['ha_close'])
        )
        dataframe.loc[buyonred, 'enter_tag'] += 'buy_downtrend_ema14_ema20'
        conditions.append(buyonred)
        buyongreen= (
            (dataframe['ha_close'] < dataframe['sma']) &
            (dataframe['tdi_rsi'] < dataframe['tdi_signal']) &
            (dataframe['ao'] < 0)
        )
        dataframe.loc[buyongreen, 'enter_tag'] += 'buy_downtrend_sma_td_ao'
        conditions.append(buyongreen)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1
        dont_buy_conditions =[]
        dont_buy_conditions.append((dataframe['pnd_volume_warn'] == -1))
        if dont_buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, dont_buy_conditions), 'enter_long'] = 0
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        sellwhengreenrise = (
            qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50']) &
                (dataframe['ha_close'] > dataframe['ema20']) &
                (dataframe['ha_open'] < dataframe['ha_close'])
        )
        dataframe.loc[sellwhengreenrise, 'exit_tag'] += 'sell_downtrend_ema20_ema50'
        conditions.append(sellwhengreenrise)
        sellwhenstartred = (
            (dataframe['ha_close'] > dataframe['sma']) &
            (dataframe['tdi_rsi'] > dataframe['tdi_signal']) &
            (dataframe['ao'] > 0)
        )
        dataframe.loc[sellwhenstartred, 'exit_tag'] += 'sell_downtrend_sma_td_ao'
        conditions.append(sellwhenstartred)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'exit_long'] = 1
        return dataframe
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None
        # credits to reinuvader for not blindly executing safety orders
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when it seems it's climbing back up
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None
        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1
        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None
        return None