# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, DecimalParameter, stoploss_from_open, IntParameter, merge_informative_pair
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
from technical.indicators import ichimoku
logger = logging.getLogger(__name__)

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif
class AwesomeEWOLamboTDISMA(IStrategy):
    INTERFACE_VERSION: int = 3
    # xNighbloodx Natblida
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {"0": 0.02, "20": 0.015, "40": 0.013, "60": 0.012, "180": 0.012, }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.7
    # Optimal timeframe for the strategy
    timeframe = '5m'
    info_timeframe = '1h'
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 12,
        "ewo_high": 3.001,
        "ewo_high_2": -5.585,
        "ewo_middle": 2.500,
        "low_offset": 0.987,
        "low_offset_2": 0.942,
        "ewo_low": -5.289,
        "lookback_candles": 3,
        "profit_threshold": 1.01,
        "rsi_buy": 58,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }
    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=True)
    ewo_middle = DecimalParameter(1.2, 2.6, default=buy_params['ewo_middle'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)
    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)
     # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)
    # Lookback candles
    lookback_candles = IntParameter(
        1, 24, default=buy_params['lookback_candles'], space='buy', optimize=True)
    # Profit Threshold
    profit_threshold = DecimalParameter(1.00, 1.02,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)
    # trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012 #when profits reach 1% the trailing stop will be activated
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
    position_adjustment_enable = True
    threshold = 0.2

    slippage_protection = {
        'retries': 6,
        'max_slippage': -0.01
    }
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].ewm(span=4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi_14'].rolling(48).mean()
        
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)
           # Add dump protection
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['price_change_rolling_std'] = dataframe['price_change'].rolling(48).std()
        dataframe['dump_warn'] = np.where(dataframe['price_change'] < -3 * dataframe['price_change_rolling_std'], -1, 0)

        # Add price spike warning
        dataframe['price_spike_warn'] = np.where(dataframe['price_change'] >  3  * dataframe['price_change_rolling_std'], -1, 0)

        # Add volume/price ratio warning
        dataframe['volume_price_ratio'] = dataframe['volume'] / dataframe['close']
        dataframe['volume_price_ratio_change'] = dataframe['volume_price_ratio'].pct_change()
        dataframe['volume_price_ratio_warn'] = np.where(dataframe['volume_price_ratio_change'] > 3 * dataframe['volume_price_ratio_change'].ewm(span=48).std(), -1, 0)
        return dataframe
     # trailing stoploss hyperopt parameters
    # hard stoploss profit
    pHSL = DecimalParameter(-0.600, -0.080, default=stoploss, decimals=3,
                            space='sell', optimize=False, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.014, decimals=3,
                             space='sell', optimize=False, load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.024, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.022, decimals=3,
                             space='sell', optimize=False, load=True)
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
       # # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        # if current_profit < 0.001 and current_time - timedelta(minutes=600) > trade.open_date_utc:
        #     return -0.005

        return stoploss_from_open(sl_profit, current_profit)
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 7 days.
        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 7:
            return 'unclog'
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema100']) and (last_candle['close'] < last_candle['ema100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True   
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe)

        return informative_1h
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get the informative pair
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        # Merge the informative pair dataframe
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.info_timeframe, ffill=True)
        # Convert to Heikin Ashi candles
        heikin_ashi_df = heikinashi(dataframe)
        dataframe['ha_close'] = heikin_ashi_df['close']
        dataframe['ha_open'] = heikin_ashi_df['open']
        #EMA
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe)

        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

         # Stoch
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
     
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['buysignal'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)
        dataframe['sellsignal'] = (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        dataframe['difference_signal'] = (dataframe['ha_close'] - dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']).sub(dataframe['ha_close'].sub(dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']).mean()).div(dataframe['ha_close'].sub(dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']).std())
        dataframe['distance'] = (dataframe['ha_close'] - dataframe['buysignal']) / dataframe['ha_close'].std()
        dataframe['buy_signal_distance'] = dataframe['distance'].abs() < self.threshold
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

       
        # Step 1: Create a new column that checks if the current candle's close price is lower than the previous one
        dataframe['is_sinking'] = dataframe['close'] < dataframe['close'].shift(1)
        # Step 2: Create a new column that checks if the last 10 candles are sinking
        dataframe['sinking_10_candles'] = dataframe['is_sinking'].rolling(window=20).sum()

        # Step 1: Create a new column that checks if the current candle's close price is higher than the previous one
        dataframe['is_rising'] = dataframe['close'] > dataframe['close'].shift(1)

        # Step 2: Create a new column that checks if the last 10 candles are rising
        dataframe['rising_10_candles'] = dataframe['is_rising'].rolling(window=15).sum()
        

        # Add TDI (Traders Dynamic Index)
        tdi_df = tdi(dataframe['close'])
        dataframe['tdi_rsi'] = tdi_df['rsi']
        dataframe['tdi_signal'] = tdi_df['rsi_signal']
        # Add Awesome Oscillator
        dataframe['ao'] = awesome_oscillator(dataframe)
        # Add Simple Moving Average for comparison
        dataframe['sma'] = sma(dataframe['close'], window=14)
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe = self.pump_dump_protection(dataframe, metadata)
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions =[]

        buy1ewo = (
                (dataframe['rsi_fast'] <35)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi_14'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'enter_tag'] += 'buy_ewo_high_rsi_'
        conditions.append(buy1ewo)

        lambo2 = (
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'buy_lambo2_'
        conditions.append(lambo2)

        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()

        buy_singking = (
            qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open']) &
            (dataframe['sinking_10_candles'] > 10) &
            (last_candle['close'] > previous_candle['close']) &
            (dataframe['volume'] > 0) 
            )
        
        dataframe.loc[buy_singking, 'enter_tag'] += 'buy_singking_'
        
        conditions.append(buy_singking)

        buy_singking_2 = (
            qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open']) &
            (dataframe['sinking_10_candles'] > 5) &
            (last_candle['close'] > previous_candle['close']) &
            (dataframe['volume'] > 0) 
            )
        dataframe.loc[buy_singking_2, 'enter_tag'] += 'buy_singking_2'
        conditions.append(buy_singking_2)

        buy_scalp = ( (dataframe['rsi_14'] < 35) &  # RSI is below 30
                (dataframe['close'] < dataframe['sma5']) &  # Price is above SMA5
                (dataframe['macd'] > dataframe['macdsignal'])   # MACD line crosses above signal line)
        )
        dataframe.loc[buy_scalp, 'enter_tag'] += 'buy_scalp_'
        
        conditions.append(buy_scalp)

        buyonred= (
            (dataframe['ha_close'] < dataframe['sma']) &
            (dataframe['tdi_rsi'] < dataframe['tdi_signal']) &
            (dataframe['ao'] < 0) &
            (dataframe['difference_signal'] < -2.5) 
        )

        dataframe.loc[buyonred, 'enter_tag'] += 'buy_downtrend_sma_td_ao'
        conditions.append(buyonred)

        
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1
    
        dont_buy_conditions =[]
        dont_buy_conditions.append((dataframe['pnd_volume_warn'] == -1))
        dont_buy_conditions.append((dataframe['dump_warn'] == -1))
        dont_buy_conditions.append(
            (
                # don't buy if there isn't 4% profit to be made
                (dataframe['close_1h'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )
        if dont_buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, dont_buy_conditions), 'enter_long'] = 0
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
       
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()

        sell_rising = ((dataframe['rising_10_candles'] > 9) &
                       (last_candle['close'] > previous_candle['close']) &
                       (dataframe['difference_signal'] >= 3.5) 
                       )
        dataframe.loc[sell_rising, 'exit_tag'] += 'sell_rising_'
        conditions.append(sell_rising)

        sellsignal =(
            (dataframe['ha_close'] > dataframe['ha_open']) &
            (dataframe['difference_signal'] >= 3.5) 
        )

        dataframe.loc[sellsignal, 'exit_tag'] += 'sell_signal'
        conditions.append(sellsignal)


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
                    # logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    # logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None
        return None