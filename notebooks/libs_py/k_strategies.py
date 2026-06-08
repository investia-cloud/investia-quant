"""
k_strategies.py — Refactored from notebooks/libs/k_strategies.ipynb
"""

###############################################################################
# DEFINIZIONE STRATEGIE (strategy_*) e INDICATORI (calculate_*)
###############################################################################

# Adotta il comportamento futuro: nessun downcast silenzioso
pd.set_option('future.no_silent_downcasting', True)

def _safe_shift_fill_bool(mask: pd.Series, shift: int = 1) -> pd.Series:
    """
    Normalizza una maschera booleana evitando FutureWarning Pandas.
    - Converte 'object' -> tipo inferito (se possibile)
    - Usa BooleanDtype che supporta NA, poi fillna, poi cast a bool nativo
    """
    if mask.dtype == 'O':
        mask = mask.infer_objects(copy=False)
    mask = mask.astype('boolean', copy=False)     # estensione pandas con <NA>
    out = mask.shift(shift)                       # conserva il dtype 'boolean'
    out = out.fillna(False)                       # nessun downcast ambiguo
    return out.astype(bool, copy=False)           # compatibile con il framework



########################
# Strategy cts_dpo
########################

def calculate_cts(data, period=14):
    price_change = data['Close'].diff(periods=period)
    avg_price_change = price_change.rolling(window=period).mean()
    avg_true_range = (data['High'] - data['Low']).rolling(window=period).mean()
    trend_score = (avg_price_change / avg_true_range) * 100
    return trend_score

def calculate_dpo(data, period=14):
    sma = data['Close'].rolling(window=period).mean()
    dpo = data['Close'] - sma
    return dpo
    
strategy_cts_dpo_param_ranges = {
    'cts_period_range': range(5, 31),
    'dpo_period_range': range(5, 31),
    'cts_upper_range': [20,30],     # Qui puoi mettere più soglie, es. [20,30,40]
    'cts_lower_range': [-20,-30]     # idem, es. [-20,-30,-40]
}

def strategy_cts_dpo(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori CTS e DPO,
    calcolati su base giornaliera.
    """
    cts_period = params.get('cts_period_range')
    dpo_period = params.get('dpo_period_range')
    cts_upper  = params.get('cts_upper_range')
    cts_lower  = params.get('cts_lower_range')
    
    train_data  =  data.copy()

    train_data['CTS'] = calculate_cts(train_data, period=cts_period)
    train_data['DPO'] = calculate_dpo(train_data, period=dpo_period)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]
          
    entries = (train_data['CTS'] > cts_upper) & (train_data['DPO'] > 0)
    exits   = (train_data['CTS'] < cts_lower) & (train_data['DPO'] < 0)
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


########################
# Strategy heikin_ashi
########################


 # Function to calculate Heikin-Ashi candles
def calculate_heikin_ashi(df):
    ha_df = df.copy()
    
    # Calculate Heikin-Ashi Close
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # Calculate Heikin-Ashi Open
    ha_df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    # Calculate Heikin-Ashi High
    ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    # Calculate Heikin-Ashi Low
    ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    
    # Ensure that 'HA_Open' and 'HA_Close' are in the DataFrame
    return ha_df
    
strategy_heikin_ashi_param_ranges = {
    # Define dynamic ranges for Heikin Ashi signal shifts
    'shift_1_range' : range(1, 13, 3),  # Range for Entry signal shift
    'shift_2_range' : range(1, 13, 3),  # Range for Entry signal shift
    'shift_3_range' : range(1, 13, 3),  # Range for Exit signal shift
    'shift_4_range' : range(1, 13, 3)  # Range for Exit signal shift
}
   
def strategy_heikin_ashi(data, params, year=None):

    # Define dynamic ranges for Heikin Ashi signal shifts
    shift_1 = params.get('shift_1_range')  # Range for Entry signal shift
    shift_2 = params.get('shift_2_range')  # Range for Entry signal shift
    shift_3 = params.get('shift_3_range')  # Range for Exit signal shift
    shift_4 = params.get('shift_4_range')  # Range for Exit signal shift

    train_data  =  data.copy()

    # Calculate Heikin Ashi Candles on the training data
    train_data = calculate_heikin_ashi(train_data)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on Heikin Ashi
    entries = (train_data['HA_Close'].shift(shift_1) > train_data['HA_Open'].shift(shift_2)) & (train_data['HA_Low'] == train_data['HA_Open'])
    exits = (train_data['HA_Close'].shift(shift_3) < train_data['HA_Open'].shift(shift_4)) & (train_data['HA_High'] == train_data['HA_Open'])    

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
########################    
# Strategy vzo_kvo
########################

# Function to calculate Volume Zone Oscillator (VZO)
def calculate_vzo(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    volume_flow = df['Volume'] * np.sign(df['Close'].diff())

    vp = volume_flow.rolling(period).sum()
    total_volume = df['Volume'].rolling(period).sum()

    vzo = (vp / total_volume) * 100
    return vzo


# Function to calculate Klinger Volume Oscillator (KVO)
def calculate_kvo(df, short_period=34, long_period=55, signal_period=13):
    dm = (df['High'] + df['Low'] + df['Close']) / 3
    trend = np.where(dm > dm.shift(1), df['Volume'], -df['Volume'])
    kvo = pd.Series(trend, index=df.index).ewm(span=short_period).mean() - pd.Series(trend, index=df.index).ewm(span=long_period).mean()
    kvo_signal = kvo.ewm(span=signal_period).mean()
    return kvo, kvo_signal
    
strategy_vzo_kvo_param_ranges = {
    # Define dynamic ranges for VZO and KVO parameters
    'vzo_period_range' : range(5, 23, 4),  # Range for VZO periods
    'short_period_range' : range(20, 46, 6),  # Range for KVO short periods
    'long_period_range' : range(50, 116, 16),  # Range for KVO long periods
    'signal_period_range' : range(5, 23, 4)  # Range for KVO signal periods
}    

def strategy_vzo_kvo(data, params, year=None):
    
    # dynamic ranges for VZO and KVO parameters
    vzo_period = params.get('vzo_period_range')  # Range for VZO periods
    short_period = params.get('short_period_range')  # Range for KVO short periods
    long_period = params.get('long_period_range')  # Range for KVO long periods
    signal_period = params.get('signal_period_range')  # Range for KVO signal periods
    
    train_data  =  data.copy()
    
    # Calculate VZO and KVO on the training data
    train_data['VZO'] = calculate_vzo(train_data, period=vzo_period)
    train_data['KVO'], train_data['KVO_Signal'] = calculate_kvo(train_data, short_period, long_period, signal_period)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on VZO and KVO
    entries = (train_data['VZO'] > 0) & (train_data['KVO'] > train_data['KVO_Signal'])
    exits = (train_data['VZO'] < 0) & (train_data['KVO'] < train_data['KVO_Signal'])
        
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
##################
# Strategy dma  
#################

# Function to calculate Displaced Moving Averages (DMA)
def calculate_dma(df, period, shift):
    return df['Close'].rolling(window=period).mean().shift(shift)

# strategy_dma_param_ranges = {
#     # Define dynamic ranges for DMA parameters
#     'dma_fast_period_range' : range(1, 52, 5),  # Range for fast DMA periods
#     'dma_slow_period_range' : range(51, 102, 5),  # Range for slow DMA periods
#     'dma_fast_shift_range' : range(1, 21),  # Range for fast DMA shifts
#     'dma_slow_shift_range' : range(1, 21)  # Range for slow DMA shifts
# }  

# Ristretto
strategy_dma_param_ranges = {
    # Define dynamic ranges for DMA parameters
    'dma_fast_period_range' : range(1, 67, 16),  # Range for fast DMA periods
    'dma_slow_period_range' : range(51, 117, 16),  # Range for slow DMA periods
    'dma_fast_shift_range' : range(1, 21, 5),  # Range for fast DMA shifts
    'dma_slow_shift_range' : range(1, 21, 5)  # Range for slow DMA shifts
}   

def strategy_dma(data, params, year=None):
    
    # Define dynamic ranges for DMA parameters
    fast_period = params.get('dma_fast_period_range')  # Range for fast DMA periods
    slow_period = params.get('dma_slow_period_range')   # Range for slow DMA periods
    fast_shift =  params.get('dma_fast_shift_range')   # Range for fast DMA shifts
    slow_shift =  params.get('dma_slow_shift_range')   # Range for slow DMA shifts
    
    train_data  =  data.copy()
    
    # Calculate DMA indicators on the training data
    train_data['DMA_fast'] = calculate_dma(train_data, period=fast_period, shift=fast_shift)
    train_data['DMA_slow'] = calculate_dma(train_data, period=slow_period, shift=slow_shift)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on DMA
    entries = train_data['DMA_fast'] > train_data['DMA_slow']
    exits = train_data['DMA_fast'] < train_data['DMA_slow']
        
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
##########################
# Strategy ichimoku
##########################

# Function to calculate Ichimoku Cloud components
def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
    high_9 = df['High'].rolling(window=tenkan_period).max()
    low_9 = df['Low'].rolling(window=tenkan_period).min()
    df['Tenkan-sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=kijun_period).max()
    low_26 = df['Low'].rolling(window=kijun_period).min()
    df['Kijun-sen'] = (high_26 + low_26) / 2

    df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(kijun_period)

    high_52 = df['High'].rolling(window=senkou_span_b_period).max()
    low_52 = df['Low'].rolling(window=senkou_span_b_period).min()
    df['Senkou Span B'] = ((high_52 + low_52) / 2).shift(kijun_period)

    df['Chikou Span'] = df['Close'].shift(-kijun_period)

    return df
    
        
strategy_ichimoku_param_ranges = {
    # Define dynamic ranges for Ichimoku periods
    'tenkan_period_range' : range(7, 24, 4),  # Range for Tenkan-sen periods
    'kijun_period_range' : range(20, 60, 10),  # Range for Kijun-sen periods
    'senkou_span_b_period_range' : range(40, 66, 6)  # Range for Senkou Span B periods
}    

def strategy_ichimoku(data, params, year=None):
    
    # Define dynamic ranges for DMA parameters
    tenkan_period = params.get('tenkan_period_range')  # Range for Tenkan-sen periods
    kijun_period = params.get('kijun_period_range')   # Range for Kijun-sen periods
    senkou_span_b_period =  params.get('senkou_span_b_period_range')   # Range for Senkou Span B periods
    
    train_data  =  data.copy()

    # Calculate Ichimoku indicators on the training data
    train_data = calculate_ichimoku(train_data, tenkan_period, kijun_period, senkou_span_b_period)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on Ichimoku Cloud
    entries = (train_data['Close'] > train_data['Senkou Span A']) & (train_data['Close'] > train_data['Senkou Span B']) & (train_data['Tenkan-sen'] > train_data['Kijun-sen'])
    exits = (train_data['Close'] < train_data['Senkou Span A']) & (train_data['Close'] < train_data['Senkou Span B']) & (train_data['Tenkan-sen'] < train_data['Kijun-sen'])
    

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy tp_ma_crossover
############################


# Define function to calculate typical price
def typical_price(df):
    return (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

# Function to calculate moving averages
def calculate_tp_moving_average(df, short_window=50, long_window=200):
    df['TP'] = typical_price(df)  # Calculate typical price
    df['Short_MA'] = df['TP'].rolling(window=short_window).mean()
    df['Long_MA'] = df['TP'].rolling(window=long_window).mean()
    
    return df
    
strategy_tp_ma_crossover_param_ranges = {
    # Define dynamic ranges for short and long windows
    'short_window_range' : range(1, 133, 33),  # Range for short MA windows
    'long_window_range' : range(101, 233, 33)  # Range for long MA windows
}

def strategy_tp_ma_crossover(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori typical_price e 
    moving_average_crossover, calcolati su base giornaliera.
    """
    
    short_window  = params.get('short_window_range')
    long_window  = params.get('long_window_range')

    train_data  =  data.copy()
    
    # Calculate ... indicators on the training data
    train_data = calculate_tp_moving_average(train_data, 
                                             short_window=short_window,
                                             long_window=long_window)  
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on ....
    entries = train_data['Short_MA'] > train_data['Long_MA'] # Buy when short MA crosses above long MA
    exits = train_data['Short_MA'] < train_data['Long_MA']   # Sell when short MA crosses below long MA

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    

############################
# Strategy ko_bb
############################

def calculate_klinger_oscillator(df, fast_period=34, slow_period=55):
    """
    Calculate the Klinger Oscillator (KO).
    """
    # Calculate the price changes
    price_change = df['Close'].diff()
    # Calculate the volume changes
    volume_change = df['Volume'].diff()

    # Fast and slow moving averages of volume
    fast_ema = volume_change.ewm(span=fast_period).mean()
    slow_ema = volume_change.ewm(span=slow_period).mean()

    # Klinger Oscillator (KO) calculation
    klinger_oscillator = fast_ema - slow_ema
    return klinger_oscillator

# Function to calculate Bollinger Bands (BB)
def calculate_bollinger_bands_ko_bb(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    rolling_mean = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()

    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band
 
# # Default params range
# strategy_ko_bb_param_ranges = {
#     # Define dynamic ranges for Klinger Oscillator and Bollinger Bands periods
#     'fast_period_range' : range(20, 41, 2),  # Range for fast period of KO
#     'slow_period_range' : range(40, 61, 2),  # Range for slow period of KO
#     'period_range' : range(15, 30, 2),  # Range for Bollinger Bands period
#     'std_dev_range' : [1.5, 2, 2.5, 3.0],  # Range for Bollinger Bands standard deviation
#     'squeeze_window_range' : range(10, 31, 2)  # Range for BB Squeeze window size
# }

# Default params range (ristretto)
strategy_ko_bb_param_ranges = {
    # Define dynamic ranges for Klinger Oscillator and Bollinger Bands periods
    'fast_period_range' : range(20, 41, 5),  # Range for fast period of KO
    'slow_period_range' : range(40, 61, 5),  # Range for slow period of KO
    'period_range' : range(15, 25, 5),  # Range for Bollinger Bands period
    'std_dev_range' : [1.5, 2],  # Range for Bollinger Bands standard deviation
    'squeeze_window_range' : range(10, 31, 5)  # Range for BB Squeeze window size
}
  
def strategy_ko_bb(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    fast_period = params.get('fast_period_range')
    slow_period = params.get('slow_period_range')
    period = params.get('period_range')
    std_dev = params.get('std_dev_range')
    squeeze_window = params.get('squeeze_window_range')
        
    train_data  =  data.copy()
    
    # Calculate Klinger Oscillator and Bollinger Bands on the training data
    train_data['Klinger_Oscillator'] = calculate_klinger_oscillator(train_data, fast_period, slow_period)
    train_data['Upper_Band'], train_data['Lower_Band'] = calculate_bollinger_bands_ko_bb(train_data, period, std_dev)

    # Bollinger Bands Squeeze: Calculate the range between the bands and check if it’s at a 20-period low
    train_data['BB_Squeeze'] = train_data['Upper_Band'] - train_data['Lower_Band']
    train_data['BB_Squeeze_Low'] = train_data['BB_Squeeze'].rolling(window=squeeze_window).min()
 
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Entry Condition: BB Squeeze + KO rising
    entries = (train_data['BB_Squeeze'] == train_data['BB_Squeeze_Low']) & (train_data['Klinger_Oscillator'] > 0)
    exits = (train_data['Klinger_Oscillator'] < 0) & (train_data['Close'] > train_data['Upper_Band'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


###############################
# Strategy sma_mf (muktiframe)  
###############################

# Function to calculate moving average crossover
def calculate_sma(df, short_window=50, long_window=200):
    short_sma = df['Close'].rolling(window=short_window).mean()
    long_sma = df['Close'].rolling(window=long_window).mean()
    return short_sma, long_sma

# strategy_sma_mf_param_ranges = {
#     # Define dynamic ranges for daily and weekly SMA parameters
#     'daily_short_window_range' : range(1, 101, 5),  # Range for short window (daily)
#     'daily_long_window_range' : range(101, 201, 5),  # Range for long window (daily)
#     'weekly_short_window_range' : range(1, 26, 2),  # Range for short window (weekly)
#     'weekly_long_window_range' : range(26, 51, 2)  # Range for long window (weekly)    
# }

# ristretto
strategy_sma_mf_param_ranges = {
    # Define dynamic ranges for daily and weekly SMA parameters
    'daily_short_window_range' : range(1, 127, 31),  # Range for short window (daily)
    'daily_long_window_range' : range(101, 227, 31),  # Range for long window (daily)
    'weekly_short_window_range' : range(1, 27, 6),  # Range for short window (weekly)
    'weekly_long_window_range' : range(26, 52, 6)  # Range for long window (weekly)    
}

def strategy_sma_mf(data, params, year=None):
    
    # Define dynamic ranges for DMA parameters
    daily_short_window = params.get('daily_short_window_range')  # Range for fast DMA periods
    daily_long_window = params.get('daily_long_window_range')   # Range for slow DMA periods
    weekly_short_window =  params.get('weekly_short_window_range')   # Range for fast DMA shifts
    weekly_long_window =  params.get('weekly_long_window_range')   # Range for slow DMA shifts
    
    
    train_data  =  data.copy()
    
    # Calculate SMA for weekly and daily data
    weekly_data = train_data.resample('W').last()  # Resample weekly data
    weekly_data['Short_SMA'], weekly_data['Long_SMA'] = calculate_sma(weekly_data, weekly_short_window, weekly_long_window)
    daily_data = train_data
    daily_data['Short_SMA'], daily_data['Long_SMA'] = calculate_sma(daily_data, daily_short_window, daily_long_window)

    # Forward-fill weekly SMA to daily data
    weekly_data = weekly_data.reindex(daily_data.index, method='ffill')
    daily_data['HTF_Trend'] = weekly_data['Short_SMA'] > weekly_data['Long_SMA']  # Weekly trend confirmation
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        daily_data = daily_data[daily_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = (daily_data['HTF_Trend']) & (daily_data['Short_SMA'] > daily_data['Long_SMA'])
    exits = (~daily_data['HTF_Trend']) & (daily_data['Short_SMA'] < daily_data['Long_SMA'])
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy cci_vortex
############################

# Function to calculate Commodity Channel Index (CCI)
def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3  # Typical Price
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci

# Function to calculate Vortex Indicator (VI)
def calculate_vortex(df, period=14):
    tr = np.maximum(df['High'] - df['Low'], 
                    np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    vm_plus = abs(df['High'] - df['Low'].shift(1))
    vm_minus = abs(df['Low'] - df['High'].shift(1))
    vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    return vi_plus, vi_minus
    
strategy_cci_vortex_param_ranges = {
    # Define dynamic ranges for CCI and Vortex periods
    'cci_period_range' : range(10, 63, 13),  #  Range for CCI periods
    'vortex_period_range' : range(10, 63, 13)  # Range for Vortex periods   
}
   
def strategy_cci_vortex(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    
    cci_period = params.get('cci_period_range')
    vortex_period = params.get('vortex_period_range')
    
    train_data  =  data.copy()
    
    # Calculate CCI and Vortex indicators on the training data
    train_data['CCI'] = calculate_cci(train_data, cci_period)
    train_data['VI+'], train_data['VI-'] = calculate_vortex(train_data, vortex_period)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on CCI and Vortex Indicator
    entries = ((train_data['CCI'] > 100) & (train_data['VI+'] > train_data['VI-'])) | \
              ((train_data['CCI'] < -100) & (train_data['VI-'] > train_data['VI+']))
    exits = ((train_data['VI+'] < train_data['VI-']) & (train_data['CCI'] > 0)) | \
            ((train_data['VI-'] < train_data['VI+']) & (train_data['CCI'] < 0))

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

    
############################
# Strategy macd_williams
############################
    
# Function to calculate MACD
def calculate_macd_williams(df, short_window=12, long_window=26, signal_window=9):
    df['MACD'] = df['Close'].ewm(span=short_window, min_periods=1).mean() - df['Close'].ewm(span=long_window, min_periods=1).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, min_periods=1).mean()
    return df

# Function to calculate Williams %R
def calculate_williams_r(df, period=14):
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    df['Williams_%R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return df

    
strategy_macd_williams_param_ranges = {
    # Define dynamic ranges for MACD and Williams %R parameters
    'short_window_range' : range(5, 18, 3),  # Range for MACD short window
    'long_window_range' : range(16, 34, 4),  # Range for MACD long window
    'signal_window_range' : range(5, 18, 3),  # Range for MACD signal window
    'williams_r_period_range' : range(10, 23, 3),  # Range for Williams %R period
    'williams_r_entry_thresholds_range' : [-10, -20, -30],  # Entry thresholds for Williams %R
    'williams_r_exit_thresholds_range' : [-70, -80, -90]  # Exit thresholds for Williams %R
}

    
def strategy_macd_williams(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    short_window = params.get('short_window_range')
    long_window = params.get('long_window_range')
    signal_window = params.get('signal_window_range')
    williams_r_period = params.get('williams_r_period_range')
    williams_r_entry_thresholds = params.get('williams_r_entry_thresholds_range')
    williams_r_exit_thresholds = params.get('williams_r_exit_thresholds_range')

    train_data  =  data.copy()
    
    # Calculate MACD and Williams %R on the training data
    train_data = calculate_macd_williams(train_data, short_window, long_window, signal_window)
    train_data = calculate_williams_r(train_data, williams_r_period)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on MACD and Williams %R
    entries = (train_data['MACD'] > train_data['MACD_Signal']) & (train_data['Williams_%R'] < williams_r_entry_thresholds)
    exits = (train_data['MACD'] < train_data['MACD_Signal']) & (train_data['Williams_%R'] < williams_r_exit_thresholds)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy tema_heikin_ashi
############################

# Function to calculate Triple Exponential Moving Average (TEMA)
def calculate_tema(df, period=30):
    ema1 = df['Close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    tema = 3 * (ema1 - ema2) + ema3
    return tema

 
# # Function to calculate Heikin-Ashi candles: gia' definita

# Pre function
# strategy_tema_heikin_ashi_prefunction=calculate_heikin_ashi

strategy_tema_heikin_ashi_param_ranges = {
    # Define dynamic ranges for MACD e TEMA parameters
    'entry_shift_1_range' : range(1, 13, 4),
    'entry_shift_2_range' : range(1, 12, 3),
    'exit_shift_1_range' : range(1, 12, 3),
    'exit_shift_2_range' : range(1, 12, 3),
    'tema_period_range' : range(1, 39, 9)
}

def strategy_tema_heikin_ashi(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    entry_shift_1 = params.get('entry_shift_1_range')
    entry_shift_2 = params.get('entry_shift_2_range')
    exit_shift_1 = params.get('exit_shift_1_range')
    exit_shift_2 = params.get('exit_shift_2_range')
    tema_period = params.get('tema_period_range')
        

    train_data  =  data.copy()
    
    # # Apply Heikin-Ashi calculation (il prefunction non serve ma puo' tornare utile)
    train_data = calculate_heikin_ashi(train_data)

    # Calculate ... indicators on the training data
    train_data['TEMA'] = calculate_tema(train_data, tema_period)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Use Heikin-Ashi Close for entry and exit signals
    entries = (train_data['HA_Close'].shift(entry_shift_1) > train_data['TEMA'].shift(entry_shift_2))
    exits = (train_data['HA_Close'].shift(exit_shift_1) < train_data['TEMA'].shift(exit_shift_2))
   
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
####################################
# Strategy macd_zero_line_rejection
####################################

# Function to calculate MACD and Histogram
def calculate_macd_zero_line_rejection(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

# Function for Zero-Line Rejection with Optimizable Shifts
def zero_line_rejection(histogram, shift_1, shift_2, shift_3):
    return (histogram.shift(shift_1) < 0) & (histogram.shift(shift_2) > histogram.shift(shift_3)) & (histogram < 0)

def bearish_zero_line_rejection(histogram, shift_1, shift_2, shift_3):
    return (histogram.shift(shift_1) > 0) & (histogram.shift(shift_2) < histogram.shift(shift_3)) & (histogram > 0)   

strategy_macd_zero_line_rejection_param_ranges = {
    # Define dynamic ranges for MACD periods, shift values, and volume rolling window
    'short_window_range' : range(8, 20, 6),
    'long_window_range' : range(20, 50, 15),
    'signal_window_range' : range(5, 18, 6),
    'shift_1_range' : range(1, 13, 6),
    'shift_2_range' : range(1, 13, 6),
    'shift_3_range' : range(1, 13, 6),
    'volume_window_range' : range(1, 13, 6)  # Volume rolling window to optimize
}
    
def strategy_macd_zero_line_rejection(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    short_window = params.get('short_window_range')
    long_window = params.get('long_window_range')
    signal_window = params.get('signal_window_range')
    shift_1 = params.get('shift_1_range')
    shift_2 = params.get('shift_2_range')
    shift_3 = params.get('shift_3_range')
    volume_window = params.get('volume_window_range')

    train_data  =  data.copy()
    
    # Calculate MACD on the training data
    train_data['MACD'], train_data['Signal'], train_data['Histogram'] = \
    calculate_macd_zero_line_rejection(train_data, short_window, long_window, signal_window)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on Zero-Line Rejection and volume optimization
    entries = bearish_zero_line_rejection(train_data['Histogram'], shift_1, shift_2, shift_3) & (train_data['Volume'] > train_data['Volume'].rolling(volume_window).mean())
    exits = zero_line_rejection(train_data['Histogram'], shift_1, shift_2, shift_3) & (train_data['Volume'] > train_data['Volume'].rolling(volume_window).mean())    
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy ichimoku_cloud
############################


# Function to calculate Ichimoku Cloud components
def calculate_ichimoku_cloud(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
    """
    Calculate Ichimoku Cloud components.
    """
    # Tenkan-Sen (Conversion Line)
    df['Tenkan_Sen'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2
    
    # Kijun-Sen (Base Line)
    df['Kijun_Sen'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=senkou_span_b_period).max() + df['Low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Line)
    df['Chikou_Span'] = df['Close'].shift(-kijun_period)
    
    return df

strategy_ichimoku_cloud_param_ranges = {
    # Define dynamic ranges for Ichimoku parameters
    'tenkan_period_range' : range(5, 38, 8),  # Tenkan-Sen period range
    'kijun_period_range' : range(5, 38, 8),  # Kijun-Sen period range
    'senkou_span_b_period_range' : range(40, 66, 6)  # Senkou Span B period range
}

def strategy_ichimoku_cloud(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    tenkan_period = params.get('tenkan_period_range')
    kijun_period = params.get('kijun_period_range')
    senkou_span_b_period = params.get('senkou_span_b_period_range')

    train_data  =  data.copy()
                
    # Calculate Ichimoku Cloud components on the training data
    train_data = calculate_ichimoku_cloud(train_data, tenkan_period, kijun_period, senkou_span_b_period)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on Ichimoku Cloud breakout strategy
    entries = (train_data['Close'] > train_data['Senkou_Span_A']) | \
              (train_data['Close'] > train_data['Senkou_Span_B']) | \
              (train_data['Tenkan_Sen'] > train_data['Kijun_Sen'])
    exits = (train_data['Close'] < train_data['Senkou_Span_A']) & \
              (train_data['Close'] < train_data['Senkou_Span_B']) & \
              (train_data['Tenkan_Sen'] < train_data['Kijun_Sen'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

    
############################
# Strategy gmma
############################

# Function to calculate Guppy MMA (GMMA)
def calculate_gmma(df, short_periods=range(3, 16, 3), long_periods=range(30, 51, 5)):
    for period in short_periods:
        df[f'GMMA_Short_{period}'] = df['Close'].ewm(span=period).mean()

    for period in long_periods:
        df[f'GMMA_Long_{period}'] = df['Close'].ewm(span=period).mean()

    return df

# strategy_gmma_param_ranges = {
#     # Define dynamic ranges for GMMA periods
#     'short_period_start_range' : range(1, 10, 2),
#     'short_period_end_range' : range(10, 31, 2),
#     'short_period_step_range' : [1, 2, 3],
#     'long_period_start_range' : range(31, 40, 2),
#     'long_period_end_range' : range(50, 71, 2),
#     'long_period_step_range' : [1, 2, 3]
# }

# ristretto
strategy_gmma_param_ranges = {
    # Define dynamic ranges for GMMA periods
    'short_period_start_range' : range(1, 10, 2),
    'short_period_end_range' : range(10, 31, 5),
    'short_period_step_range' : [1, 2, 3],
    'long_period_start_range' : range(31, 40, 2),
    'long_period_end_range' : range(50, 71, 5),
    'long_period_step_range' : [1, 2, 3]
}


def strategy_gmma(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    short_period_start = params.get('short_period_start_range')
    short_period_end = params.get('short_period_end_range')
    short_period_step = params.get('short_period_step_range')
    long_period_start = params.get('long_period_start_range')
    long_period_end = params.get('long_period_end_range')
    long_period_step = params.get('long_period_step_range')


    short_periods = range(short_period_start, short_period_end, short_period_step)
    long_periods = range(long_period_start, long_period_end, long_period_step)

    train_data  =  data.copy()


    # Calculate GMMA on the training data
    train_data = calculate_gmma(train_data, short_periods, long_periods)

    # Define GMMA Compression (entry condition)
    short_gmma_cols = [f'GMMA_Short_{p}' for p in short_periods]
    long_gmma_cols = [f'GMMA_Long_{p}' for p in long_periods]

    train_data['GMMA_Compression'] = train_data[short_gmma_cols].mean(axis=1) < train_data[long_gmma_cols].mean(axis=1)

        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on ....
    entries = train_data['GMMA_Compression']
    exits = train_data[short_gmma_cols[-1]] < train_data[long_gmma_cols[-1]]

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy trend_following
############################

def calculate_tf_sma(df, sma_window):
    sma = df['Close'].rolling(window=sma_window).mean()
    return sma
    
strategy_trend_following_param_ranges = {
    # Define dynamic ranges for Timeframe DMA parameters
    'timeframe_range' : ['1D', '1W', '1M'],
    'sma_window_range' : [3, 4, 5, 10]
}

def strategy_trend_following(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.

    
    Buy when the price closes above the high of the past 20 days.
    Sell when the price closes below the low of the past 20 days.

    or

    Buy when the monthly candle closes above the 10-period moving average.
    Sell when the monthly candle closes below the 10-period moving average.


    """
    
    # Define dynamic ranges for Timeframe DMA parameters
    timeframe = params.get('timeframe_range')
    sma_window = params.get('sma_window_range')
    
    
    daily_data  = data.copy()
    train_data = data.copy()

    # Resampling
    if timeframe=='1D':
        pass
    elif timeframe=='1W':
        train_data = train_data.resample('W').last()  # Resample weekly data
    elif timeframe=='1M':
        train_data = train_data.resample('ME').last()
    else:
        return None,None
        
    # Calculate SMA for weekly and daily data
    train_data['SMA'] = calculate_tf_sma(train_data,sma_window)
    
    # Forward-fill train data to daily data
    train_data = train_data.reindex(daily_data.index, method='ffill')
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = train_data['Close'].shift(1) > train_data['SMA']
    exits =   train_data['Close'].shift(1) < train_data['SMA']


    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy mean_reversion
############################
    
strategy_mean_reversion_param_ranges = {
    # Define dynamic ranges for Timeframe DMA parameters
    'timeframe_range' : ['1D', '1W', '1M'],
    'sma_window_range' : [3, 4, 5, 10]
}

def strategy_mean_reversion(data, params, year=None):
    """
    Genera segnali (entries, exits) usando un approccio mean-reversion,
    basato sulla deviazione del prezzo dalla media mobile.
    
    Buy when the price closes significantly below the moving average.
    Sell when the price reverts to or above the moving average.
    """
    
    # Definizione dei parametri
    timeframe = params.get('timeframe_range', '1D')
    sma_window = params.get('sma_window_range', 20)
    
    daily_data = data.copy()
    train_data = data.copy()
    
    # Resampling
    if timeframe == '1D':
        pass
    elif timeframe == '1W':
        train_data = train_data.resample('W').last()
    elif timeframe == '1M':
        train_data = train_data.resample('ME').last()
    else:
        return None, None
    
    # Calcolo della SMA
    train_data['SMA'] = calculate_tf_sma(train_data, sma_window)
    
    # Forward-fill train data to daily data
    train_data = train_data.reindex(daily_data.index, method='ffill')
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]
    
    # Generazione dei segnali mean-reversion basati sulla SMA
    entries = train_data['Close'].shift(1) < train_data['SMA']
    exits = train_data['Close'].shift(1) > train_data['SMA']
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
##############################
# Strategy bb_trend_following
##############################


def calculate_bollinger_bands_bb(df, sma_window, bollinger_mult):
    sma = df['Close'].rolling(window=sma_window).mean()
    std = df['Close'].rolling(window=sma_window).std()
    upper_band = sma + (bollinger_mult * std)
    lower_band = sma - (bollinger_mult * std)
    return sma, upper_band, lower_band
    
    
strategy_bb_trend_following_param_ranges = {
    # Define dynamic ranges for Timeframe DMA parameters
    'timeframe_range' : ['1D', '1W', '1M'],
    'sma_window_range' : [3, 4, 5, 10, 12, 20, 30],
    'bollinger_mult_range' : [2, 3, 4]
}

def strategy_bb_trend_following(data, params, year=None):
    """
    Genera segnali (entries, exits) usando un approccio trend-following,
    basato sulle Bande di Bollinger.
    
    Buy when the price closes above the upper Bollinger Band.
    Sell when the price closes below the lower Bollinger Band.
    """
    
    timeframe = params.get('timeframe_range', '1D')
    sma_window = params.get('sma_window_range', 20)
    bollinger_mult = params.get('bollinger_mult', 2)
    
    daily_data = data.copy()
    train_data = data.copy()
    
    # Resampling
    if timeframe == '1D':
        pass
    elif timeframe == '1W':
        train_data = train_data.resample('W').last()
    elif timeframe == '1M':
        train_data = train_data.resample('ME').last()
    else:
        return None, None
    
    # Calcolo delle Bande di Bollinger
    train_data['SMA'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_bollinger_bands_bb(train_data, sma_window, bollinger_mult)
    
    # Forward-fill train data to daily data
    train_data = train_data.reindex(daily_data.index, method='ffill')
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]
    
    # Generazione dei segnali trend-following basati sulle Bande di Bollinger
    entries = train_data['Close'].shift(1) > train_data['Upper_Band']
    exits = train_data['Close'].shift(1) < train_data['Lower_Band']
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

#####################################
# Strategy strategy_bb_mean_reversion
#####################################

strategy_bb_mean_reversion_param_ranges = {
    # Define dynamic ranges for Timeframe DMA parameters
    'timeframe_range' : ['1D', '1W', '1M'],
    'sma_window_range' : [3, 4, 5, 10, 12, 20, 30],
    'bollinger_mult_range' : [2, 3, 4]
}

def strategy_bb_mean_reversion(data, params, year=None):
    """
    Genera segnali (entries, exits) usando un approccio mean-reversion,
    basato sulle Bande di Bollinger.
    
    Buy when the price closes below the lower Bollinger Band.
    Sell when the price closes above the upper Bollinger Band.
    """
    
    # timeframe = params.get('timeframe_range', '1D')
    timeframe = params.get('timeframe_range', ['1D', '1W', '1M'])
    sma_window = params.get('sma_window_range', [3, 4, 5, 10, 12, 20, 30])
    bollinger_mult = params.get('bollinger_mult_range', [2, 3, 4])
    
    daily_data = data.copy()
    train_data = data.copy()
    
    # Resampling
    if timeframe == '1D':
        pass
    elif timeframe == '1W':
        train_data = train_data.resample('W').last()
    elif timeframe == '1M':
        train_data = train_data.resample('ME').last()
    else:
        return None, None
    
    # Calcolo delle Bande di Bollinger
    train_data['SMA'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_bollinger_bands_bb(train_data, sma_window, bollinger_mult)
    
    # Forward-fill train data to daily data
    train_data = train_data.reindex(daily_data.index, method='ffill')
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]
    
    # Generazione dei segnali mean-reversion basati sulle Bande di Bollinger
    entries = train_data['Close'].shift(1) < train_data['Lower_Band']
    exits = train_data['Close'].shift(1) > train_data['Upper_Band']
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy hma_heikin_ashi
############################

# Function to calculate Hull Moving Average (HMA)
def calculate_hma(df, period=30):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    # Calculate WMA for the given period
    wma_half = df['Close'].ewm(span=half_length).mean()
    wma_full = df['Close'].ewm(span=period).mean()

    # Hull Moving Average formula
    hma = wma_half * 2 - wma_full
    hma = hma.ewm(span=sqrt_length).mean()  # Apply WMA to the result
    return hma

# Function to calculate Heikin-Ashi candles
# Gia' definita

# Default params range
strategy_hma_heikin_ashi_param_ranges = {
    'entry_shift_1_range' : range(1, 13, 4),
    'entry_shift_2_range' : range(1, 12, 3),
    'exit_shift_1_range' : range(1, 12, 3),
    'exit_shift_2_range' : range(1, 12, 3),
    'hma_period_range' : range(5, 38, 8)
}

def strategy_hma_heikin_ashi(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
        
    entry_shift_1 = params.get('entry_shift_1_range')
    entry_shift_2 = params.get('entry_shift_2_range')
    exit_shift_1 = params.get('exit_shift_1_range')
    exit_shift_2 = params.get('exit_shift_2_range')
    hma_period = params.get('hma_period_range')
        
    train_data  =  data.copy()
    
    # Apply Heikin-Ashi calculation (il prefunction non serve ma puo' tornare utile)
    train_data = calculate_heikin_ashi(train_data)
    
    # Calculate ... indicators on the training data
    train_data['HMA'] = calculate_hma(train_data, hma_period)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Use Heikin-Ashi Close for entry and exit signals
    entries = (train_data['HA_Close'].shift(entry_shift_1) > train_data['HMA'].shift(entry_shift_2))
    exits = (train_data['HA_Close'].shift(exit_shift_1) < train_data['HMA'].shift(exit_shift_2))
   
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
###############################
# Strategy macd_daily_weekly
###############################

# Function to calculate MACD using (Open + High + Low + Close) / 4
def calculate_macd_daily_weekly(df, short_window=12, long_window=26, signal_window=9):
    typical_price = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    short_ema = typical_price.ewm(span=short_window, adjust=False).mean()
    long_ema = typical_price.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line
    

# Default params range
strategy_macd_daily_weekly_param_ranges = {
    # Define dynamic ranges for MACD parameters
    'short_window_range' : range(5, 17, 3),  # Range for short window
    'long_window_range' : range(20, 33, 3),  # Range for long window
    'signal_window_range' : range(5, 17, 3)  # Range for signal window
}

def strategy_macd_daily_weekly(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    short_window = params.get('short_window_range')
    long_window = params.get('long_window_range')
    signal_window = params.get('signal_window_range')

    train_data  =  data.copy()
    
    # Calculate MACD for weekly and daily data
    weekly_data = train_data.resample('W').last()  # Resample weekly data
    weekly_data['MACD'], weekly_data['Signal'] = calculate_macd_daily_weekly(weekly_data, short_window, long_window, signal_window)
    daily_data = train_data
    daily_data['MACD'], daily_data['Signal'] = calculate_macd_daily_weekly(daily_data, short_window, long_window, signal_window)

    # Forward-fill weekly MACD to daily data
    weekly_data = weekly_data.reindex(daily_data.index, method='ffill')
    daily_data['HTF_Trend'] = weekly_data['MACD'] < weekly_data['Signal']  # Weekly trend confirmation

        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        daily_data = daily_data[daily_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = (daily_data['HTF_Trend']) & (daily_data['MACD'] > daily_data['Signal'])
    exits = (~daily_data['HTF_Trend']) & (daily_data['MACD'] < daily_data['Signal'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

#################################
# Strategy macd_mf (multiframe ) 
#################################

# Function to calculate MACD
def calculate_macd_mf(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

# Default params range
strategy_macd_mf_param_ranges = {
    # Define dynamic ranges for MACD parameters
    'short_window_range' : range(5, 17, 3),  # Range for short window
    'long_window_range' : range(20, 33, 3),  # Range for long window
    'signal_window_range' : range(5, 17, 3)  # Range for signal window
}

def strategy_macd_mf(data, params, year=None):
    
    # Define dynamic ranges for DMA parameters
    short_window = params.get('short_window_range')  # Range for fast DMA periods
    long_window = params.get('long_window_range')   # Range for slow DMA periods
    signal_window =  params.get('signal_window_range')   # Range for fast DMA shifts
    
    
    train_data  =  data.copy()
    
    # Calculate MACD for weekly and daily data
    weekly_data = train_data.resample('W').last()  # Resample weekly data
    weekly_data['MACD'], weekly_data['Signal'] = calculate_macd_mf(weekly_data, short_window, long_window, signal_window)
    daily_data = train_data
    daily_data['MACD'], daily_data['Signal'] = calculate_macd_mf(daily_data, short_window, long_window, signal_window)
  


    # Forward-fill weekly MACD to daily data
    weekly_data = weekly_data.reindex(daily_data.index, method='ffill')
    daily_data['HTF_Trend'] = weekly_data['MACD'] < weekly_data['Signal']  # Weekly trend confirmation
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        daily_data = daily_data[daily_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = (daily_data['HTF_Trend']) & (daily_data['MACD'] > daily_data['Signal'])
    exits = (~daily_data['HTF_Trend']) & (daily_data['MACD'] < daily_data['Signal'])
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy hma_atr
############################

# Function to calculate Weighted Moving Average (WMA)
def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Function to calculate Hull Moving Average (HMA)
def hma_atr(series, period):

    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    wma_half = wma(series, half_length)
    wma_full = wma(series, period)
    hma_series = wma(2 * wma_half - wma_full, sqrt_length)

    return hma_series

# Function to calculate ATR
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = tr.rolling(window=period).mean()
    return atr 


# Default params range
strategy_hma_atr_param_ranges = {
    # Define dynamic ranges for HMA and ATR parameters
    'hma_short_period_range' : range(1, 25, 8),  # Range for short-term HMA periods
    'hma_long_period_range' : range(21, 42, 10),  # Range for long-term HMA periods
    'atr_period_range' : range(10, 34, 12),  # Range for ATR periods
    'atr_multiplier_range' : [1.5, 2, 2.5],  # ATR multiplier range
    'entry_shift_range' : range(1, 12, 3),  # Range for entry shift
    'exit_shift_range' : range(1, 12, 3)  # Range for exit shift
}
  
def strategy_hma_atr(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    hma_short_period = params.get('hma_short_period_range')
    hma_long_period = params.get('hma_long_period_range')
    atr_period = params.get('atr_period_range')
    atr_multiplier = params.get('atr_multiplier_range')
    entry_shift = params.get('entry_shift_range')
    exit_shift = params.get('exit_shift_range')
    
    train_data  =  data.copy()
    
    # Calculate HMA and ATR indicators on the training data
    train_data['HMA_Short'] = hma_atr(train_data['Close'], hma_short_period)
    train_data['HMA_Long'] = hma_atr(train_data['Close'], hma_long_period)
    train_data['ATR'] = calculate_atr(train_data, atr_period)


    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on HMA crossover & ATR
    entries = (train_data['HMA_Short'] > train_data['HMA_Long']) & \
              (train_data['Close'] > train_data['Close'].shift(entry_shift) + atr_multiplier * train_data['ATR'])
    exits = (train_data['HMA_Short'] < train_data['HMA_Long']) & \
            (train_data['Close'] < train_data['Close'].shift(exit_shift) - atr_multiplier * train_data['ATR'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
   
    
############################
# Strategy fractal_vortex
############################
 
# Function to calculate the Vortex Indicator (VI+ and VI-)
def calculate_fractal_vortex(data, period):
    """
    Calculate the Vortex Indicator (VI+ and VI-).
    """
    high_diff = data['High'].diff()
    low_diff = data['Low'].diff()

    # True range
    tr = np.maximum(np.maximum(data['High'] - data['Low'], (data['High'] - data['Close'].shift()).abs()), (data['Low'] - data['Close'].shift()).abs())

    # Vortex Indicator (+ and -)
    vi_plus = (high_diff.where(high_diff > 0, 0).rolling(window=period).sum() / tr.rolling(window=period).sum()) * 100
    vi_minus = (low_diff.where(low_diff > 0, 0).rolling(window=period).sum() / tr.rolling(window=period).sum()) * 100
    return vi_plus, vi_minus

# Function to calculate Fractal Chaos Bands (Upper Band and Lower Band)
def calculate_fractal_bands(data, period):
    """
    Calculate Fractal Chaos Bands (Upper Band and Lower Band).
    """
    upper_band = data['Close'].rolling(window=period).max()
    lower_band = data['Close'].rolling(window=period).min()
    return upper_band, lower_band

# Default params range
strategy_fractal_vortex_param_ranges = {
    'vortex_period_range' : range(5, 31),
    'fractal_period_range' : range(5, 31)
}

def strategy_fractal_vortex(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    vortex_period = params.get('vortex_period_range')
    fractal_period = params.get('fractal_period_range')
    
    train_data  =  data.copy()
    
    # Calculate indicators on the training data
    train_data['VI+'], train_data['VI-'] = calculate_fractal_vortex(train_data, vortex_period)
    train_data['UB'], train_data['LB'] = calculate_fractal_bands(train_data, fractal_period)
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals
    entries = (
        (train_data['Close'] < train_data['UB']) &  # Close is below Upper Band
        (train_data['VI+'] > train_data['VI-'])    # VI+ is above VI-
    )
    exits = (
        (train_data['Close'] > train_data['LB']) &  # Close is above Lower Band
        (train_data['VI-'] > train_data['VI+'])    # VI- is above VI+
    )

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy hma_stc
############################

# Function to calculate Hull Moving Average (HMA)
def hma_stc(series, period):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half_length).apply(
        lambda x: np.dot(x, range(1, half_length + 1)) / sum(range(1, half_length + 1)), raw=True
    )
    wma_full = series.rolling(window=period).apply(
        lambda x: np.dot(x, range(1, period + 1)) / sum(range(1, period + 1)), raw=True
    )
    hma = (2 * wma_half - wma_full).rolling(window=sqrt_length).mean()
    return hma

# Function to calculate Schaff Trend Cycle (STC)
def calculate_stc(data, short_n, long_n, cycle_n):
    macd = data['Close'].ewm(span=short_n, adjust=False).mean() - data['Close'].ewm(span=long_n, adjust=False).mean()
    macd_signal = macd.ewm(span=cycle_n, adjust=False).mean()
    stc = (macd - macd_signal).ewm(span=cycle_n, adjust=False).mean()
    data['STC'] = stc
    return data

# Default params range
# strategy_hma_stc_param_ranges = {
#     # Define dynamic ranges for parameters
#     'hma_short_period_range' : range(1, 21, 2),
#     'hma_long_period_range' : range(20, 41, 2),
#     'short_n_range' : range(1, 16, 2),
#     'long_n_range' : range(18, 31, 2),
#     'cycle_n_range' : range(1, 15, 2)
# }

# --- Griglia parametri RIDOTTA ---
strategy_hma_stc_param_ranges = {
    'hma_short_period_range': [8, 12, 16],
    'hma_long_period_range' : [24, 32, 40],
    'short_n_range'         : [5, 10],
    'long_n_range'          : [20, 26, 30],
    'cycle_n_range'         : [5,10],
}
  
def strategy_hma_stc(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    hma_short_period = params.get('hma_short_period_range')
    hma_long_period = params.get('hma_long_period_range')
    short_n = params.get('short_n_range')
    long_n = params.get('long_n_range')
    cycle_n = params.get('cycle_n_range')
    
    train_data  =  data.copy()
    
    # Calculate indicators on the training data
    train_data = calculate_stc(train_data.copy(), short_n, long_n, cycle_n)
    train_data['HMA_Short'] = hma_stc(train_data['Close'], hma_short_period)
    train_data['HMA_Long'] = hma_stc(train_data['Close'], hma_long_period)
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals
    entries = (
        (train_data['STC'] > 0) &  # STC is positive
        (train_data['HMA_Short'] > train_data['HMA_Long'])  # Short HMA > Long HMA
    )
    exits = (
        (train_data['STC'] < 0) &  # STC is negative
        (train_data['HMA_Short'] < train_data['HMA_Long'])  # Short HMA < Long HMA
    )

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy macd_aroon
############################

 # Aroon Indicator
def aroon_indicator(df, period=14):
    df['Aroon_Up'] = ((df['High'].rolling(window=period).apply(lambda x: np.argmax(x) + 1) / period) * 100)
    df['Aroon_Down'] = ((df['Low'].rolling(window=period).apply(lambda x: np.argmin(x) + 1) / period) * 100)
    return df['Aroon_Up'], df['Aroon_Down']

# MACD Indicator
def macd_aroon(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df['MACD'], df['MACD_Signal']
    
# Default params range
strategy_macd_aroon_param_ranges = {
    # Define dynamic ranges for parameters
    'aroon_period_range' : range(10, 31, 5),
    'fast_period_range' : range(5, 21, 5),
    'slow_period_range' : range(20, 31, 2),
    'macd_signal_range' : range(5, 21, 5)
}
  
def strategy_macd_aroon(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    aroon_period = params.get('aroon_period_range')
    fast_period = params.get('fast_period_range')
    slow_period = params.get('slow_period_range')
    macd_signal = params.get('macd_signal_range')
    
    train_data  =  data.copy()
    
    # Calculate indicators on the training data
    train_data['Aroon_Up'], train_data['Aroon_Down'] = aroon_indicator(train_data.copy(), period=aroon_period)
    train_data['MACD'], train_data['MACD_Signal'] = macd_aroon(train_data.copy(), fast_period=fast_period, slow_period=slow_period, signal_period=macd_signal)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals
    entries = (
        (train_data['Aroon_Down'] < train_data['Aroon_Up']) &  # Aroon Down < Aroon Up
        (train_data['MACD'] > train_data['MACD_Signal'])  # MACD Line crosses above Signal Line
    )
    exits = (
        (train_data['Aroon_Down'] > train_data['Aroon_Up']) &  # Aroon Down > Aroon Up
        (train_data['MACD'] < train_data['MACD_Signal'])  # MACD Line crosses below Signal Line
    )

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy ema_confluence
############################

# Function to calculate Exponential Moving Averages (EMAs) with reusable parameters
def calculate_emas(df, close_col='Close', periods=None):
    if periods is None:
        periods = [20, 50, 100, 200]  # Default periods if none are provided

    for period in periods:
        df[f'EMA_{period}'] = df[close_col].ewm(span=period, adjust=False).mean()
    return df

# Function to detect confluence zone with reusable periods
def detect_confluence(df, periods=None):
    if periods is None:
        periods = [20, 50, 100, 200]  # Default periods if none are provided

    ema_columns = [f'EMA_{period}' for period in periods]
    df['Confluence_Upper'] = df[ema_columns].max(axis=1)
    df['Confluence_Lower'] = df[ema_columns].min(axis=1)
    return df

# # Default params range
# strategy_ema_confluence_param_ranges = {
#     # Define dynamic ranges for EMA periods
#     'MA_1_range' : range(5, 21),
#     'MA_2_range' : range(21, 51, 2),
#     'MA_3_range' : range(51, 101, 5),
#     'MA_4_range' : range(101, 201, 5)
# }

# Default params range (ristretto)
strategy_ema_confluence_param_ranges = {
    # Define dynamic ranges for EMA periods
    'MA_1_range' : range(5, 25, 5),
    'MA_2_range' : range(21, 54, 8),
    'MA_3_range' : range(51, 111, 15),
    'MA_4_range' : range(101, 227, 31)
}

def strategy_ema_confluence(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    MA_1 = params.get('MA_1_range')
    MA_2 = params.get('MA_2_range')
    MA_3 = params.get('MA_3_range')
    MA_4 = params.get('MA_4_range')

    train_data  =  data.copy()
    
    # Calculate EMAs on the training data
    train_data = calculate_emas(train_data, periods=[MA_1, MA_2, MA_3, MA_4])
    train_data = detect_confluence(train_data, periods=[MA_1, MA_2, MA_3, MA_4])
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on confluence zone
    entries = (train_data['Close'] > train_data['Confluence_Upper']) & (train_data['Volume'] > train_data['Volume'].rolling(20).mean() * 1.5)
    exits = (train_data['Close'] < train_data['Confluence_Lower'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy hma_smi
############################

# Function to calculate the Hull Moving Average (HMA)
# gia' definita

def calculate_hma_smi(data, period):
    """
    Calculate the Hull Moving Average (HMA).
    """
    wma_half = data['Close'].ewm(span=period // 2).mean()
    wma_full = data['Close'].ewm(span=period).mean()
    hma = (2 * wma_half -wma_full).ewm(span=int(np.sqrt(period))).mean()
    return hma
    
# Function to calculate the Stochastic Momentum Index (SMI)
def calculate_smi(data, period=14, smooth_k=3, smooth_d=3):
    """
    Calculate the Stochastic Momentum Index (SMI).
    """
    high_max = data['High'].rolling(window=period).max()
    low_min = data['Low'].rolling(window=period).min()
    smi = 100 * (data['Close'] - (high_max + low_min) / 2) / (high_max - low_min)
    smi = smi.ewm(span=smooth_k).mean()  # Smooth K
    smi = smi.ewm(span=smooth_d).mean()  # Smooth D
    return smi


# Default params range
strategy_hma_smi_param_ranges = {
    # Define dynamic ranges for parameters
    'smi_period_range' : range(5, 31),  # SMI period range
    'hma_period_range' : range(5, 31)  # HMA period range
}
  
def strategy_hma_smi(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    smi_period = params.get('smi_period_range')
    hma_period = params.get('hma_period_range')
    
    train_data  =  data.copy()
    
    # Calculate indicators on the training data
    train_data['SMI'] = calculate_smi(train_data, period=smi_period)
    train_data['Fast_HMA'] = calculate_hma_smi(train_data, period=hma_period)
    train_data['Slow_HMA'] = calculate_hma_smi(train_data, period=hma_period * 2)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on the fast and slow HMA crossovers and SMI
    entries = (train_data['Fast_HMA'] > train_data['Slow_HMA']) & (train_data['SMI'] > -40)
    exits = (train_data['Fast_HMA'] < train_data['Slow_HMA']) & (train_data['SMI'] < 40)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy ma_slope_atr_rsi
############################

# Function to calculate Moving Average Slope
def calculate_ma_slope(df, period=50):
    ma = df['Close'].rolling(window=period).mean()
    first_derivative = ma.diff()
    second_derivative = first_derivative.diff()
    return first_derivative, second_derivative

# Function to calculate ATR (Average True Range). 
# Gia' definita

# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Default params range
strategy_ma_slope_atr_rsi_param_ranges = {
    # Define dynamic ranges for periods
    'ma_slope_period_range' : range(5, 38, 8),  # Range for MA Slope period
    'atr_period_range' : range(5, 38, 8),  # Range for ATR period
    'rsi_period_range' : range(5, 38, 8)  # Range for RSI period
}
  
def strategy_ma_slope_atr_rsi(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    ma_slope_period = params.get('ma_slope_period_range')
    atr_period = params.get('atr_period_range')
    rsi_period = params.get('rsi_period_range')
  
    train_data  =  data.copy()
    
    # Calculate indicators on the training data
    train_data['MA_Slope'], _ = calculate_ma_slope(train_data, period=ma_slope_period)
    train_data['ATR'] = calculate_atr(train_data, period=atr_period)
    train_data['RSI'] = calculate_rsi(train_data, period=rsi_period)

    # Sideways market filter (Low ATR and RSI between 40-60)
    train_data['Sideways'] = (train_data['ATR'] < train_data['ATR'].rolling(window=50).mean() * 0.8) & \
                             (train_data['RSI'].between(40, 60))

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Entry and Exit signals
    entries = (train_data['MA_Slope'] > 0) & (~train_data['Sideways'])
    exits = (train_data['MA_Slope'] < 0)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy kst_vri
############################

# Function to calculate the KST (Know Sure Thing) and Signal line
def calculate_kst(df, short_period=10, long_period=15, roc_short_period=5, roc_long_period=10):
    """
    Calculate KST (Know Sure Thing) and its Signal line.
    """
    # Rate of Change (ROC) for short and long periods
    roc_short = df['Close'].pct_change(periods=short_period) * 100
    roc_long = df['Close'].pct_change(periods=long_period) * 100
    roc_mid = df['Close'].pct_change(periods=roc_short_period) * 100
    roc_longer = df['Close'].pct_change(periods=roc_long_period) * 100

    # Smoothed ROC using weighted moving averages
    wma_short = roc_short.rolling(window=short_period).mean()
    wma_long = roc_long.rolling(window=long_period).mean()
    wma_mid = roc_mid.rolling(window=roc_short_period).mean()
    wma_longer = roc_longer.rolling(window=roc_long_period).mean()

    # KST = weighted sum of ROCs
    kst = (wma_short * 1) + (wma_long * 2) + (wma_mid * 3) + (wma_longer * 4)

    # Signal line: EMA of KST
    kst_signal = kst.ewm(span=9).mean()

    return kst, kst_signal

# Function to calculate KST for a specific time period
def calculate_kst_indicator(df, kst_params):
    kst, kst_signal = calculate_kst(df, *kst_params)
    df['KST'] = kst
    df['KST_Signal'] = kst_signal
    return df

# Function to calculate Relative Vigor Index (RVI)
def calculate_rvi(df, period=14):
    """
    Calculate Relative Vigor Index (RVI).
    """
    close_open = df['Close'] - df['Open']
    high_low = df['High'] - df['Low']

    rvi = close_open.rolling(window=period).sum() / high_low.rolling(window=period).sum()

    return rvi
 
# Default params range
strategy_kst_vri_param_ranges = {
    # Define dynamic ranges for KST and RVI periods
    'kst_short_period_range' : range(4, 28, 8),  # Range for KST short period
    'kst_long_period_range' : range(14, 35, 10),  # Range for KST long period
    'roc_short_period_range' : range(3, 11, 2),  # Range for KST ROC short period
    'roc_long_period_range' : range(5, 18, 6),   # Range for KST ROC long period
    'rvi_period_range' : range(10, 23, 6)        # Range for RVI period
}
  
def strategy_kst_vri(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    kst_short_period = params.get('kst_short_period_range')
    kst_long_period = params.get('kst_long_period_range')
    roc_short_period = params.get('roc_short_period_range')
    roc_long_period = params.get('roc_long_period_range')
    rvi_period = params.get('rvi_period_range')
    
    train_data  =  data.copy()
    
    # Calculate KST and RVI on the training data
    train_data = calculate_kst_indicator(train_data, (kst_short_period, kst_long_period, roc_short_period, roc_long_period))
    train_data['RVI'] = calculate_rvi(train_data, rvi_period)
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on KST and RVI
    entries = (train_data['KST'] > train_data['KST_Signal']) & (train_data['RVI'] > 0)
    exits = (train_data['KST'] < train_data['KST_Signal']) & (train_data['RVI'] < 0)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy macd_ravi
############################

# Function to calculate RAVI (Range Action Verification Index)
def calculate_ravi(df, short_period=14, long_period=28):
    """
    Calculate the Range Action Verification Index (RAVI).
    """
    high_roll = df['High'].rolling(window=long_period).max() - df['Low'].rolling(window=long_period).min()
    low_roll = df['High'].rolling(window=short_period).max() - df['Low'].rolling(window=short_period).min()
    ravi = (low_roll - high_roll) / high_roll * 100
    return ravi

# Function to calculate MACD and Signal Line
def calculate_macd_ravi(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal Line.
    """
    macd = df['Close'].ewm(span=fast_period, adjust=False).mean() - df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, macd_signal


# Default params range
strategy_macd_ravi_param_ranges = {
    # Define dynamic ranges for RAVI and MACD periods
    'ravi_short_period_range' : range(10, 28, 9),  # Range for RAVI short period
    'ravi_long_period_range' : range(20, 60, 20),   # Range for RAVI long period
    'macd_fast_period_range' : range(9, 18, 3),     # Range for MACD fast period
    'macd_slow_period_range' : range(20, 33, 6),    # Range for MACD slow period
    'macd_signal_period_range' : range(6, 15, 3),   # Range for MACD signal period
    'ravi_entry_thres_range' : range(-10, -55, -15),  # Range for RAVI entry threshold
    'ravi_exit_thres_range' : range(-50, -110, -20)  # Range for RAVI exit threshold
}
  
def strategy_macd_ravi(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    ravi_short_period = params.get('ravi_short_period_range')
    ravi_long_period = params.get('ravi_long_period_range')
    macd_fast_period = params.get('macd_fast_period_range')
    macd_slow_period = params.get('macd_slow_period_range')
    macd_signal_period = params.get('macd_signal_period_range')
    ravi_entry_thres = params.get('ravi_entry_thres_range')
    ravi_exit_thres = params.get('ravi_exit_thres_range')
    
    train_data  =  data.copy()
    
    # Calculate RAVI and MACD on the training data
    train_data['RAVI'] = calculate_ravi(train_data, ravi_short_period, ravi_long_period)
    train_data['MACD'], train_data['MACD_Signal'] = calculate_macd_ravi(train_data, macd_fast_period, macd_slow_period, macd_signal_period)

        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on MACD and RAVI strategy
    entries = (train_data['MACD'] > train_data['MACD_Signal']) & (train_data['RAVI'] < ravi_entry_thres)
    exits = (train_data['MACD'] < train_data['MACD_Signal']) & (train_data['RAVI'] > ravi_exit_thres)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy dpo_sroc
############################
 
# Function to calculate Shifted Detrended Price Oscillator (DPO)
def calculate_sdpo(df, period=20):
    """
    Calculate Detrended Price Oscillator (DPO).
    """
    shifted_sma = df['Close'].rolling(window=period).mean().shift(int(period / 2) + 1)
    dpo = df['Close'] - shifted_sma
    return dpo


# Function to calculate Smoothed Rate of Change (S-ROC)
def calculate_sroc(df, period=14, smoothing=3):
    """
    Calculate Smoothed Rate of Change (S-ROC).
    """
    roc = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
    sroc = roc.rolling(window=smoothing).mean()  # Apply smoothing
    return sroc

# Default params range
strategy_dpo_sroc_param_ranges = {
    # Define the range of parameters for optimization
    'dpo_period_range' : range(10, 36, 6),  # Try DPO periods from 10 to 30
    'sroc_period_range' : range(10, 36, 6),  # Try S-ROC periods from 10 to 30
    'sroc_smoothing_range' : range(1, 5, 1)  # Try smoothing values from 1 to 5
}

  
def strategy_dpo_sroc(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    dpo_period = params.get('dpo_period_range')
    sroc_period = params.get('sroc_period_range')
    sroc_smoothing = params.get('sroc_smoothing_range')

    train_data  =  data.copy()
    
    # Calculate DPO and S-ROC with current parameters
    train_data['DPO'] = calculate_sdpo(train_data, period=dpo_period)
    train_data['SROC'] = calculate_sroc(train_data, period=sroc_period, smoothing=sroc_smoothing)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Define Entry and Exit signals based on DPO & S-ROC
    entries = (train_data['DPO'] > 0) & (train_data['SROC'] > train_data['SROC'].shift(5))
    exits = (train_data['DPO'] < 0) & (train_data['SROC'] < train_data['SROC'].shift(5))

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy macd_bb
############################

# Function to calculate MACD
def calculate_macd_bb(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Function to calculate Bollinger Bands
def calculate_bollinger_bands_macd_bb(df, period=20, std_dev=2):
    rolling_mean = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

# Default params range
strategy_macd_bb_param_ranges = {
    # Define parameter ranges
    'macd_short_range' : range(8, 17, 3),
    'macd_long_range' : range(20, 33, 6),
    'macd_signal_range' : range(5, 13, 2),
    'bb_period_range' : range(10, 34, 12),
    'bb_std_range' : [1.5, 2, 2.5, 3.0],
    'squeeze_window_range' : range(10, 34, 12)
}
  
def strategy_macd_bb(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    macd_short = params.get('macd_short_range')
    macd_long = params.get('macd_long_range')
    macd_signal = params.get('macd_signal_range')
    bb_period = params.get('bb_period_range')
    bb_std = params.get('bb_std_range')
    squeeze_window = params.get('squeeze_window_range')
    
    train_data  =  data.copy()
    
    train_data['MACD'], train_data['MACD_Signal'] = calculate_macd_bb(train_data, macd_short, macd_long, macd_signal)
    train_data['Upper_Band'], train_data['Lower_Band'] = calculate_bollinger_bands_macd_bb(train_data, bb_period, bb_std)
    train_data['BB_Squeeze'] = train_data['Upper_Band'] - train_data['Lower_Band']
    train_data['BB_Squeeze_Low'] = train_data['BB_Squeeze'].rolling(window=squeeze_window).min()
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on indicators
    entries = (train_data['BB_Squeeze'] == train_data['BB_Squeeze_Low']) & (train_data['MACD'] > train_data['MACD_Signal'])
    exits = (train_data['MACD'] < train_data['MACD_Signal']) & (train_data['Close'] > train_data['Upper_Band'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
######################################
# Strategy ema_trendline_projection
######################################

# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(series, span=200):
    return series.ewm(span=span, adjust=False).mean()

def calculate_trendline_projection(df, window=30):
    projected_trendline = np.full(len(df), np.nan)

    window = int(window)  # Ensure window is an integer

    for i in range(window, len(df)):
        y = df['Close'][i-window:i].values.reshape(-1, 1)
        x = np.arange(0, window).reshape(-1, 1)

        model = LinearRegression().fit(x, y)
        projected_value = model.predict([[window]])[0][0]  # Predict next step

        projected_trendline[i] = projected_value

    return pd.Series(projected_trendline, index=df.index)

 
# Default params range
strategy_ema_trendline_projection_param_ranges = {
    'ema_span_range' : range(10, 201, 5),
    'trendline_window_range' : range(10, 101, 5)
}

 
def strategy_ema_trendline_projection(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    ema_span = params.get('ema_span_range')
    trendline_window = params.get('trendline_window_range')
    
    train_data  =  data.copy()
        
    # Calculate EMA and Projected Trendline
    train_data['EMA'] = calculate_ema(train_data['Close'], span=ema_span)
    train_data['Projected_Trendline'] = calculate_trendline_projection(train_data, window=trendline_window)
       
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Define Entry and Exit signals based on Trendline Projection
    entries = (train_data['Close'] > train_data['Projected_Trendline']) & (train_data['Close'] > train_data['EMA'])
    exits = (train_data['Close'] < train_data['Projected_Trendline']) & (train_data['Close'] < train_data['EMA'])

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy cmo
############################

# Function to calculate Chande Momentum Oscillator (CMO)
def calculate_cmo(df, window=14):
    up_moves = df['Close'].diff().where(lambda x: x > 0, 0)
    down_moves = -df['Close'].diff().where(lambda x: x < 0, 0)
    cmo = 100 * (up_moves.rolling(window=window).sum() - down_moves.rolling(window=window).sum()) / (up_moves.rolling(window=window).sum() + down_moves.rolling(window=window).sum())
    return cmo

# Default params range
strategy_cmo_param_ranges = {
    # Define dynamic ranges for CMO window and thresholds
    'daily_window_range' : range(5, 41, 12),  # Range for daily CMO window (smaller for faster response)
    'weekly_window_range' : range(1, 33, 8),  # Range for weekly CMO window (larger for smoother trends)
    'weekly_optimize_range' : range(10, 62, 26),  # Weekly CMO optimization range
    'daily_optimize_entry_range' : range(-50, 2, 26),  # Daily CMO entry threshold optimization
    'daily_optimize_exit_range' : range(10, 62, 26)  # Daily CMO exit threshold optimization
}

def strategy_cmo(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    daily_window = params.get('daily_window_range')
    weekly_window = params.get('weekly_window_range')
    weekly_optimize = params.get('weekly_optimize_range')
    daily_optimize_entry = params.get('daily_optimize_entry_range')
    daily_optimize_exit = params.get('daily_optimize_exit_range')
    
    train_data  =  data.copy()
    
    # Calculate CMO for weekly and daily data
    weekly_data = train_data.resample('W').last()  # Resample weekly data
    weekly_data['CMO'] = calculate_cmo(weekly_data, weekly_window)
    daily_data = train_data
    daily_data['CMO'] = calculate_cmo(daily_data, daily_window)

    # Forward-fill weekly CMO to daily data for trend confirmation
    weekly_data = weekly_data.reindex(daily_data.index, method='ffill')
    daily_data['HTF_Trend'] = weekly_data['CMO'] > weekly_optimize  # Weekly trend confirmation
   
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        daily_data = daily_data[daily_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = (daily_data['CMO'] < daily_optimize_entry) & (daily_data['HTF_Trend'])
    exits = (daily_data['CMO'] > daily_optimize_exit) & (~daily_data['HTF_Trend'])


    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy macd_tp_williams
############################

# Nota: differisce dalla strategia macd_williams nel calcolo del MACD 
#       che in questa verisone usa il tipical price

# Function to calculate MACD using (Open + High + Low + Close) / 4
def calculate_macd_tp_williams(df, short_window=12, long_window=26, signal_window=9):
    df['Typical_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['MACD'] = df['Typical_Price'].ewm(span=short_window, min_periods=1).mean() - \
                 df['Typical_Price'].ewm(span=long_window, min_periods=1).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, min_periods=1).mean()
    return df

# Function to calculate Williams %R
# gia' definita

# Default params range
strategy_macd_tp_williams_param_ranges = {
    # Define dynamic ranges for MACD and Williams %R parameters
    'short_window_range' : range(5, 18, 3),  # Range for MACD short window
    'long_window_range' : range(16, 34, 4),  # Range for MACD long window
    'signal_window_range' : range(5, 18, 3),  # Range for MACD signal window
    'williams_r_period_range' : range(10, 23, 3),  # Range for Williams %R period
    'williams_r_entry_thresholds_range' : [-10, -20, -30],  # Entry thresholds for Williams %R
    'williams_r_exit_thresholds_range' : [-70, -80, -90]  # Exit thresholds for Williams %R

}
  
def strategy_macd_tp_williams(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    short_window = params.get('short_window_range')
    long_window = params.get('long_window_range')
    signal_window = params.get('signal_window_range')
    williams_r_period = params.get('williams_r_period_range')
    williams_r_entry_thresholds = params.get('williams_r_entry_thresholds_range')
    williams_r_exit_thresholds = params.get('williams_r_exit_thresholds_range')
    
    train_data  =  data.copy()
    
    # Calculate MACD and Williams %R on the training data
    train_data = calculate_macd_tp_williams(train_data, short_window, long_window, signal_window)
    train_data = calculate_williams_r(train_data, williams_r_period)
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on MACD and Williams %R
    entries = (train_data['MACD'] > train_data['MACD_Signal']) & (train_data['Williams_%R'] < williams_r_entry_thresholds)
    exits = (train_data['MACD'] < train_data['MACD_Signal']) & (train_data['Williams_%R'] < williams_r_exit_thresholds)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy cci_aroon
############################

# Function to calculate Aroon Oscillator
def calculate_cci_aroon(df, period=14):
    """
    Calculate the Aroon Oscillator.
    """
    aroon_up = ((df['High'].rolling(window=period).apply(lambda x: np.argmax(x) + 1) - 1) / period) * 100
    aroon_down = ((df['Low'].rolling(window=period).apply(lambda x: np.argmin(x) + 1) - 1) / period) * 100
    aroon_oscillator = aroon_up - aroon_down
    return aroon_oscillator

# Function to calculate Commodity Channel Index (CCI)
# gia' definita
# def calculate_cci(df, period=20):

# Default params range
strategy_cci_aroon_param_ranges = {
    # Define dynamic ranges for Aroon and CCI periods
    'aroon_period_range' : range(5, 130, 31),  # Range for Aroon Oscillator periods
    'cci_period_range' : range(5, 130, 31)  # Range for CCI periods
}
  
def strategy_cci_aroon(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    aroon_period = params.get('aroon_period_range')
    cci_period = params.get('cci_period_range')
    
    train_data  =  data.copy()
    
    # Calculate Aroon Oscillator and CCI on the training data
    train_data['Aroon_Oscillator'] = calculate_cci_aroon(train_data, period=aroon_period)
    train_data['CCI'] = calculate_cci(train_data, period=cci_period)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Define entry and exit signals based on Aroon Oscillator and CCI
    entries = (train_data['Aroon_Oscillator'] > 0) & (train_data['CCI'] > 100)
    exits = (train_data['Aroon_Oscillator'] < 0) & (train_data['CCI'] < -100)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy macd_cfo
############################

# Function to calculate the Chande Forecast Oscillator (CFO)
def calculate_cfo(df, period=10):
    """
    Calculate the Chande Forecast Oscillator (CFO).
    """
    high_rolling = df['High'].rolling(window=period).max()
    low_rolling = df['Low'].rolling(window=period).min()

    # Forecast values
    forecast = 100 * (df['Close'] - low_rolling) / (high_rolling - low_rolling) - 50

    return forecast

# Function to calculate the MACD and its signal line
def calculate_macd_cfo(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the MACD and Signal line.
    """
    fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()

    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()

    return macd, macd_signal

 
# Default params range
strategy_macd_cfo_param_ranges = {
    # Define dynamic ranges for CFO and MACD parameters
    'cfo_period_range' : range(5, 23, 4),  # Range for CFO period
    'fast_period_range' : range(5, 23, 4),  # Range for MACD fast period
    'slow_period_range' : range(20, 46, 6),  # Range for MACD slow period
    'signal_period_range' : range(5, 23, 4)  # Range for MACD signal period
}
  
def strategy_macd_cfo(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    cfo_period = params.get('cfo_period_range')
    fast_period = params.get('fast_period_range')
    slow_period = params.get('slow_period_range')
    signal_period = params.get('signal_period_range')
    
    train_data  =  data.copy()
    
    # Calculate CFO and MACD on the training data
    train_data['CFO'] = calculate_cfo(train_data, period=cfo_period)
    train_data['MACD'], train_data['MACD_Signal'] = calculate_macd_cfo(train_data, fast_period, slow_period, signal_period)
    train_data['OSMA'] = train_data['MACD'] - train_data['MACD_Signal']

        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on CFO and MACD
    entries = (train_data['CFO'] > 0) & (train_data['OSMA'] > 0)
    exits = (train_data['CFO'] < 0) & (train_data['OSMA'] < 0)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    

###########################
# Strategy simple_sma   
###########################

strategy_simple_sma_param_ranges = {
    # Define dynamic ranges for daily and weekly SMA parameters
    'short_window_range' : range(3, 12, 2),  # Range for short window (daily)
    'long_window_range' : range(10, 50, 2),  # Range for long window (daily)
}

def strategy_simple_sma(data, params, year=None):
    
    # Define dynamic ranges for DMA parameters
    short_window = params.get('short_window_range')  # Range for fast DMA periods
    long_window = params.get('long_window_range')   # Range for slow DMA periods
    
    
    train_data  =  data.copy()
    
    # Calculate SMA for weekly and daily data
    # daily_data = train_data
    train_data['Short_SMA'], train_data['Long_SMA'] = calculate_sma(train_data, short_window, long_window)
    
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals for the strategy
    entries = train_data['Short_SMA'] > train_data['Long_SMA']
    exits = train_data['Short_SMA'] < train_data['Long_SMA']
    
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


#######################################
# Da qui in poi sono conformi (shift)
#######################################

############################
# Strategy macd_kvo
############################

# === MACD ===
def ind_macd_kvo_macd(df: pd.DataFrame,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> tuple[pd.Series, pd.Series]:
    close = df['Close']
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False, min_periods=1).mean()
    return macd, macd_signal

# === KVO (versione semplificata e robusta) ===
def ind_macd_kvo_kvo(df: pd.DataFrame,
                     short_period: int = 34,
                     long_period: int = 55,
                     signal_period: int = 13) -> tuple[pd.Series, pd.Series]:
    high = df['High']; low = df['Low']; close = df['Close']; vol = df['Volume']

    # True Range (robusto alle prime barre)
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(hl, hc), lc)

    # Volume Force semplificata: segno dal delta del close
    vf = np.where(close > close.shift(1), vol, np.where(close < close.shift(1), -vol, 0))
    vf = pd.Series(vf, index=df.index)

    # Oscillatore come differenza fra somme mobili (proxy del Klinger)
    kvo = vf.rolling(window=short_period, min_periods=1).sum() - vf.rolling(window=long_period, min_periods=1).sum()
    kvo_signal = kvo.rolling(window=signal_period, min_periods=1).mean()
    return kvo, kvo_signal

# --- Griglia parametri per WF Optimization ---
# strategy_macd_kvo_param_ranges = {
#     'short_period_range' : range(10, 50, 5),
#     'long_period_range'  : range(30, 100, 5),
#     'signal_period_range': range(5, 20, 5),
#     'fast_period_range'  : range(5, 20, 5),
#     'slow_period_range'  : range(20, 50, 5),
#     'macd_signal_range'  : range(5, 15, 2)
# }
strategy_macd_kvo_param_ranges = {
    'short_period_range' : range(30, 40, 5),
    'long_period_range'  : range(50, 80, 5),
    'signal_period_range': range(5, 20, 5),
    'fast_period_range'  : range(5, 15, 5),
    'slow_period_range'  : range(20, 30, 5),
    'macd_signal_range'  : range(5, 15, 2)
}

# --- Funzione di strategia ---
def strategy_macd_kvo(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Segnali long quando MACD > Signal e KVO > Signal.
    Segnali exit quando MACD < Signal e KVO < Signal.

    Ordine K_Strategy:
    1) indicatori su tutto df
    2) slicing per anno
    3) entries/exits
    4) shift di 1 barra e normalizzazione bool
    """
    # Leggi parametri
    short_p  = int(params.get('short_period_range'))
    long_p   = int(params.get('long_period_range'))
    sig_p    = int(params.get('signal_period_range'))
    fast_p   = int(params.get('fast_period_range'))
    slow_p   = int(params.get('slow_period_range'))
    macd_sig = int(params.get('macd_signal_range'))

    df = data.copy()

    # --- Calcolo indicatori (sempre su tutto il df)
    macd, macd_signal = ind_macd_kvo_macd(df, fast_period=fast_p, slow_period=slow_p, signal_period=macd_sig)
    kvo,  kvo_signal  = ind_macd_kvo_kvo(df, short_period=short_p, long_period=long_p, signal_period=sig_p)

    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['KVO'] = kvo
    df['KVO_Signal'] = kvo_signal

    # --- Slicing per anno (dopo gli indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Se combinazione parametri “illegale”, produci nessun trade (serie False) senza rompere il backtest
    if (short_p >= long_p) or (fast_p >= slow_p):
        entries = pd.Series(False, index=df.index)
        exits   = pd.Series(False, index=df.index)
    else:
        # --- Condizioni entry/exit
        entries = (df['KVO'] > df['KVO_Signal']) & (df['MACD'] > df['MACD_Signal'])
        exits   = (df['KVO'] < df['KVO_Signal']) & (df['MACD'] < df['MACD_Signal'])

    # --- Shift e normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy pvt_fisher
############################

# --- PVT (Price Volume Trend) ---
def ind_pvt_fisher_pvt(df: pd.DataFrame) -> pd.Series:
    """
    PVT vettorizzato: cumsum( pct_change(Close) * Volume ).
    Evita indicizzazione posizionale ambigua e gestisce NaN/zero volumi.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    close = df['Close'].astype(float)
    vol   = df['Volume'].fillna(0.0).astype(float)
    ret   = close.pct_change().fillna(0.0)
    pvt   = (ret * vol).cumsum()
    return pvt

# --- Fisher Transform su prezzo normalizzato (Ehlers-like) ---
def ind_pvt_fisher_fisher(df: pd.DataFrame,
                          period: int = 10,
                          signal_span: int = 9) -> tuple[pd.Series, pd.Series]:
    """
    Fisher Transform di X = 2*((Close - LL) / (HH - LL) - 0.5), con:
    - rolling HH/LL su 'period'
    - epsilon su denominatore
    - clip di X a (-0.999, 0.999) per evitare log(0)/inf
    Ritorna: (fisher, fisher_signal)
    """
    if df.empty:
        empty = pd.Series(dtype=float, index=df.index)
        return empty, empty

    high = df['High'].astype(float)
    low  = df['Low'].astype(float)
    close= df['Close'].astype(float)

    hh = high.rolling(window=period, min_periods=1).max()
    ll = low.rolling(window=period, min_periods=1).min()
    denom = (hh - ll).replace(0.0, np.nan)

    # X in [-1, 1] (con gestione NaN)
    x = 2.0 * ((close - ll) / denom - 0.5)
    x = x.clip(lower=-0.999, upper=0.999)

    with np.errstate(divide='ignore', invalid='ignore'):
        fisher = 0.5 * np.log((1.0 + x) / (1.0 - x))

    fisher_signal = fisher.ewm(span=signal_span, adjust=False, min_periods=1).mean()
    return fisher, fisher_signal

# --- Griglia parametri per WFO ---
strategy_pvt_fisher_param_ranges = {
    'fisher_period_range': range(5, 21),   # 5..20
    'pvt_shift_range'    : range(10, 31)   # 10..30
}

# --- Funzione di strategia ---
def strategy_pvt_fisher(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Segnali:
      Entry: PVT > PVT.shift(pvt_shift)  AND Fisher > Fisher_Signal
      Exit : PVT < PVT.shift(pvt_shift)  AND Fisher < Fisher_Signal

    Ordine K_Strategy obbligatorio:
      1) Calcolo indicatori su tutto il df
      2) Slicing per anno
      3) Definizione entries/exits
      4) Shift di 1 barra e normalizzazione boolean
    """
    fisher_period = int(params.get('fisher_period_range'))
    pvt_shift     = int(params.get('pvt_shift_range'))

    df = data.copy()

    # 1) Indicatori su tutto il df
    df['PVT'] = ind_pvt_fisher_pvt(df)
    df['Fisher'], df['Fisher_Signal'] = ind_pvt_fisher_fisher(df, period=fisher_period, signal_span=9)

    # 2) Slicing per anno (dopo il calcolo degli indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    entries = (df['PVT'] > df['PVT'].shift(pvt_shift)) & (df['Fisher'] > df['Fisher_Signal'])
    exits   = (df['PVT'] < df['PVT'].shift(pvt_shift)) & (df['Fisher'] < df['Fisher_Signal'])

    # 4) Shift + normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits


###########################
# Strategy bollinger
###########################

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, period=20, std_dev=2):
    
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper_BB'] = df['SMA'] + (df['STD'] * std_dev)
    df['Lower_BB'] = df['SMA'] - (df['STD'] * std_dev)
    df['BB_Width'] = df['Upper_BB'] - df['Lower_BB']  # Bollinger Band Width
    
    return df

strategy_bollinger_param_ranges = {
    # Define dynamic ranges for short and long windows
    'period_range' : range(1, 39, 9),  # Range for Bollinger Bands period
    'std_dev_range' : [1.5, 2, 2.5, 3.0],  # Range for standard deviation (multiplier)
    'squeeze_period_range' : range(1, 79, 19)  # Range for rolling window in squeeze condition
}
  
def strategy_bollinger(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    period = params.get('period_range')
    std_dev = params.get('std_dev_range')
    squeeze_period = params.get('squeeze_period_range')

    train_data  =  data.copy()
    
    # Calculate Bollinger Bands on the training data
    train_data = calculate_bollinger_bands(train_data, period, std_dev)
    
    # Define squeeze condition
    train_data['Squeeze'] = train_data['BB_Width'] == train_data['BB_Width'].rolling(squeeze_period).min()

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Define entry and exit conditions based on BB strategy
    entries = train_data['Squeeze']
    exits = train_data['Close'] < train_data['Lower_BB']

    # return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    # Shift + normalizzazione
    return (
        entries.shift(1).astype(bool).fillna(False),
        exits.shift(1).astype(bool).fillna(False)
    )


############################
# Strategy kst_rsx
############################

# RSX (approximate)
def rsx(close, length=14):
    pi = np.pi
    alpha = 1.0 / np.exp(1.0 / length)

    price = close.values
    f = np.zeros_like(price)
    g = np.zeros_like(price)
    h = np.zeros_like(price)
    j = np.zeros_like(price)
    k = np.zeros_like(price)
    rsx = np.zeros_like(price)

    for i in range(6, len(price)):
        f[i] = price[i] - price[i-1]
        g[i] = f[i] + 0.5 * f[i-1] + 0.33 * f[i-2] + 0.25 * f[i-3]
        h[i] = g[i] - 0.5 * g[i-1] + 0.33 * g[i-2] - 0.25 * g[i-3]
        j[i] = alpha * h[i] + (1 - alpha) * j[i-1]
        k[i] = alpha * j[i] + (1 - alpha) * k[i-1]
        rsx[i] = 50 + (k[i] / (np.abs(k[i]) + 1e-10)) * min(50, np.abs(k[i]) * 100)

    return pd.Series(rsx, index=close.index)
    
# KST indicator
def calculate_kst_rsx(close, r1=10, r2=15, r3=20, r4=30):
    roc1 = close.pct_change(r1) * 100
    roc2 = close.pct_change(r2) * 100
    roc3 = close.pct_change(r3) * 100
    roc4 = close.pct_change(r4) * 100

    rcma1 = roc1.rolling(10).mean()
    rcma2 = roc2.rolling(10).mean()
    rcma3 = roc3.rolling(10).mean()
    rcma4 = roc4.rolling(15).mean()

    kst = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
    signal = kst.rolling(9).mean()
    return kst, signal


# Default params range
strategy_kst_rsx_param_ranges = {
    'rsx_lengths' : range(5, 53, 24),
    # 'kst_roc_params' : list(itertools.product(range(5, 45, 20), repeat=4))
    # 'kst_roc_params' : (range(5, 35, 10),) * 4
    'r1_range' : range(5, 35, 10),
    'r2_range' : range(5, 35, 10),
    'r3_range' : range(5, 35, 10),
    'r4_range' : range(5, 35, 10)

}
  
def strategy_kst_rsx(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    rsx_length = params.get('rsx_lengths')
    # kst_roc_params = params.get('kst_roc_params')
    r1  =  params.get('r1_range')
    r2  =  params.get('r2_range')
    r3  =  params.get('r3_range')
    r4  =  params.get('r4_range')

    train_data  =  data.copy()

    # Calculate ... indicators on the training data
    train_data['RSX'] = rsx(train_data['Close'], length=rsx_length)
    train_data['KST'], train_data['KST_Signal'] = calculate_kst_rsx(train_data['Close'], r1, r2, r3, r4)

    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]
        
    # Generate entry and exit signals based on ....
    entries = (train_data['RSX'] < 30) & (train_data['KST'] > train_data['KST_Signal'])
    exits = (train_data['RSX'] > 70) & (train_data['KST'] < train_data['KST_Signal'])
        
    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)#.to_numpy()
    shifted_exits = exits.shift(1).astype(bool).fillna(False)#.to_numpy()
    

    return shifted_entries, shifted_exits


############################
# Strategy vix_fractal
############################

# --- VIX Fix ---
def vix_fix(df, period):
    highest_close = df['Close'].rolling(window=period).max()
    vix = ((highest_close - df['Low']) / highest_close) * 100
    vix_avg = vix.rolling(window=period).mean()
    return vix, vix_avg

# --- Fractal Bands ---
def fractal_bands(df, period):
    upper_band = df['High'].rolling(window=period).max()
    lower_band = df['Low'].rolling(window=period).min()
    return upper_band, lower_band

# Default params range
strategy_vix_fractal_param_ranges = {
    'vix_period_range' : range(5, 65, 15),                     # Periods to try for VIX Fix
    'fractal_period_range' : range(5, 65, 15),                 # Periods to try for Fractal Bands
    'vix_spike_threshold_range' : [1.5, 2.0, 2.5, 3.0],    # Threshold for VIX Spike
    'shift_delay_range' : [1,2,3,4,5]                      # nuovo parametro non previsto da K
}
        
  
def strategy_vix_fractal(data, params, year=None):
    """
    Genera segnali (entries, exits) usando indicatori ...,
    calcolati su base giornaliera.
    """
    vix_period = params.get('vix_period_range')
    fractal_period = params.get('fractal_period_range')
    vix_spike_threshold = params.get('vix_spike_threshold_range')
    shift_delay = params.get('shift_delay_range')

    train_data  =  data.copy()
    
    # Calculate ... indicators on the training data
    train_data['VIX'], train_data['VIX_avg'] = vix_fix(train_data, vix_period)
    train_data['Upper'], train_data['Lower'] = fractal_bands(train_data, fractal_period)

    train_data['vix_spike'] = train_data['VIX'] > vix_spike_threshold * train_data['VIX_avg']
    train_data['near_lower'] = train_data['Close'] <= train_data['Lower'] * 1.01
    train_data['near_upper'] = train_data['Close'] >= train_data['Upper'] * 0.99
        
    # Keep only the year to avoid missing values from indicator calculation
    if year is not None:
        train_data = train_data[train_data.index.year == year]

    # Generate entry and exit signals based on ....
    entries = (train_data['vix_spike'] | train_data['near_lower'])
    exits = ((~train_data['vix_spike']) & train_data['near_upper'])
    
    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(shift_delay).astype(bool).fillna(False)#.to_numpy()
    shifted_exits = exits.shift(shift_delay).astype(bool).fillna(False)#.to_numpy()

    return shifted_entries, shifted_exits
    # return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy uo_gapo
############################

# === Ultimate Oscillator ===
def ind_uo_gapo_ultimate_oscillator(df, short=7, medium=14, long=28):
    high = df['High']
    low = df['Low']
    close = df['Close']

    TR = np.maximum(np.maximum(high - low, (high - close.shift(1)).abs()), (low  - close.shift(1)).abs())

    BP = close - np.minimum(low, close.shift(1))

    avg_short  = BP.rolling(short).sum()  / TR.rolling(short).sum()
    avg_medium = BP.rolling(medium).sum() / TR.rolling(medium).sum()
    avg_long   = BP.rolling(long).sum()   / TR.rolling(long).sum()

    UO = 100 * (4 * avg_short + 2 * avg_medium + avg_long) / (4 + 2 + 1)
    return UO
    
# === GAPO ===
def ind_uo_gapo_gapo(df, period=14):
    highest = df['High'].rolling(period).max()
    lowest  = df['Low'].rolling(period).min()
    n = (highest - lowest) / df['Close'].rolling(period).mean()
    n = n.replace(0, np.nan)
    return np.log(n) / np.log(period)
    

# --- Griglia parametri per WF Optimization ---
strategy_uo_gapo_param_ranges = {
    'short_range'  : range(3, 12, 3),
    'medium_range' : range(10, 25, 5),
    'long_range'   : range(21, 42, 7),
    'gapo_range'   : range(10, 25, 5),
    # 'shift_range'  : range(1, 7, 2)
    'shift_range'  : range(2, 13, 3)

}

# --- Funzione di strategia ---
def strategy_uo_gapo(data, params, year=None):
    """
    Genera segnali (entries, exits) usando Ultimate Oscillator + GAPO.
    Regole:
      Entry  se:
        UO < UO.shift(shift)
        AND GAPO_Trend < 0
      Exit   se:
        UO > UO.shift(shift)
        AND GAPO_Trend > 0
    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    short       = params.get('short_range')
    medium      = params.get('medium_range')
    long        = params.get('long_range')
    gapo_period = params.get('gapo_range')
    shift       = params.get('shift_range')

    df = data.copy()

    # Indicatori
    df['UO'] = ind_uo_gapo_ultimate_oscillator(df, short, medium, long)
    df['GAPO'] = ind_uo_gapo_gapo(df, gapo_period)
    df['GAPO_Trend'] = df['GAPO'].diff()
    

    # Filtra per anno dopo il calcolo degli indicatori per mantenere l'allineamento
    if year is not None:
        df = df[df.index.year == year]

    # Entry e Exit
    entries = (df['UO'] < df['UO'].shift(shift)) & (df['GAPO_Trend'] < 0)
    exits   = (df['UO'] > df['UO'].shift(shift)) & (df['GAPO_Trend'] > 0)
    
    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

    
############################
# Strategy kvo_cci
############################

# === KVO Calculation ===
def ind_kvo_cci_kvo(df, fast=34, slow=55, signal=13):
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    trend = close > close.shift(1)
    dm = high - low
    cm = np.where(trend, dm, -dm)
    vf = cm * volume

    vf = pd.Series(vf, index=df.index)
    kvo = vf.ewm(span=fast, min_periods=1).mean() - vf.ewm(span=slow, min_periods=1).mean()
    signal_line = kvo.ewm(span=signal, min_periods=1).mean()
    return kvo, signal_line

# === CCI Calculation ===
def ind_kvo_cci_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    cci = (tp - sma) / (0.015 * mad)
    return cci


# --- Griglia parametri per WF Optimization ---
strategy_kvo_cci_param_ranges = {
    'fast_range'       : range(5, 27, 7),   # KVO fast EMA
    'slow_range'       : range(25, 45, 5),  # KVO slow EMA
    'signal_range'     : range(5, 25, 5),   # KVO signal EMA
    'cci_period_range' : range(10, 34, 12),  # CCI period
    'cci_shift_range'  : range(1, 12, 3)    # barre di shift per il trigger CCI
}


# --- Funzione di strategia ---
def strategy_kvo_cci(data, params, year=None):
    """
    Genera segnali (entries, exits) usando KVO + CCI.
    Regole:
      Entry  se:
        KVO > Signal  AND  KVO < 0 & Signal < 0
        AND CCI > -100  &  CCI.shift(cci_shift) <= -100  (cross-up da -100)
      Exit   se:
        KVO < Signal  AND  KVO > 0 & Signal > 0
        AND CCI < 100  &  CCI.shift(cci_shift) >= 100    (cross-down da 100)
    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    fast       = params.get('fast_range')
    slow       = params.get('slow_range')
    signal     = params.get('signal_range')
    cci_period = params.get('cci_period_range')
    cci_shift  = params.get('cci_shift_range')

    df = data.copy()

    
    df['KVO'], df['KVO_Signal'] = ind_kvo_cci_kvo(df, fast, slow, signal)
    df['CCI'] = ind_kvo_cci_cci(df, period=cci_period)

    # Filtra per anno dopo il calcolo degli indicatori per mantenere l'allineamento
    if year is not None:
        df = df[df.index.year == year]

    # Condizioni di ingresso/uscita
    entries = (
        (df['KVO'] > df['KVO_Signal']) &
        (df['KVO'] < 0) & (df['KVO_Signal'] < 0) &
        (df['CCI'] > -100) & (df['CCI'].shift(cci_shift) <= -100)
    )

    exits = (
        (df['KVO'] < df['KVO_Signal']) &
        (df['KVO'] > 0) & (df['KVO_Signal'] > 0) &
        (df['CCI'] < 100) & (df['CCI'].shift(cci_shift) >= 100)
    )

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    


    return shifted_entries, shifted_exits


# =========================
# Strategia macd_candles
# =========================

# =========================
# Indicatori
# =========================
def ind_macd_candles_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def ind_macd_candles_candles(df: pd.DataFrame, body_ratio: float = 0.3):
    prev  = df.shift(1)
    prev2 = df.shift(2)

    bullish_engulfing = (
        (prev['Close'] < prev['Open']) &
        (df['Close'] > df['Open']) &
        (df['Open']  < prev['Close']) &
        (df['Close'] > prev['Open'])
    )
    bearish_engulfing = (
        (prev['Close'] > prev['Open']) &
        (df['Close'] < df['Open']) &
        (df['Open']  > prev['Close']) &
        (df['Close'] < prev['Open'])
    )

    body   = (df['Close'] - df['Open']).abs()
    range_ = (df['High'] - df['Low']).replace(0, np.nan)
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']

    bullish_pin = (lower_wick > 2 * body) & ((body / range_) < body_ratio)
    bearish_pin = (upper_wick > 2 * body) & ((body / range_) < body_ratio)

    inside_bar = (df['High'] < prev['High']) & (df['Low'] > prev['Low'])

    small_body_prev = (prev['High'] - prev['Low']).replace(0, np.nan)
    small_body_prev = (prev['Close'] - prev['Open']).abs() / small_body_prev < 0.3

    morning_star = (
        (prev2['Close'] < prev2['Open']) &
        small_body_prev &
        (df['Close'] > df['Open']) &
        (df['Close'] > prev2['Open'])
    )
    evening_star = (
        (prev2['Close'] > prev2['Open']) &
        small_body_prev &
        (df['Close'] < df['Open']) &
        (df['Close'] < prev2['Open'])
    )

    # Cast esplicito a 'boolean' per evitare FutureWarning su fillna in downstream
    castb = lambda s: s.astype('boolean')
    return dict(
        bullish_engulfing=castb(bullish_engulfing),
        bearish_engulfing=castb(bearish_engulfing),
        bullish_pin=castb(bullish_pin),
        bearish_pin=castb(bearish_pin),
        inside_bar=castb(inside_bar),
        morning_star=castb(morning_star),
        evening_star=castb(evening_star),
    )


# =========================
# Griglia parametri per WFO 
# =========================
strategy_macd_candles_param_ranges = {
    'fast_range'       : range(5, 18, 3),
    'slow_range'       : range(10, 36, 6),
    'signal_range'     : range(5, 18, 3),
    'body_ratio_range' : [0.1, 0.2, 0.3, 0.4],
    'shift_range'      : range(1, 5, 1)
}


# =========================
# Funzione strategia
# =========================
def strategy_macd_candles(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Genera segnali (entries, exits) usando MACD + pattern candlestick.
    Il filtro per anno è applicato DOPO il calcolo degli indicatori.
    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    df = data.copy()

    fast       = params.get('fast_range')
    slow       = params.get('slow_range')
    signal     = params.get('signal_range')
    body_ratio = params.get('body_ratio_range')
    shift_k    = params.get('shift_range')

    # Indicatori su tutto il dataset
    df['MACD'], df['MACD_signal'] = ind_macd_candles_macd(df, fast=fast, slow=slow, signal=signal)

    patt = ind_macd_candles_candles(df, body_ratio=body_ratio)
    for k, v in patt.items():
        df[k] = v  # già dtype 'boolean'

    # Filtro temporale per anno (dopo indicatori)
    if year is not None:
        df = df[df.index.year == year]

    macd_bullish = (df['MACD'] > df['MACD_signal']).astype('boolean')
    macd_bearish = (df['MACD'] < df['MACD_signal']).astype('boolean')

    # Regole (tipi già 'boolean'; per le serie shiftate usiamo fillna dopo cast)
    entries = (
        (df['bullish_engulfing'] |
         df['bullish_pin'] |
         df['morning_star'] |
         df['inside_bar'].shift(1).astype('boolean').fillna(False)) &
        macd_bullish
    ).astype('boolean')

    exits = (
        (df['bearish_engulfing'] |
         df['bearish_pin'] |
         df['evening_star'] |
         df['inside_bar'].shift(1).astype('boolean').fillna(False)) &
        macd_bearish
    ).astype('boolean')

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    entries = entries.shift(shift_k).astype('boolean').fillna(False).astype(bool)
    exits   = exits.shift(shift_k).astype('boolean').fillna(False).astype(bool)

    entries = entries.shift(1).astype(bool).fillna(False)
    exits   = exits.shift(1).astype(bool).fillna(False)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy name
############################
 
# # Default params range
# strategy_name_param_ranges = {
#     'param1_range' : range(1, 10, 2),
#     'param2_range' : range(1, 10, 2)
# }
  
# def strategy_name(data, params, year=None):
#     """
#     Genera segnali (entries, exits) usando indicatori ...,
#     calcolati su base giornaliera.
#     """
#     param1 = params.get('param1')
#     param2 = params.get('param2')
    
#     train_data  =  data.copy()
    
#     # Calculate ... indicators on the training data
#     train_data['X'] = calculate_X(train_data, period=param1)
#     train_data['Y'] = calculate_Y(train_data, period=param2)
        
#     # Keep only the year to avoid missing values from indicator calculation
#     if year is not None:
#         train_data = train_data[train_data.index.year == year]

#     # Generate entry and exit signals based on ....
#     entries = train_data['X'] > train_data['Y']
#     exits = train_data['X'] < train_data['Y']

#     return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


    
############################
# Strategy kst_dpo
############################

# === KST (Know Sure Thing) ===
def ind_kst_dpo_kst(df, r_tuple=(10, 15, 20, 30), n_tuple=(10, 10, 10, 15), sig=9):
    """
    r_tuple: (r1, r2, r3, r4) finestre ROC in barre
    n_tuple: (n1, n2, n3, n4) SMA applicate alle ROC
    sig    : periodo della signal line (SMA del KST)
    """
    close = df['Close']
    r1, r2, r3, r4 = r_tuple
    n1, n2, n3, n4 = n_tuple

    roc1 = close.pct_change(r1) * 100.0
    roc2 = close.pct_change(r2) * 100.0
    roc3 = close.pct_change(r3) * 100.0
    roc4 = close.pct_change(r4) * 100.0

    sma1 = roc1.rolling(n1).mean()
    sma2 = roc2.rolling(n2).mean()
    sma3 = roc3.rolling(n3).mean()
    sma4 = roc4.rolling(n4).mean()

    kst = sma1 + (sma2 * 2.0) + (sma3 * 3.0) + (sma4 * 4.0)
    signal = kst.rolling(sig).mean()
    return kst, signal


# === DPO (Detrended Price Oscillator) ===
def ind_kst_dpo_dpo(df, period=20):
    """
    DPO = Close - SMA(period) shiftata di floor(period/2)+1
    """
    close = df['Close']
    shift_k = int(period / 2) + 1
    shifted_ma = close.rolling(window=period).mean().shift(shift_k)
    dpo = close - shifted_ma
    return dpo


# --- Griglia parametri per WF Optimization ---
strategy_kst_dpo_param_ranges = {
    # DPO
    'dpo_period_range' : range(5, 101, 5),

    # KST: combinazioni tipiche per ROC e rispettive SMA (liste di tuple)
    'r_values_list'    : [
        (5, 10, 15, 25),
        (10, 15, 20, 30),
        (15, 20, 25, 35),
        (20, 25, 30, 40),
    ],
    'n_values_list'    : [
        (10, 10, 10, 15),
        (15, 15, 15, 20),
        (20, 20, 20, 25),
        (25, 25, 25, 30),
    ],

    # KST signal line
    'sig_range'        : range(5, 101, 5)
}


# --- Funzione di strategia ---
def strategy_kst_dpo(data, params, year=None):
    """
    Genera segnali (entries, exits) usando KST + DPO.

    Regole base (coerenti con il codice di partenza):
      Entry se: KST > 0 AND DPO > 0
      Exit  se: KST < 0 AND DPO < 0

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    import numpy as np
    import pandas as pd

    # Estrazione parametri dalla griglia WFO
    dpo_period = params.get('dpo_period_range')
    r_tuple    = params.get('r_values_list')
    n_tuple    = params.get('n_values_list')
    sig        = params.get('sig_range')

    df = data.copy()

    # Calcolo indicatori
    df['KST'], df['KST_Signal'] = ind_kst_dpo_kst(df, r_tuple=r_tuple, n_tuple=n_tuple, sig=sig)
    df['DPO'] = ind_kst_dpo_dpo(df, period=dpo_period)

    # Filtro per anno dopo il calcolo degli indicatori per mantenere l'allineamento
    if year is not None:
        df = df[df.index.year == year]

    # Condizioni di ingresso/uscita (versione essenziale conforme alla tua logica di base)
    entries = (df['KST'] > 0) & (df['DPO'] > 0)
    exits   = (df['KST'] < 0) & (df['DPO'] < 0)

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

# ============================================================
# Strategy gold_sp500_momentum
# ============================================================


# === Helper di estrazione robusto (dict / MultiIndex / colonne piatte) ===
def _extract_close_df(data_obj, key: str) -> pd.DataFrame:
    """
    Ritorna un DataFrame con almeno la colonna 'Close' per il ticker 'key'.
    Supporta:
      - dict: data_obj[key] -> df con colonna 'Close'
      - DataFrame MultiIndex: colonne (key, 'Close')
      - DataFrame piatto: colonne tipo 'GOLD_Close' o 'Close_GOLD'
    """
    # 1) dict di DataFrame
    if isinstance(data_obj, dict):
        if key not in data_obj:
            raise KeyError(f"Atteso data['{key}'], chiavi presenti: {list(data_obj.keys())}")
        df = data_obj[key]
        if 'Close' not in df.columns:
            raise KeyError(f"data['{key}'] non ha colonna 'Close'. Colonne: {list(df.columns)}")
        return df

    # 2) DataFrame (MultiIndex o piatto)
    if isinstance(data_obj, pd.DataFrame):
        cols = data_obj.columns

        # 2a) MultiIndex (livello 0 = ticker)
        if isinstance(cols, pd.MultiIndex):
            lvl0 = cols.get_level_values(0)
            if key in set(lvl0):
                sub = data_obj.xs(key, axis=1, level=0)
                if 'Close' not in sub.columns:
                    raise KeyError(f"MultiIndex: trovato '{key}' ma manca 'Close'. Colonne: {list(sub.columns)}")
                return sub

        # 2b) colonne piatte: prova pattern comuni
        # pattern 1: 'GOLD_Close'
        cand1 = [c for c in cols if str(c).upper().startswith(key.upper() + '_CLOSE')]
        # pattern 2: 'Close_GOLD'
        cand2 = [c for c in cols if str(c).upper().endswith('_' + key.upper()) and 'CLOSE' in str(c).upper()]
        # pattern 3: unica colonna 'Close' (caso df già del singolo key)
        if 'Close' in cols and len(cols) <= 6:  # euristica soft
            return data_obj

        if cand1:
            return pd.DataFrame({'Close': data_obj[cand1[0]]}, index=data_obj.index)
        if cand2:
            return pd.DataFrame({'Close': data_obj[cand2[0]]}, index=data_obj.index)

    # se arrivi qui, non sei riuscito a estrarre
    raise KeyError(
        f"Impossibile estrarre 'Close' per '{key}'. "
        "Fornisci data come dict {'GOLD':df,'SP500':df} con colonna 'Close', "
        "oppure DataFrame MultiIndex con colonne (ticker,'Close'), "
        "oppure colonne piatte tipo 'GOLD_Close' / 'Close_GOLD'."
    )


# === Ratio GOLD/SP500 ===
def ind_gold_sp500_momentum_ratio(df_gold: pd.DataFrame, df_sp500: pd.DataFrame) -> pd.Series:
    """
    Ritorna la Series del rapporto GOLD/SP500 (Close_GOLD / Close_SP500)
    allineata all'indice temporale di GOLD.
    """
    ratio = df_gold['Close'] / df_sp500['Close'].reindex(df_gold.index)
    return ratio


# === EMA sul ratio ===
def ind_gold_sp500_momentum_ema(series: pd.Series, span: int = 50) -> pd.Series:
    """
    EMA del ratio con min_periods=span per ridurre rumore di warm-up.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


# --- Griglia parametri per WF Optimization (range oggetto, step contenuti) ---
strategy_gold_sp500_momentum_param_ranges = {
    'ema_span_range'   : range(50, 201, 25),  # filtro trend sul ratio
    'roc_period_range' : range(60, 181, 30),  # momentum (≈ 3–6 mesi)
    # Shift operativo fisso a 1 barra come da regole di progetto
}


# --- Funzione di strategia ---
def strategy_gold_sp500_momentum(data, params: dict, year: int | None = None):
    """
    Genera segnali (entries, exits) usando il rapporto GOLD/SP500.

    Regole base:
      Entry: ratio > EMA(ratio)  AND  ROC_ratio(roc_p) > 0
      Exit : ratio < EMA(ratio)  OR   ROC_ratio(roc_p) < 0

    Ordine OBBLIGATORIO (K_Strategy):
      1) Calcolo indicatori sull’intero df
      2) Filtro per anno con slicing (se year è passato)
      3) Definizione entries/exits
      4) Shift di 1 barra e normalizzazione (no look-ahead)
    """
    # --- Estrazione robusta dei prezzi (accetta dict / MultiIndex / piatto)
    df_gold  = _extract_close_df(data, 'GOLD').copy()
    df_sp500 = _extract_close_df(data, 'SP500').copy()

    # --- Parametri dalla griglia WFO ---
    ema_span = params.get('ema_span_range')
    roc_p    = params.get('roc_period_range')

    # --- (1) Indicatori su tutto il dataset ---
    ratio     = ind_gold_sp500_momentum_ratio(df_gold, df_sp500)
    ema_ratio = ind_gold_sp500_momentum_ema(ratio, span=ema_span)
    roc_ratio = ratio.pct_change(roc_p)

    # Costruzione DataFrame di lavoro (allineato a GOLD)
    df = pd.DataFrame(index=df_gold.index)
    df['ratio']     = ratio
    df['ema_ratio'] = ema_ratio
    df['roc_ratio'] = roc_ratio

    # --- (2) Filtro per anno dopo il calcolo degli indicatori ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole di segnale ---
    entries = (df['ratio'] > df['ema_ratio']) & (df['roc_ratio'] > 0)
    exits   = (df['ratio'] < df['ema_ratio']) | (df['roc_ratio'] < 0)

    # --- (4) Shift di 1 barra per evitare look-ahead ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy zscore_mr
############################

# === Z-Score Calculation ===
def ind_zscore_mr_z(df, window=20, ddof=0):
    """
    Calcola lo Z-score sul 'Close' con media e deviazione standard rolling.
    Restituisce una Serie 'Z'. Le prime 'window' barre saranno NaN.
    """
    close = df['Close'].astype(float)
    roll = close.rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std(ddof=ddof).replace(0.0, np.nan)
    z = (close - mean) / std
    return z

# --- Griglia parametri per WF Optimization ---
strategy_zscore_mr_param_ranges = {
    'window_range'  : range(10, 61, 10),          # finestra rolling per media/std
    # 'entry_z_range' : [1.5, 2.0, 2.5, 3.0],       # soglia |Z| per l'entry (usa -entry_z)
    'entry_z_range' : [2.0],       # soglia |Z| per l'entry (usa -entry_z)
    # 'exit_z_range'  : [0.5, 1.0, 1.5, 2.0],       # soglia Z per l'exit
    'exit_z_range'  : [2.0],       # soglia Z per l'exit
    'z_shift_range' : range(1, 6, 1)              # barre di shift per il trigger/cross
}

# --- Funzione di strategia ---
def strategy_zscore_mr(data, params, year=None):
    """
    Genera segnali (entries, exits) usando Z-score di mean reversion.
    Regole:
      Entry  se:  Z < -entry_z  &  Z.shift(z_shift) >= -entry_z   (cross-down sotto -entry_z)
      Exit   se:  Z >  exit_z   &  Z.shift(z_shift) <=  exit_z    (cross-up sopra  exit_z)
    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    window   = params.get('window_range')
    entry_z  = params.get('entry_z_range')
    exit_z   = params.get('exit_z_range')
    z_shift  = params.get('z_shift_range')

    df = data.copy()
    df['Z'] = ind_zscore_mr_z(df, window=window, ddof=0)

    # Filtra per anno dopo il calcolo degli indicatori per mantenere l'allineamento
    if year is not None:
        df = df[df.index.year == year]

    # Condizioni di ingresso/uscita con trigger a cross (evita segnali ripetuti)
    entries = df['Z'] < -float(entry_z) 
    exits   = df['Z'] >  float(exit_z)

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy zscore_mr_momo
############################

# === Z-Score (mean reversion) ===
def ind_zscore_mr_momo_z(df, window=20, ddof=0):
    close = df['Close'].astype(float)
    roll = close.rolling(window=window, min_periods=window)
    mean = roll.mean()
    std = roll.std(ddof=ddof).replace(0.0, np.nan)
    return (close - mean) / std

# === EMA (trend filter) ===
def ind_zscore_mr_momo_ema(df, span=50):
    return df['Close'].astype(float).ewm(span=span, adjust=False, min_periods=span).mean()

# === ADX (trend strength) ===
def ind_zscore_mr_momo_adx(df, period=14):
    high = df['High'].astype(float)
    low  = df['Low'].astype(float)
    close = df['Close'].astype(float)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    tr_rma      = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_dm_rma = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_dm_rma= pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di  = 100 * (plus_dm_rma / tr_rma).replace([np.inf, -np.inf], np.nan)
    minus_di = 100 * (minus_dm_rma / tr_rma).replace([np.inf, -np.inf], np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx

# === Donchian Channels ===
def ind_zscore_mr_momo_donchian(df, window=20):
    upper = df['High'].rolling(window=window, min_periods=window).max()
    lower = df['Low'].rolling(window=window, min_periods=window).min()
    return upper, lower

# --- Griglia parametri per WF Optimization ---
strategy_zscore_mr_momo_param_ranges = {
    'ema_trend_span_range' : [30, 50, 100],
    'adx_period_range'     : [10, 14, 20],
    'adx_thresh_range'     : [18, 20, 25, 30],
    'z_window_range'       : [15, 20, 30, 40],
    # 'entry_z_range'        : [1.5, 2.0, 2.5, 3.0],
    # 'exit_z_range'         : [0.5, 1.0, 1.5, 2.0],
    'entry_z_range' : [2.0],       # soglia |Z| per l'entry (usa -entry_z)
    'exit_z_range'  : [2.0],       # soglia Z per l'exit
    'z_shift_range'        : [1, 2, 3],
    'donchian_range'       : [20, 40, 60],
    'momo_shift_range'     : [1, 2, 3]
}

# --- Funzione di strategia ---
def strategy_zscore_mr_momo(data, params, year=None):
    """
    Regime Range: mean-reversion con Z-score.
    Regime Trend: momentum con breakout Donchian + filtro EMA.
    """
    ema_span   = params.get('ema_trend_span_range')
    adx_p      = params.get('adx_period_range')
    adx_th     = float(params.get('adx_thresh_range'))
    z_win      = params.get('z_window_range')
    entry_z    = float(params.get('entry_z_range'))
    exit_z     = float(params.get('exit_z_range'))
    z_shift    = params.get('z_shift_range')
    don_w      = params.get('donchian_range')
    momo_shift = params.get('momo_shift_range')

    df = data.copy()

    df['Z']        = ind_zscore_mr_momo_z(df, window=z_win, ddof=0)
    df['EMA_T']    = ind_zscore_mr_momo_ema(df, span=ema_span)
    df['ADX']      = ind_zscore_mr_momo_adx(df, period=adx_p)
    df['DonU'], df['DonL'] = ind_zscore_mr_momo_donchian(df, window=don_w)

    if year is not None:
        df = df[df.index.year == year]

    # Regimi
    trending = (df['ADX'] > adx_th) & (df['Close'] > df['EMA_T'])
    ranging  = ~trending

    # RANGE entries/exits
    entries_range = (df['Z'] < -entry_z) & (df['Z'].shift(z_shift) >= -entry_z)
    exits_range   = (df['Z'] >  exit_z)  & (df['Z'].shift(z_shift) <=  exit_z)

    # TREND entries/exits
    entries_trend = (
        (df['Close'] > df['DonU']) &
        (df['Close'].shift(momo_shift) <= df['DonU'].shift(momo_shift)) &
        (df['Close'] > df['EMA_T']) &
        (df['ADX'] > adx_th)
    )
    exits_trend = (df['Close'] < df['EMA_T'])

    # Combinazione
    entries = (ranging & entries_range) | (trending & entries_trend)
    exits   = (ranging & exits_range)   | (trending & exits_trend)

    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy cog_qqe
############################

# === COG (Center of Gravity) ===
def ind_cog_qqe_cog(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    COG calcolato come media mobile pesata sui 'Close' con pesi 1..period,
    applicati a barre retroshiftate (i=0..period-1).
    """
    close = df['Close']
    denom = period * (period + 1) / 2.0
    # somma_{i=0..period-1} (i+1) * Close.shift(i)
    weighted_sum = sum((i + 1) * close.shift(i) for i in range(period))
    cog = weighted_sum / denom
    return cog

# === QQE (mod su RSI) ===
def ind_cog_qqe_qqe(df: pd.DataFrame,
                    rsi_period: int = 14,
                    smoothing_factor: int = 5,
                    wilders_period: int = 14):
    """
    QQE (variante reale su RSI) con smoothing e ATR sull'RSI.
    Ritorna: (smoothed_rsi, signal_line, atr_rsi)
    """
    close = df['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # RSI (EMA style)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # smoothing RSI
    smoothed_rsi = rsi.ewm(span=smoothing_factor, adjust=False, min_periods=1).mean()

    # ATR sull'RSI (True Range dell'RSI smussato)
    rsi_tr = smoothed_rsi.diff().abs()
    atr_rsi = rsi_tr.ewm(span=wilders_period, adjust=False, min_periods=1).mean()

    # trailing/linea QQE e sua signal
    d_factor = 4.236  # costante tipica nella letteratura QQE
    qqe_trailing = smoothed_rsi - (atr_rsi * d_factor)
    signal_line = qqe_trailing.ewm(span=smoothing_factor, adjust=False, min_periods=1).mean()

    return smoothed_rsi, signal_line, atr_rsi


# --- Griglia parametri per WF Optimization ---
strategy_cog_qqe_param_ranges = {
    'cog_range'       : range(5, 26, 7),   # period COG
    'rsi_range'       : range(8, 24, 4),   # period RSI base
    'smoothing_range' : range(3, 12, 4),   # smoothing RSI / signal
    'wilders_range'   : range(8, 24, 4),   # periodo ATR_RSI (tipo Wilder)
    'shift_range'     : range(10, 34, 12)   # barre per confronto momentum QQE
}


# --- Funzione di strategia ---
def strategy_cog_qqe(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Genera segnali (entries, exits) usando COG + QQE (mod su RSI).
    Regole (sul df completo; filtro per anno applicato SOLO come maschera a valle):
      Entry se:
        Close < COG
        AND QQE (smoothed RSI) > QQE.shift(shift)   [momentum RSI in miglioramento]
      Exit se:
        Close > COG
        AND QQE (smoothed RSI) < QQE.shift(shift)   [momentum RSI in deterioramento]

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    Il filtro year, se passato, MASCHERA i segnali mantenendo l’allineamento con l’indice originale.
    """

    # Leggi parametri (coerenti con le chiavi della griglia WFO)
    cog_p     = params.get('cog_range')
    rsi_p     = params.get('rsi_range')
    smooth_p  = params.get('smoothing_range')
    wilders_p = params.get('wilders_range')
    shift     = params.get('shift_range')

    df = data.copy()

    # --- Calcolo indicatori
    df['COG'] = ind_cog_qqe_cog(df, period=cog_p)
    df['QQE'], df['QQE_Signal'], df['ATR_RSI'] = ind_cog_qqe_qqe(
        df, rsi_period=rsi_p, smoothing_factor=smooth_p, wilders_period=wilders_p
    )

    # --- Filtro per anno dopo il calcolo degli indicatori ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Condizioni entry/exit
    entries = (df['Close'] < df['COG']) & (df['QQE'] > df['QQE'].shift(shift))
    exits   = (df['Close'] > df['COG']) & (df['QQE'] < df['QQE'].shift(shift))
    
    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy stdch_momvol
############################

# === STD Channel (SMA ± 2*STD) ===
def ind_stdch_momvol_channel(df: pd.DataFrame, sma_period: int = 20):
    sma = df['Close'].rolling(sma_period).mean()
    std = df['Close'].rolling(sma_period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return sma, upper, lower

# === Momentum & Momentum Delta ===
def ind_stdch_momvol_momentum(df: pd.DataFrame, momentum_period: int = 10):
    mom = df['Close'] - df['Close'].shift(momentum_period)
    mom_delta = mom - mom.shift(1)
    return mom, mom_delta

# === Volume Filter (spike su media mobile) ===
def ind_stdch_momvol_volume(df: pd.DataFrame, volume_period: int = 20):
    vol_avg = df['Volume'].rolling(volume_period).mean()
    vol_spike = df['Volume'] > vol_avg
    return vol_avg, vol_spike


# --- Griglia parametri per WF Optimization (conforme all'originale) ---
strategy_stdch_momvol_param_ranges = {
    'sma_range'      : range(5, 38, 8),  # periodo SMA / bande STD
    'momentum_range' : range(5, 38, 8),  # periodo momentum
    'volume_range'   : range(5, 38, 8),  # periodo media volume
}


# --- Funzione di strategia ---
def strategy_stdch_momvol(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry: Close > Upper  & Momentum > 0  & Momentum_Delta > 0  & Volume_Spike
    Exit : Close < Lower  & Momentum < 0  & Momentum_Delta < 0  & Volume_Spike
    Ordine: indicatori → filtro anno (slicing) → condizioni → shift (1 barra)
    """
    sma_period      = params.get('sma_range')
    momentum_period = params.get('momentum_range')
    volume_period   = params.get('volume_range')

    df = data.copy()

    # --- Calcolo indicatori sull'intero df ---
    df['SMA'], df['Upper'], df['Lower'] = ind_stdch_momvol_channel(df, sma_period=sma_period)
    df['Momentum'], df['Momentum_Delta'] = ind_stdch_momvol_momentum(df, momentum_period=momentum_period)
    df['Volume_Avg'], df['Volume_Spike'] = ind_stdch_momvol_volume(df, volume_period=volume_period)

    # --- Filtro per anno dopo il calcolo degli indicatori ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Condizioni entry/exit ---
    entries = (
        (df['Close'] > df['Upper']) &
        (df['Momentum'] > 0) &
        (df['Momentum_Delta'] > 0) &
        (df['Volume_Spike'])
    )
    exits = (
        (df['Close'] < df['Lower']) &
        (df['Momentum'] < 0) &
        (df['Momentum_Delta'] < 0) &
        (df['Volume_Spike'])
    )

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy vwap_std (No Mister K)
############################

# === VWAP ± STD Bands ===
def ind_vwap_std_bands(df: pd.DataFrame, window: int = 20):
    """
    Calcola VWAP, deviazione standard su VWAP (rolling 'window'), e bande:
    Upper = VWAP + STD, Lower = VWAP - STD.

    Se la colonna 'VWAP' è assente, viene calcolata come rolling
    somma(Close*Volume)/somma(Volume) sullo stesso 'window'.
    """
    if 'VWAP' in df.columns:
        vwap = df['VWAP']
    else:
        vwap = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()

    vwap_std = vwap.rolling(window=window).std()
    upper = vwap + vwap_std
    lower = vwap - vwap_std
    return vwap, vwap_std, upper, lower


# --- Griglia parametri per WF Optimization ---
# (soglia 0.5% fissa come nell'idea di partenza)
strategy_vwap_std_param_ranges = {
    'window_range': range(5, 31)  # finestra per VWAP e STD
}


# --- Funzione di strategia ---
def strategy_vwap_std(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia long-only su bande VWAP±STD con soglia 0.5%.

    Regole:
      Entry se: (Close - Lower)/Lower <= 0.005
      Exit  se: (Upper - Close)/Upper <= 0.005

    Ordine conforme:
      1) Calcolo indicatori sull’intero df
      2) Filtro anno con slicing
      3) Definizione entries/exits
      4) Shift di 1 barra e normalizzazione
    """
    window = params.get('window_range')
    df = data.copy()

    # 1) Calcolo indicatori
    df['VWAP'], df['VWAP_STD'], df['Upper'], df['Lower'] = ind_vwap_std_bands(df, window=window)

    # 2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Condizioni entry/exit (long-only) con soglia 0.5%
    thr = 0.005
    entries = ((df['Close'] - df['Lower']) / df['Lower'] <= thr)
    exits   = ((df['Upper'] - df['Close']) / df['Upper'] <= thr)

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    

############################
# Strategy vwap_z_ema_x
############################

# === VWAP & STD (rolling) ===
def ind_vwap_z_ema_x_vwap(df: pd.DataFrame, window: int = 20):
    """
    Se 'VWAP' esiste la usa; altrimenti calcola la VWAP rolling:
    VWAP = sum(Close*Volume)/sum(Volume) sullo stesso 'window'.
    Ritorna (vwap, vwap_std).
    """
    if 'VWAP' in df.columns:
        vwap = df['VWAP']
    else:
        vol_roll = df['Volume'].rolling(window=window).sum()
        vwap = (df['Close'] * df['Volume']).rolling(window=window).sum() / vol_roll
    vwap_std = vwap.rolling(window=window).std()
    return vwap, vwap_std

# === Filtro di trend (EMA) ===
def ind_vwap_z_ema_x_trend(df: pd.DataFrame, ema_period: int = 200):
    """
    EMA sul Close come filtro di regime.
    """
    ema = df['Close'].ewm(span=ema_period, adjust=False, min_periods=1).mean()
    return ema

# --- Griglia parametri per WF Optimization ---
strategy_vwap_z_ema_x_param_ranges = {
    'window_range'     : range(5, 38, 8),        # finestra VWAP/STD
    'ema_range'        : range(50, 316, 66),  # filtro trend EMA
    'z_entry_x10_range': range(5, 25, 5),        # soglia entry 0.5..2.0
    'z_exit_x10_range' : range(0, 20, 5),        # soglia exit  0.0..1.5
}

# --- Funzione di strategia ---
def strategy_vwap_z_ema_x(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Long-only su pullback a VWAP in uptrend.
      Z = (Close - VWAP) / STD(VWAP)
      Entry: (Close > EMA) & (Z <= -z_entry)
      Exit : (Z >= z_exit) | (Close < EMA)

    Ordine conforme:
      1) Calcolo indicatori  2) Slicing per anno  3) Condizioni  4) Shift 1 barra
    """
    window   = params.get('window_range')
    ema_p    = params.get('ema_range')
    z_in_x10 = params.get('z_entry_x10_range')
    z_out_x10= params.get('z_exit_x10_range')

    df = data.copy()

    # 1) Indicatori sull’intero df
    df['VWAP'], df['VWAP_STD'] = ind_vwap_z_ema_x_vwap(df, window=window)
    df['EMA'] = ind_vwap_z_ema_x_trend(df, ema_period=ema_p)
    df['Z'] = (df['Close'] - df['VWAP']) / df['VWAP_STD'].replace(0, np.nan)

    # 2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Condizioni entry/exit
    z_entry = (z_in_x10 or 10) / 10.0   # default 1.0 se non passato
    z_exit  = (z_out_x10 or 0) / 10.0   # default 0.0 se non passato

    entries = (df['Close'] > df['EMA']) & (df['Z'] <= -z_entry)
    exits   = (df['Z'] >= z_exit) | (df['Close'] < df['EMA'])

    # Buy/sell signals are shifted forward by one day to simulate realistic execution (act after a signal appears).
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

    
######################################################
# AI generated STRATEGIES based on "Rare" indicators
######################################################


#######################################
# Indicatori "Rare" (versioni robuste)
#######################################

# === GAPO (Gopalakrishnan Range Index) ===
def ind_rare_gapo(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    GAPO classico: log10(HighestHigh(period) - LowestLow(period)) / log10(period)
    """
    hh = df['High'].rolling(period, min_periods=1).max()
    ll = df['Low'].rolling(period, min_periods=1).min()
    rng = (hh - ll).replace(0, np.nan)
    gapo = np.log10(rng) / np.log10(period)
    return gapo.bfill().fillna(0.0)
    # return gapo.fillna(method='bfill').fillna(0.0)


# === TSI (True Strength Index) + Signal ===
def ind_rare_tsi(df: pd.DataFrame, r: int = 25, s: int = 13, signal_period: int = 7):
    """
    TSI standard: 100 * EMA_r(EMA_s(mtm)) / EMA_r(EMA_s(|mtm|))
    Ritorna: (tsi, tsi_signal)
    """
    mtm = df['Close'].diff()
    ema1_mtm = mtm.ewm(span=s, adjust=False, min_periods=1).mean()
    ema2_mtm = ema1_mtm.ewm(span=r, adjust=False, min_periods=1).mean()

    abs_mtm = mtm.abs()
    ema1_abs = abs_mtm.ewm(span=s, adjust=False, min_periods=1).mean()
    ema2_abs = ema1_abs.ewm(span=r, adjust=False, min_periods=1).mean().replace(0, np.nan)

    tsi = 100 * (ema2_mtm / ema2_abs)
    tsi_signal = tsi.ewm(span=signal_period, adjust=False, min_periods=1).mean()
    return tsi.fillna(0.0), tsi_signal.fillna(0.0)

# === KST (Know Sure Thing) + Signal ===
def ind_rare_kst(df: pd.DataFrame,
                 roc_periods=(10, 15, 20, 30),
                 sma_periods=(10, 10, 10, 15),
                 weights=(1, 2, 3, 4),
                 signal_period: int = 9):
    """
    KST classico: somma pesata di SMA(ROC_n) con pesi 1..4.
    Ritorna: (kst, kst_signal)
    """
    close = df['Close']
    rocs = [(close.pct_change(p) * 100.0) for p in roc_periods]
    smas = [roc.rolling(window=sma, min_periods=1).mean() for roc, sma in zip(rocs, sma_periods)]
    kst = sum(w * s for w, s in zip(weights, smas))
    kst_signal = kst.rolling(window=signal_period, min_periods=1).mean()
    return kst.fillna(0.0), kst_signal.fillna(0.0)

# === EFI (Elder Force Index) ===
def ind_rare_efi(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """
    EFI standard: EMA_period( (Close.diff)*Volume )
    """
    raw = df['Close'].diff() * df['Volume']
    efi = raw.ewm(span=period, adjust=False, min_periods=1).mean()
    return efi.fillna(0.0)

# === TII (Trend Intensity Index) ===
def ind_rare_tii(df: pd.DataFrame, period: int = 30) -> pd.Series:
    """
    TII: % di chiusure sopra la SMA(period) negli ultimi 'period' punti * 100 (0..100).
    """
    ma = df['Close'].rolling(period, min_periods=1).mean()
    above = (df['Close'] > ma).astype(int)
    tii = 100 * above.rolling(period, min_periods=1).mean()
    return tii.fillna(0.0)


############################
# Strategy gapo_tsi
############################

# --- Griglia parametri per WF Optimization ---
strategy_gapo_tsi_param_ranges = {
    'gapo_period'   : range(10, 55, 15),
    'tsi_r'         : range(20, 50, 10),
    'tsi_s'         : range(7, 23, 4),
    'tsi_signal'    : range(5, 17, 4),
    'gapo_low_thr'  : range(-3, 0),
    'gapo_high_thr' : range(0, 3)
}

# --- Funzione di strategia ---
def strategy_gapo_tsi(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry:  GAPO < gapo_low_thr  AND  TSI > TSI_signal
    Exit:   GAPO > gapo_high_thr OR   TSI < TSI_signal
    """
    gp   = params.get('gapo_period')
    r    = params.get('tsi_r')
    s    = params.get('tsi_s')
    sigp = params.get('tsi_signal')
    low  = params.get('gapo_low_thr')
    high = params.get('gapo_high_thr')

    df = data.copy()

    # 1) Indicatori
    df['GAPO'] = ind_rare_gapo(df, period=gp)
    df['TSI'], df['TSI_Signal'] = ind_rare_tsi(df, r=r, s=s, signal_period=sigp)

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    entries = (df['GAPO'] < low) & (df['TSI'] > df['TSI_Signal'])
    exits   = (df['GAPO'] > high) | (df['TSI'] < df['TSI_Signal'])

    # 4) Shift + normalizzazione
    return (
        entries.shift(1).astype(bool).fillna(False),
        exits.shift(1).astype(bool).fillna(False)
    )


############################
# Strategy kst_efi
############################

# --- Griglia parametri per WF Optimization ---
strategy_kst_efi_param_ranges = {
    # ROC (4×4×4×3 = 192)
    'roc1'      : range(10, 17, 2),   # {10,12,14,16}        (4)
    'roc2'      : range(15, 23, 2),   # {15,17,19,21}        (4)
    'roc3'      : range(20, 29, 3),   # {20,23,26,29}        (4)
    'roc4'      : range(30, 39, 3),   # {30,33,36}           (3)

    # smoothing KST (2×2×2×1 = 8)
    'sma1'      : range(10, 13, 2),   # {10,12}              (2)
    'sma2'      : range(10, 13, 2),   # {10,12}              (2)
    'sma3'      : range(10, 13, 2),   # {10,12}              (2)
    'sma4'      : range(15, 16, 1),   # {15}                 (1)

    # segnale KST (3)
    'kst_signal': range(7, 12, 2),    # {7,9,11}             (3)

    # EFI (4)
    'efi_period': range(10, 17, 2),   # {10,12,14,16}        (4)
}
# Totale = 192 * 8 * 3 * 4 = 18.432  (< 50k)

# --- Funzione di strategia ---
def strategy_kst_efi(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry:  KST > KST_signal  AND  EFI > 0
    Exit:   KST < KST_signal  OR   EFI < 0
    """
    roc1 = params.get('roc1'); roc2 = params.get('roc2')
    roc3 = params.get('roc3'); roc4 = params.get('roc4')
    sma1 = params.get('sma1'); sma2 = params.get('sma2')
    sma3 = params.get('sma3'); sma4 = params.get('sma4')
    ksig = params.get('kst_signal')
    ep   = params.get('efi_period')

    df = data.copy()

    # 1) Indicatori
    df['KST'], df['KST_Signal'] = ind_rare_kst(
        df,
        roc_periods=(roc1, roc2, roc3, roc4),
        sma_periods=(sma1, sma2, sma3, sma4),
        weights=(1, 2, 3, 4),
        signal_period=ksig
    )
    df['EFI'] = ind_rare_efi(df, period=ep)

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    entries = (df['KST'] > df['KST_Signal']) & (df['EFI'] > 0)
    exits   = (df['KST'] < df['KST_Signal']) | (df['EFI'] < 0)

    # 4) Shift + normalizzazione
    return (
        entries.shift(1).astype(bool).fillna(False),
        exits.shift(1).astype(bool).fillna(False)
    )


############################
# Strategy tii_filter
############################

# --- Griglia parametri per WF Optimization ---
strategy_tii_filter_param_ranges = {
    'tii_period'  : range(20, 73, 13),   # finestra TII
    'ma_period'   : range(50, 250, 50), # MA di trend
    'low_thr'     : range(35, 55, 5),   # tipico 40
    'high_thr'    : range(55, 75, 5)    # tipico 60
}

# --- Funzione di strategia ---
def strategy_tii_filter(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Trend-following con filtro d'intensità:
      Entry:  TII > high_thr  AND  Close > MA
      Exit:   TII < low_thr   OR   Close < MA
    """
    tp  = params.get('tii_period')
    mp  = params.get('ma_period')
    lo  = params.get('low_thr')
    hi  = params.get('high_thr')

    df = data.copy()

    # 1) Indicatori
    df['TII'] = ind_rare_tii(df, period=tp)
    df['MA']  = df['Close'].rolling(mp, min_periods=1).mean()

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    entries = (df['TII'] > hi) & (df['Close'] > df['MA'])
    exits   = (df['TII'] < lo) | (df['Close'] < df['MA'])

    # 4) Shift + normalizzazione
    return (
        entries.shift(1).astype(bool).fillna(False),
        exits.shift(1).astype(bool).fillna(False)
    )


############################
# Strategy tii_meanrev
############################

# --- Griglia parametri per WF Optimization ---
strategy_tii_meanrev_param_ranges = {
    'tii_period'  : range(20, 80, 20),
    'ma_period'   : range(50, 250, 50),
    'low_thr'     : range(30, 50, 5),   # area "ipervenduto" TII
    'high_thr'    : range(55, 75, 5),   # presa profitti su ipercomprato TII
    'cross_shift' : range(1, 5, 1)      # barre per conferma del cross su low_thr
}

# --- Funzione di strategia ---
def strategy_tii_meanrev(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Mean-reversion con conferma di forza:
      Entry:  (TII cross-up sopra low_thr con conferma 'cross_shift')  AND  Close > MA
      Exit:   (TII > high_thr)  OR  (Close < MA)
    """
    tp   = params.get('tii_period')
    mp   = params.get('ma_period')
    lo   = params.get('low_thr')
    hi   = params.get('high_thr')
    shft = params.get('cross_shift')

    df = data.copy()

    # 1) Indicatori
    df['TII'] = ind_rare_tii(df, period=tp)
    df['MA']  = df['Close'].rolling(mp, min_periods=1).mean()

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    # Cross-up confermato: TII supera 'lo' e resta sopra rispetto a shft barre fa
    cross_up = (df['TII'] > lo) & (df['TII'].shift(shft) <= lo)
    entries  = cross_up & (df['Close'] > df['MA'])
    exits    = (df['TII'] > hi) | (df['Close'] < df['MA'])

    # 4) Shift + normalizzazione
    return (
        entries.shift(1).astype(bool).fillna(False),
        exits.shift(1).astype(bool).fillna(False)
    )

# #
# # The 5 Rare Indicators Every Trader Should Know
# #

# # ---- Gopalakrishnan Range Index (GAPO) ----
# def gapo(df, period=14):
#     df['range'] = df['High'] - df['Low']
#     df['max_range'] = df['range'].rolling(window=period).max()
#     df['GAPO'] = df['range'] / df['max_range'] * 100
#     return df

# # ---- True Strength Index (TSI) ----
# def tsi(df, short_period=25, long_period=13):
#     diff = df['Close'].diff()
#     abs_diff = diff.abs()
#     momentum = diff.ewm(span=short_period).mean() / abs_diff.ewm(span=long_period).mean()
#     df['TSI'] = momentum * 100
#     return df

# # ---- Know Sure Thing (KST) ----
# def kst(df, long_period=34, short_period=23, signal_period=10):
#     roc1 = df['Close'].pct_change(periods=10) * 100
#     roc2 = df['Close'].pct_change(periods=15) * 100
#     roc3 = df['Close'].pct_change(periods=20) * 100
#     roc4 = df['Close'].pct_change(periods=30) * 100
    
#     kst = (roc1.rolling(window=long_period).mean() +
#            roc2.rolling(window=short_period).mean() +
#            roc3.rolling(window=short_period).mean() +
#            roc4.rolling(window=long_period).mean())
    
#     df['KST'] = kst
#     df['KST_signal'] = kst.rolling(window=signal_period).mean()
#     return df

# # ---- Elder Force Index (EFI) ----
# def efi(df, period=13):
#     df['EFI'] = (df['Close'].diff(periods=1) * df['Volume']) / df['Close']
#     df['EFI'] = df['EFI'].rolling(window=period).mean()
#     return df

# # ---- Trend Intensity Index (TII) ----
# def tii(df, period=14):
#     df['trend'] = df['Close'] - df['Close'].shift(period)
#     df['trend_intensity'] = df['trend'] / df['Close'].rolling(window=period).std()
#     return df


# finalize_strategy_registry()


# From Medium Posts

############################
# Strategy fvg (v3 · long-only · trend+ATR+RR)
############################

# === Indicatori di supporto ===
def ind_fvg_bullish(df: pd.DataFrame,
                    lookback_period: int = 10,
                    body_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Rileva FVG rialzisti (gap fra high di i-2 e low di i) se la candela i-1 è abbastanza 'forte'.
    Ritorna DataFrame con colonne:
      - FVG_START (first_high)  [limite inferiore del gap]
      - FVG_END   (third_low)   [limite superiore del gap]
    """
    idx = df.index
    start_list, end_list = [np.nan]*len(df), [np.nan]*len(df)

    for i in range(2, len(df)):
        first_high   = df['High'].iloc[i-2]
        middle_open  = df['Open'].iloc[i-1]
        middle_close = df['Close'].iloc[i-1]
        third_low    = df['Low'].iloc[i]

        # body medio su lookback
        a, b = max(0, i-1-lookback_period), i-1
        prev_bodies = (df['Close'].iloc[a:b] - df['Open'].iloc[a:b]).abs()
        avg_body = prev_bodies.mean()
        if not np.isfinite(avg_body) or avg_body == 0:
            avg_body = 0.001

        middle_body = abs(middle_close - middle_open)

        # FVG bullish: Low(i) > High(i-2) e candela (i-1) 'forte'
        if (third_low > first_high) and (middle_body > avg_body * body_multiplier):
            start_list[i] = float(first_high)
            end_list[i]   = float(third_low)

    return pd.DataFrame({
        'FVG_START': pd.Series(start_list, index=idx),
        'FVG_END'  : pd.Series(end_list,   index=idx),
    })


def ind_fvg_trend_atr(df: pd.DataFrame,
                      ema_span: int = 200,
                      atr_len: int = 14) -> pd.DataFrame:
    """
    Trend & volatilità: EMA, ATR (tipo Wilder EWM).
    """
    ema = df['Close'].ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()

    prev_close = df['Close'].shift(1)
    tr = np.maximum(np.maximum((df['High'] - df['Low']).abs(), (df['High'] - prev_close).abs()), (df['Low'] - prev_close).abs())
    atr = tr.ewm(alpha=1/atr_len, adjust=False, min_periods=1).mean()

    return pd.DataFrame({'EMA': ema, 'ATR': atr})


# --- Griglia parametri per WFO (contenuta) ---
strategy_fvg_param_ranges = {
    'lookback_range' : [5, 10, 15],     # corpo medio per "forza" candela 2
    'bodymult_range' : [1.0, 1.5, 2.0], # richiesta di displacement
    'wait_range'     : [4, 8, 12],      # barre max per retrace nel gap
    'ema_span_range' : [150, 200, 250], # filtro trend
    'atr_len_range'  : [10, 14],        # ATR per stop buffer
    'atr_k_range'    : [0.0, 0.5],      # buffer stop (0 = secco su START)
    'rr_mult_range'  : [1.5, 2.0, 3.0], # take-profit a multiplo del rischio
}


# --- Strategia FVG long con TP RR e stop ATR ---
def strategy_fvg(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole (long-only):
      - FVG bullish nasce: zona [START, END] 'pendente' per wait_range barre.
      - Filtro trend: Close > EMA.
      - ENTRY: prima barra in cui Low <= END e Close >= START (tocca il gap ma chiude sopra),
               con Close > EMA.
      - STOP: START - atr_k*ATR (al momento dell'entry).
      - TAKE PROFIT: entry + rr_mult * (entry - stop). Se High >= TP → exit.
      - INVALIDAZIONE pre-entry: se passa wait_range senza toccare il gap o Close <= START.
      - 1 trade alla volta; segnali shiftati di 1 barra.
    """
    df = data.copy()

    # 1) Indicatori su tutto il df
    lookback = params.get('lookback_range')
    bodymult = params.get('bodymult_range')
    max_wait = params.get('wait_range')
    ema_span = params.get('ema_span_range')
    atr_len  = params.get('atr_len_range')
    atr_k    = params.get('atr_k_range')
    rr_mult  = params.get('rr_mult_range')

    fvg = ind_fvg_bullish(df, lookback_period=lookback, body_multiplier=bodymult)
    tv  = ind_fvg_trend_atr(df, ema_span=ema_span, atr_len=atr_len)
    df  = pd.concat([df, fvg, tv], axis=1)

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Segnali
    entries = pd.Series(False, index=df.index)
    exits   = pd.Series(False, index=df.index)

    # Stato
    pending_start = None
    pending_end   = None
    pending_age   = 0
    in_position   = False
    entry_i       = None
    entry_px      = None
    stop_lvl      = None
    tp_lvl        = None

    for i, (ts, row) in enumerate(df.iterrows()):
        # Nuovo FVG?
        if np.isfinite(row.get('FVG_START', np.nan)) and np.isfinite(row.get('FVG_END', np.nan)):
            pending_start = float(row['FVG_START'])
            pending_end   = float(row['FVG_END'])
            pending_age   = 0

        if not in_position and (pending_start is not None):
            pending_age += 1

            # invalidazione pre-entry
            if (pending_age > max_wait) or (row['Close'] <= pending_start):
                pending_start = None; pending_end = None; pending_age = 0
            else:
                # Entry: tocca il gap (Low <= END) e chiude sopra START, in trend (Close > EMA)
                if (row['Low'] <= pending_end) and (row['Close'] >= pending_start) and (row['Close'] > row['EMA']):
                    entries.iloc[i] = True
                    in_position = True
                    entry_i  = i
                    entry_px = float(row['Close'])
                    # stop e target fissati all'entry
                    stop_lvl = pending_start - float(atr_k) * float(row['ATR'])
                    # assicura stop sotto il prezzo
                    if stop_lvl >= entry_px:
                        stop_lvl = pending_start * 0.999
                    risk    = entry_px - stop_lvl
                    tp_lvl  = entry_px + float(rr_mult) * risk
                    # il gap usato non serve più
                    pending_start = None; pending_end = None; pending_age = 0

        if in_position:
            # hit TP intrabar? (usa High)
            hit_tp  = (row['High'] >= tp_lvl) if (tp_lvl is not None) else False
            # stop su Close
            hit_stp = (row['Close'] <= stop_lvl) if (stop_lvl is not None) else False

            if hit_tp or hit_stp:
                exits.iloc[i] = True
                in_position = False
                entry_i = None; entry_px = None
                stop_lvl = None; tp_lvl = None

    # 4) Shift 1 barra e normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy fvg_trail (long-only · trend+ATR trailing)
############################

# --- Griglia WFO (contenuta) ---
strategy_fvg_trail_param_ranges = {
    'lookback_range' : [5, 10, 15],
    'bodymult_range' : [1.0, 1.5],
    'wait_range'     : [6, 12],        # barre per “toccare” il gap
    'ema_span_range' : [150, 200, 250],
    'atr_len_range'  : [10, 14],
    'trail_k_range'  : [2.0, 3.0],     # Chandelier: stop = max_high - k*ATR
    'ema_exit_range' : [0, 1],         # 1 = aggiunge exit su Close<EMA
}

# --- Strategia con trailing ATR (niente TP) ---
def strategy_fvg_trail(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole (long-only):
      • Nasce un FVG bullish → zona [START, END] 'pendente' per wait_range barre.
      • Filtro trend: Close > EMA.
      • ENTRY: prima barra in cui Low <= END e Close >= START e Close > EMA.
      • STOP iniziale: START (o poco sotto se preferisci) → da lì parte trailing.
      • TRAILING: Chandelier long = max(High dall’entry) - trail_k*ATR(barra corrente).
      • EXIT: Close <= max(trailing_stop, EMA) se ema_exit_range==1, altrimenti solo trailing.
      • 1 trade alla volta; segnali shiftati di 1 barra.
    """
    df = data.copy()

    # 1) Indicatori su tutto il df
    lookback = params.get('lookback_range')
    bodymult = params.get('bodymult_range')
    max_wait = params.get('wait_range')
    ema_span = params.get('ema_span_range')
    atr_len  = params.get('atr_len_range')
    trail_k  = params.get('trail_k_range')
    ema_exit = bool(params.get('ema_exit_range'))

    fvg = ind_fvg_bullish(df, lookback_period=lookback, body_multiplier=bodymult)
    tv  = ind_fvg_trend_atr(df, ema_span=ema_span, atr_len=atr_len)
    df  = pd.concat([df, fvg, tv], axis=1)

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Segnali
    entries = pd.Series(False, index=df.index)
    exits   = pd.Series(False, index=df.index)

    # Stato
    pending_start = None; pending_end = None; pending_age = 0
    in_position = False; entry_i = None
    max_high_since_entry = None; trail_stop = None

    for i, (ts, row) in enumerate(df.iterrows()):
        # nuovo FVG?
        if np.isfinite(row.get('FVG_START', np.nan)) and np.isfinite(row.get('FVG_END', np.nan)):
            pending_start = float(row['FVG_START']); pending_end = float(row['FVG_END']); pending_age = 0

        if not in_position and (pending_start is not None):
            pending_age += 1
            # invalidazione pre-entry
            if (pending_age > max_wait) or (row['Close'] <= pending_start):
                pending_start = None; pending_end = None; pending_age = 0
            else:
                # entry: tocca gap e chiude sopra START, con trend attivo
                if (row['Low'] <= pending_end) and (row['Close'] >= pending_start) and (row['Close'] > row['EMA']):
                    entries.iloc[i] = True
                    in_position = True
                    entry_i = i
                    max_high_since_entry = float(row['High'])
                    # stop iniziale: al livello START (con leggero buffer facoltativo)
                    trail_stop = max_high_since_entry - float(trail_k) * float(row['ATR'])
                    if trail_stop >= row['Close']:   # safety
                        trail_stop = pending_start * 0.999
                    # consumato il gap
                    pending_start = None; pending_end = None; pending_age = 0

        if in_position:
            # aggiorna massimo e trailing
            if float(row['High']) > max_high_since_entry:
                max_high_since_entry = float(row['High'])
            dyn_trail = max_high_since_entry - float(trail_k) * float(row['ATR'])
            trail_stop = max(trail_stop, dyn_trail)  # non si allarga mai

            # condizione di uscita
            hit_trail = row['Close'] <= trail_stop
            hit_ema   = (row['Close'] <= row['EMA']) if ema_exit else False
            if hit_trail or hit_ema:
                exits.iloc[i] = True
                in_position = False
                entry_i = None
                max_high_since_entry = None
                trail_stop = None

    # 4) Shift 1 barra
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
################################
# Strategy stochastic_breakout
################################

# === Indicatore Stocastico (%K e %D) ===
def ind_stochastic_breakout(df: pd.DataFrame,
                            k_period: int = 14,
                            d_period: int = 3,
                            smooth_k: int = 1) -> tuple[pd.Series, pd.Series]:
    """
    Calcola lo Stochastic Oscillator (%K, %D).
    %K: (Close - LowMin) / (HighMax - LowMin) * 100
    %D: SMA di %K.
    """
    low_min = df['Low'].rolling(k_period, min_periods=1).min()
    high_max = df['High'].rolling(k_period, min_periods=1).max()
    k_fast = 100 * (df['Close'] - low_min) / (high_max - low_min)
    k_smooth = k_fast.rolling(smooth_k, min_periods=1).mean()
    d_line = k_smooth.rolling(d_period, min_periods=1).mean()
    return k_smooth, d_line


# --- Griglia parametri per WF Optimization ---
strategy_stochastic_breakout_param_ranges = {
    'k_period'   : range(10, 21, 2),   # periodo %K
    'd_period'   : range(2, 6, 1),     # periodo %D
    'smooth_k'   : range(1, 4, 1),     # smoothing %K
    'lookback'   : range(10, 31, 5)    # barre per breakout su resistenza
}


# --- Funzione di strategia ---
def strategy_stochastic_breakout(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Genera segnali usando lo Stochastic Oscillator.
    Regole:
      Entry long se:
        - %K incrocia %D verso il basso
        - %K era > 80 (zona overbought)
        - Prezzo rompe il massimo delle ultime 'lookback' barre
      Exit long se:
        - %K < 20 (esaurimento momentum)
        - oppure prezzo scende sotto la media dei minimi delle ultime 'lookback' barre
    Segnali shiftati di 1 barra per evitare look-ahead.
    """
    k_p   = params.get('k_period')
    d_p   = params.get('d_period')
    sm_k  = params.get('smooth_k')
    lb    = params.get('lookback')

    df = data.copy()

    # --- Calcolo indicatori
    df['%K'], df['%D'] = ind_stochastic_breakout(df, k_period=k_p, d_period=d_p, smooth_k=sm_k)

    # --- Filtro per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Condizioni entry
    crossover_down = (df['%K'].shift(1) > df['%D'].shift(1)) & (df['%K'] < df['%D'])
    overbought = df['%K'].shift(1) > 80
    breakout = df['Close'] > df['High'].shift(1).rolling(lb, min_periods=1).max()
    entries = crossover_down & overbought & breakout

    # --- Condizioni exit
    exits = (df['%K'] < 20) | (df['Close'] < df['Low'].shift(1).rolling(lb, min_periods=1).mean())

    # --- Shift segnali
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy hma_volume
############################

# === Hull Moving Average (HMA) ===
def ind_hma_volume_hma(series: pd.Series, period: int) -> pd.Series:
    """
    Calcola la Hull Moving Average (HMA) standard su una serie.
    """
    wma_half = series.rolling(period // 2).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    wma_full = series.rolling(period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(int(np.sqrt(period))).mean()
    return hma

def ind_hma_volume_vhma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """
    Calcola la Hull Moving Average pesata per il volume (VHMA).
    """
    nominal = close * volume
    wma_half = (nominal.rolling(period // 2).sum() / volume.rolling(period // 2).sum())
    wma_full = (nominal.rolling(period).sum() / volume.rolling(period).sum())
    diff = 2 * wma_half - wma_full
    vhma = diff.rolling(int(np.sqrt(period))).mean()
    return vhma


# --- Griglia parametri per WF Optimization ---
strategy_hma_volume_param_ranges = {
    'short_period' : range(10, 21, 2),   # HMA breve
    'long_period'  : range(20, 41, 2),   # HMA lunga
    'use_volume'   : range(0, 2, 1)      # 0=price HMA, 1=volume HMA
}


# --- Funzione di strategia ---
def strategy_hma_volume(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia basata su Hull Moving Average (HMA) e variante volume-weighted.
    Entry long se HMA corta > HMA lunga (Golden Cross).
    Exit se HMA corta < HMA lunga (Death Cross).
    Se 'use_volume'=1, usa la HMA volume-based (VHMA).
    """

    short_p = params.get('short_period')
    long_p  = params.get('long_period')
    use_vol = params.get('use_volume')

    df = data.copy()

    if use_vol == 1:
        df['HMA_short'] = ind_hma_volume_vhma(df['Close'], df['Volume'], short_p)
        df['HMA_long']  = ind_hma_volume_vhma(df['Close'], df['Volume'], long_p)
    else:
        df['HMA_short'] = ind_hma_volume_hma(df['Close'], short_p)
        df['HMA_long']  = ind_hma_volume_hma(df['Close'], long_p)

    # --- Filtro anno ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Segnali ---
    entries = df['HMA_short'] > df['HMA_long']
    exits   = df['HMA_short'] < df['HMA_long']

    # --- Shift di 1 barra ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits


############################
# Strategy wad_cks
############################

# === WAD (Williams Accumulation/Distribution) ===
def ind_wad_cks_wad(df: pd.DataFrame) -> pd.Series:
    """
    Calcolo vettoriale del WAD.
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    prev_close = close.shift(1)

    # Accumulation/Distribution (AD) step
    ad_up   = close - np.minimum(low, prev_close)
    ad_down = close - np.maximum(high, prev_close)
    ad = np.where(close > prev_close, ad_up,
         np.where(close < prev_close, ad_down, 0.0))

    wad = pd.Series(np.nan_to_num(ad).cumsum(), index=df.index, name='WAD')
    return wad

# === Chande Kroll Stop (versione coerente con snippet fornito) ===
def ind_wad_cks_cks(df: pd.DataFrame,
                    atr_period: int = 10,
                    stop_period: int = 10,
                    multiplier: float = 1.5) -> tuple[pd.Series, pd.Series]:
    """
    High/Low stop come da definizione semplificata proposta:
    ATR = rolling mean di (max(High, Low) - min(Low, High)) = (High - Low).
    """
    high = df['High']
    low  = df['Low']

    tr = (high.combine(low, max) - low.combine(high, min))  # equivalente a (High - Low)
    atr = tr.rolling(window=atr_period, min_periods=1).mean()

    high_stop = high.rolling(window=stop_period, min_periods=1).max() - multiplier * atr
    low_stop  = low.rolling(window=stop_period,  min_periods=1).min() + multiplier * atr
    high_stop.name = 'CKS_High'
    low_stop.name  = 'CKS_Low'
    return high_stop, low_stop

# --- Griglia parametri per WF Optimization ---
strategy_wad_cks_param_ranges = {
    'atr_period' : range(5, 37, 8),     # 5..29 step 3
    'stop_period': range(5, 37, 8),     # 5..29 step 3
    'multiplier' : [1.0, 1.5, 2.0],     # 3 valori
    'rise_shift' : range(5, 38, 8),     # 5,10,15,20,25,30
    'fall_shift' : range(5, 38, 8)      # 5,10,15,20,25,30
}
# Totale combinazioni = 9 * 9 * 3 * 6 * 6 = 8.748 (griglia contenuta)

# --- Funzione di strategia ---
def strategy_wad_cks(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Genera segnali (entries, exits) usando WAD + Chande Kroll Stop.
    Regole segnale (coerenti con lo snippet fornito):
      Entry se:
        (CKS_High > CKS_Low) AND (WAD > WAD.shift(rise_shift)) AND (CKS_High > Close)
      Exit se:
        (CKS_High < CKS_Low) AND (WAD < WAD.shift(fall_shift)) AND (CKS_Low  < Close)

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    Il filtro year, se passato, esegue slicing del df DOPO il calcolo indicatori.
    """
    # 1) Calcolo indicatori sull’intero df
    atr_p   = int(params.get('atr_period'))
    stop_p  = int(params.get('stop_period'))
    mult    = float(params.get('multiplier'))
    rshift  = int(params.get('rise_shift'))
    fshift  = int(params.get('fall_shift'))

    df = data.copy()
    df['WAD'] = ind_wad_cks_wad(df)
    df['CKS_High'], df['CKS_Low'] = ind_wad_cks_cks(df, atr_period=atr_p, stop_period=stop_p, multiplier=mult)

    # 2) Slicing per anno (dopo il calcolo indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Definizione entries/exits sul df già filtrato
    wad_rising  = df['WAD'] > df['WAD'].shift(rshift)
    wad_falling = df['WAD'] < df['WAD'].shift(fshift)

    entries = (df['CKS_High'] > df['CKS_Low']) & wad_rising & (df['CKS_High'] > df['Close'])
    exits   = (df['CKS_High'] < df['CKS_Low']) & wad_falling & (df['CKS_Low']  < df['Close'])

    # 4) Shift di 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy macd_adx_sma
############################

# === MACD ===
def ind_macd_adx_sma_macd(df: pd.DataFrame,
                          macd_fast: int = 12,
                          macd_slow: int = 26,
                          macd_signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """
    Restituisce (MACD, MACD_Signal) usando EMA su Close.
    """
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal_line = macd.ewm(span=macd_signal, adjust=False).mean()
    macd.name = 'MACD'
    macd_signal_line.name = 'MACD_Signal'
    return macd, macd_signal_line

# === ADX (coerente con lo snippet fornito) ===
def ind_macd_adx_sma_adx(df: pd.DataFrame, adx_window: int = 14) -> pd.Series:
    """
    ADX semplificato: usa TR= max(high-low, |high-prev_close|, |low-prev_close|),
    DI+ e DI- con rolling mean e ADX = rolling mean del DX.
    """
    high, low, close = df['High'], df['Low'], df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0  # coerente allo snippet dell'utente (può produrre DI- negativi)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = tr.rolling(window=adx_window, min_periods=1).mean()

    plus_di = 100 * (plus_dm.rolling(window=adx_window, min_periods=1).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=adx_window, min_periods=1).mean() / atr.replace(0, np.nan))
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx = dx.rolling(window=adx_window, min_periods=1).mean().fillna(0.0)
    adx.name = 'ADX'
    return adx

# === SMA ===
def ind_macd_adx_sma_sma(df: pd.DataFrame, sma_window: int = 20) -> pd.Series:
    sma = df['Close'].rolling(window=sma_window, min_periods=1).mean()
    sma.name = 'SMA'
    return sma

# --- Griglia parametri per WF Optimization ---
strategy_macd_adx_sma_param_ranges = {
    'macd_fast'   : range(8, 14, 2),        # 8..12
    'macd_slow'   : range(20, 28, 2),       # 20..26
    'macd_signal' : range(7, 11),        # 7..10
    'adx_window'  : range(10, 23, 6),       # 10..20
    'sma_window'  : range(5, 61, 28)      # 5..49 step 2
}
# Combinazioni totali = 5 * 7 * 4 * 11 * 23 = 35,420  (sotto 50k)

# --- Funzione di strategia ---
def strategy_macd_adx_sma(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Segnali long-only basati su:
      Entry se:
        (MACD > MACD_Signal) &
        (MACD.shift(1) <= MACD_Signal.shift(1)) &
        (MACD > 0) &
        (ADX > 20) &
        (Close > SMA)

      Exit se:
        (MACD < MACD_Signal) &
        (MACD.shift(1) >= MACD_Signal.shift(1)) &
        (MACD < 0) &
        (Close < SMA)

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    # 1) Calcolo indicatori sull’intero df
    macd_fast   = int(params.get('macd_fast'))
    macd_slow   = int(params.get('macd_slow'))
    macd_signal = int(params.get('macd_signal'))
    adx_window  = int(params.get('adx_window'))
    sma_window  = int(params.get('sma_window'))

    df = data.copy()
    df['MACD'], df['MACD_Signal'] = ind_macd_adx_sma_macd(df, macd_fast, macd_slow, macd_signal)
    df['ADX'] = ind_macd_adx_sma_adx(df, adx_window)
    df['SMA'] = ind_macd_adx_sma_sma(df, sma_window)

    # 2) Slicing per anno (dopo il calcolo degli indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Definizione entries/exits sul df già filtrato
    entries = (
        (df['MACD'] > df['MACD_Signal']) &
        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)) &
        (df['MACD'] > 0) &
        (df['ADX'] > 20) &
        (df['Close'] > df['SMA'])
    )

    exits = (
        (df['MACD'] < df['MACD_Signal']) &
        (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)) &
        (df['MACD'] < 0) &
        (df['Close'] < df['SMA'])
    )

    # 4) Shift di 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy macd_cross_hyst
############################

# === MACD (base) ===
def ind_macd_cross_hyst_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calcola MACD (EMA fast - EMA slow) e linea di segnale su 'Close'.
    Ritorna: (macd, macd_signal, macd_delta)
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_delta = macd - macd_signal
    return macd, macd_signal, macd_delta


# === Soglia dinamica sull'isteresi del MACD ===
def ind_macd_cross_hyst_threshold(df: pd.DataFrame, hyst_period: int = 20, k10: int = 20) -> pd.Series:
    """
    Soglia dinamica = k * ATR-like del macd_delta.
    - hyst_period: periodo Wilder per smussare l'ampiezza del delta (tipo ATR).
    - k10: fattore moltiplicativo *10 (es. 15 -> 1.5).
    Ritorna una Serie 'th' >= 0 da usare come banda di isteresi.
    """
    # Nota: questa funzione presuppone che df contenga 'MACD_DELTA'
    delta = df['MACD_DELTA']
    # "True Range" del delta: ampiezza assoluta dei movimenti
    tr_delta = delta.diff().abs()
    atr_like = tr_delta.ewm(alpha=1 / hyst_period, adjust=False, min_periods=1).mean()
    k = k10 / 10.0
    th = k * atr_like
    return th


# --- Griglia parametri per WF Optimization (≈ 4.9k combinazioni, compatta) ---
strategy_macd_cross_hyst_param_ranges = {
    'fast_range'       : range(8, 20, 4),    # 8,10,12,14,16
    'slow_range'       : range(20, 36, 4),   # 20..32
    'signal_range'     : range(7, 15, 2),    # 7,9,11,13
    'hyst_period_rng'  : range(10, 36, 6),   # 10,15,20,25,30
    'hyst_k10_rng'     : range(10, 50, 10)    # 1.0..4.0 (step 0.5) tramite *10
}


# --- Funzione di strategia ---
def strategy_macd_cross_hyst(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    MACD con banda di isteresi (dead-zone) sul differenziale MACD-Segnale.
    Logica long-only:
      - Entry se macd_delta attraversa +soglia (uscita dalla banda verso l'alto)
      - Exit  se macd_delta attraversa -soglia (uscita dalla banda verso il basso)

    La banda di isteresi riduce i falsi incroci in laterale, preservando l'esposizione nei trend.
    I segnali sono shiftati di 1 barra per evitare look-ahead (esecuzione realistica su Open).
    """

    fast_p   = params.get('fast_range')
    slow_p   = params.get('slow_range')
    signal_p = params.get('signal_range')
    hper     = params.get('hyst_period_rng')
    k10      = params.get('hyst_k10_rng')

    df = data.copy()

    # 1) --- Calcolo indicatori sull’intero df
    df['MACD'], df['MACD_Signal'], df['MACD_DELTA'] = ind_macd_cross_hyst_macd(
        df, fast=fast_p, slow=slow_p, signal=signal_p
    )
    df['HYST_TH'] = ind_macd_cross_hyst_threshold(df, hyst_period=hper, k10=k10)

    # 2) --- Slicing per anno (dopo aver calcolato gli indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) --- Condizioni entry/exit sul df già filtrato
    # Attraversamenti "forti" con banda di isteresi simmetrica
    th = df['HYST_TH']

    # Cross-up: delta da <= +th (o sotto) a > +th
    cross_up   = (df['MACD_DELTA'] >  th) & (df['MACD_DELTA'].shift(1) <= th.shift(1))
    # Cross-down: delta da >= -th (o sopra) a < -th
    cross_down = (df['MACD_DELTA'] < -th) & (df['MACD_DELTA'].shift(1) >= (-th).shift(1))

    entries = cross_up
    exits   = cross_down

    # 4) --- Shift 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy mom_trend_atr
############################

# === SMA di trend (long) ===
def ind_mom_trend_atr_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False).mean()

# === Momentum assoluto (ROC) ===
def ind_mom_trend_atr_roc(df: pd.DataFrame, lookback: int = 126) -> pd.Series:
    close = df['Close']
    return (close / close.shift(lookback) - 1.0)

# === ATR “classico” ===
def ind_mom_trend_atr_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low), (high - prev_close).abs()), (low - prev_close).abs())
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    return atr


# --- Griglia parametri per WF Optimization (≈ 9,9k combinazioni) ---
# Obiettivo: battere B&H con migliore profilo rischio/rendimento su SPY,
# mantenendo esposizione elevata ma controllando i rientri in laterale.
strategy_mom_trend_atr_param_ranges = {
    'sma_l_range'    : range(150, 300, 50),  # 150..250 (11)
    'roc_lb_range'   : range(60, 260, 50),   # 60,90,120,150,180,210 (6)
    'atr_p_range'    : range(10, 34, 12),     # 10,15,20,25,30 (5)
    'k_entry10_rng'  : range(10, 34, 12),     # 1.0..3.0 ATR (5)
    'k_exit10_rng'   : range(15, 47, 16)      # 1.5..4.0 ATR (6)
}
# Tot: 11*6*5*5*6 = 9.900 (ben sotto la nostra soglia)


# --- Funzione di strategia ---
def strategy_mom_trend_atr(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY long-only con filtro di trend (SMA lunga), momentum assoluto (ROC)
    e banda ATR asimmetrica (entry sopra SMA+K*ATR, exit sotto SMA-K*ATR).
    
    Regole:
      Entry se:  Close > SMA_long + K_entry*ATR  AND  ROC(lookback) > 0
      Exit  se:  Close < SMA_long - K_exit*ATR   OR   ROC(lookback) < 0

    Note:
    - Indicatori calcolati su 'Close'; esecuzione realistica su 'Open' nel tuo runner.
    - Segnali shiftati di 1 barra (no look-ahead).
    """

    sma_l   = params.get('sma_l_range')
    roc_lb  = params.get('roc_lb_range')
    atr_p   = params.get('atr_p_range')
    k_e10   = params.get('k_entry10_rng')
    k_x10   = params.get('k_exit10_rng')

    df = data.copy()

    # 1) --- Indicatori sull'intero df
    df['SMA_L']  = ind_mom_trend_atr_sma(df, period=sma_l)
    df['ROC']    = ind_mom_trend_atr_roc(df, lookback=roc_lb)
    df['ATR']    = ind_mom_trend_atr_atr(df, period=atr_p)

    k_entry = k_e10 / 10.0
    k_exit  = k_x10 / 10.0
    df['UP_BAND'] = df['SMA_L'] + k_entry * df['ATR']
    df['DN_BAND'] = df['SMA_L'] - k_exit  * df['ATR']

    # 2) --- Slicing per anno (dopo calcolo indicatori)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) --- Condizioni entry/exit sul df filtrato
    entries = (df['Close'] > df['UP_BAND']) & (df['ROC'] > 0)
    exits   = (df['Close'] < df['DN_BAND']) | (df['ROC'] < 0)

    # 4) --- Shift 1 barra per utilizzo su 'Open'
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy mom_trend_atr_vix
############################

# === SMA di trend (long) ===
def ind_mom_trend_atr_vix_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()

# === Momentum assoluto (ROC) ===
def ind_mom_trend_atr_vix_roc(df: pd.DataFrame, lookback: int = 126) -> pd.Series:
    c = df['Close']
    return (c / c.shift(lookback) - 1.0)

# === ATR classico (Wilder-like con EWM) ===
def ind_mom_trend_atr_vix_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = np.maximum(np.maximum((h - l), (h - pc).abs()), (l - pc).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()

# === Risk Regime via VIX Term Structure (ON se contango: VIX3M - VIX >= soglia) ===
def ind_mom_trend_atr_vix_riskflag(df: pd.DataFrame,
                                   vix_col: str = 'VIX',
                                   vix3m_col: str = 'VIX3M',
                                   ma_days: int = 3,
                                   thr10: int = 0,
                                   persist_days: int = 0) -> pd.Series:
    """
    Ritorna Serie booleana 'RISK_ON'.
    - usa df[vix_col], df[vix3m_col]; se assenti -> sempre True (nessun filtro).
    - ma_days: smoothing (SMA) della differenza VIX3M-VIX.
    - thr10: soglia in decimi di punto (2 -> 0.2 pt).
    - persist_days: richiede condizione vera per N giorni consecutivi (0 = nessuna persistenza).
    """
    if vix_col not in df.columns or vix3m_col not in df.columns:
        return pd.Series(True, index=df.index)

    spread = (df[vix3m_col] - df[vix_col]).rolling(ma_days, min_periods=1).mean()
    thr = thr10 / 10.0
    base_on = (spread >= thr)

    if persist_days <= 0:
        return base_on.astype(bool)

    win = persist_days + 1
    return (base_on.rolling(win, min_periods=1).min() == 1.0)

# --- Hook DI STRATEGIA: arricchisce il df con VIX e VIX3M (fallback VXV) ---
def ind_mom_trend_atr_vix_augment_df(df: pd.DataFrame, start: str, end: str, loader) -> pd.DataFrame:
    """
    Aggiunge colonne 'VIX' e 'VIX3M' allineate all'indice del df.
    Usa il loader della pipeline (load_ohlcv) per garantire coerenza d'indice.
    """
    try:
        vix   = loader("^VIX",   start, end)['Close'].rename("VIX")
        vix3m = loader("^VIX3M", start, end)['Close'].rename("VIX3M")
        if vix3m.dropna().empty:
            # fallback storico se ^VIX3M non disponibile
            vix3m = loader("^VXV", start, end)['Close'].rename("VIX3M")
    except Exception:
        return df  # fallback: strategia gestirà RISK_ON=True

    risk = pd.concat([vix, vix3m], axis=1).ffill()
    risk = risk.reindex(df.index).ffill()
    return df.join(risk, how='left')

# --- Griglia parametri per WF Optimization (≈ 6.8k combinazioni) ---
# strategy_mom_trend_atr_vix_param_ranges = {
#     'sma_l_range'     : range(170, 231, 10),  # 170..230
#     'roc_lb_range'    : range(120, 181, 30),  # 120,150,180
#     'atr_p_range'     : range(14, 21, 3),     # 14,17,20
#     'k_entry10_rng'   : range(10, 21, 5),     # 1.0,1.5,2.0
#     'k_exit10_rng'    : range(25, 36, 5),     # 2.5,3.0,3.5
#     'risk_ma_rng'     : range(3, 6, 2),       # 3,5
#     'risk_thr10_rng'  : range(0, 6, 2),       # 0.0,0.2,0.4
#     'risk_persist_rng': range(0, 4, 3)        # 0,3
# }

strategy_mom_trend_atr_vix_param_ranges = {
    'sma_l_range'     : range(170, 290, 60),  # 170..230
    'roc_lb_range'    : range(140, 160, 10),  # 120,150,180
    'atr_p_range'     : range(14, 26, 6),     # 14,17,20
    'k_entry10_rng'   : range(10, 25, 5),     # 1.0,1.5,2.0
    'k_exit10_rng'    : range(25, 31, 2),     # 2.5,3.0,3.5
    'risk_ma_rng'     : range(3, 7, 2),       # 3,5
    'risk_thr10_rng'  : range(0, 6, 2),       # 0.0,0.2,0.4
    'risk_persist_rng': range(0, 6, 3)        # 0,3
}

# --- Funzione di strategia ---
def strategy_mom_trend_atr_vix(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY long-only con filtro Risk ON/OFF derivato da VIX term structure (VIX3M - VIX).
    Regole:
      Entry se:  RISK_ON
                 AND Close > SMA_long + K_entry*ATR
                 AND ROC(lookback) > 0
      Exit  se:  (NOT RISK_ON)
                 OR Close < SMA_long - K_exit*ATR
                 OR ROC(lookback) < 0

    Indicatori calcolati su 'Close'; segnali shiftati di 1 barra (esecuzione realistica su 'Open').
    Se le colonne VIX/VIX3M mancano, il filtro risk resta ON (fallback interno).
    """

    sma_l   = params.get('sma_l_range')
    roc_lb  = params.get('roc_lb_range')
    atr_p   = params.get('atr_p_range')
    ke10    = params.get('k_entry10_rng')
    kx10    = params.get('k_exit10_rng')
    rma     = params.get('risk_ma_rng')
    rthr10  = params.get('risk_thr10_rng')
    rp      = params.get('risk_persist_rng')

    df = data.copy()

    # 1) --- Indicatori sull'intero df
    df['SMA_L'] = ind_mom_trend_atr_vix_sma(df, period=sma_l)
    df['ROC']   = ind_mom_trend_atr_vix_roc(df, lookback=roc_lb)
    df['ATR']   = ind_mom_trend_atr_vix_atr(df, period=atr_p)

    k_entry = ke10 / 10.0
    k_exit  = kx10 / 10.0
    df['UP_BAND'] = df['SMA_L'] + k_entry * df['ATR']
    df['DN_BAND'] = df['SMA_L'] - k_exit  * df['ATR']

    df['RISK_ON'] = ind_mom_trend_atr_vix_riskflag(
        df, vix_col='VIX', vix3m_col='VIX3M', ma_days=rma, thr10=rthr10, persist_days=rp
    ).astype(bool)

    # 2) --- Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) --- Segnali sul df filtrato
    entries = (df['RISK_ON']) & (df['Close'] > df['UP_BAND']) & (df['ROC'] > 0)
    exits   = (~df['RISK_ON']) | (df['Close'] < df['DN_BAND']) | (df['ROC'] < 0)

    # 4) --- Shift 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy spy_rs_breakout
############################

# === SMA lunga su SPY ===
def ind_spy_rs_breakout_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()

# === ATR classico ===
def ind_spy_rs_breakout_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = np.maximum(np.maximum((h - l), (h - pc).abs()), (l - pc).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()

# === Momentum assoluto (ROC) ===
def ind_spy_rs_breakout_roc(df: pd.DataFrame, lookback: int = 126) -> pd.Series:
    c = df['Close']
    return (c / c.shift(lookback) - 1.0)

# === Relative Strength SPY vs IEF ===
def ind_spy_rs_breakout_rs(df: pd.DataFrame, rs_ma: int = 100):
    """
    Richiede colonna 'IEF' (Close di IEF) nel df.
    Ritorna: (rs, rs_ma_line)
    """
    if 'IEF' not in df.columns:
        # fallback: nessun filtro RS (sempre True)
        rs = pd.Series(1.0, index=df.index)
        rs_line = rs
        return rs, rs_line
    rs = (df['Close'] / df['IEF']).replace([np.inf, -np.inf], np.nan)
    rs_line = rs.ewm(span=rs_ma, adjust=False, min_periods=1).mean()
    return rs, rs_line

# === Hook dati extra: aggiunge IEF (Close) ===
def ind_spy_rs_breakout_augment_df(df: pd.DataFrame, start: str, end: str, loader):
    """
    Carica IEF e lo aggiunge come colonna 'IEF' allineata all'indice del df.
    """
    try:
        ief = loader("IEF", start, end)['Close'].rename("IEF")
    except Exception:
        return df
    ief = ief.reindex(df.index).ffill()
    return df.join(ief, how='left')

# --- Griglia per WFO (≈ 16.5k combinazioni, compatta) ---
strategy_spy_rs_breakout_param_ranges = {
    'sma_l_range'     : range(150, 300, 50),  # 150..250 (11)
    'roc_lb_range'    : range(90, 270, 60),   # 90,120,150,180,210 (5)
    'atr_p_range'     : range(10, 25, 5),     # 10,15,20 (3)
    'k_exit10_rng'    : range(20, 42, 7),     # 2.0..3.5 (4)
    'rs_ma_rng'       : range(50, 200, 50),   # 50,75,100,125,150 (5)
    'hh_lb_range'     : range(20, 67, 23),    # 20,30,40,50,60 (5)
}
# Totale: 11*5*3*4*5*5 = 16,500

# --- Strategia ---
def strategy_spy_rs_breakout(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry se:
      - Close > SMA_L
      - ROC(lookback) > 0
      - (RS > RS_MA)  OR  (Close > HH_N_prev)  [breakout override]
    Exit se:
      - Close < SMA_L - k_exit*ATR  OR  RS < RS_MA  OR  ROC < 0

    Segnali shiftati di 1 barra (esecuzione su 'Open' nel runner).
    """
    sma_l  = params.get('sma_l_range')
    roc_lb = params.get('roc_lb_range')
    atr_p  = params.get('atr_p_range')
    kx10   = params.get('k_exit10_rng')
    rs_ma  = params.get('rs_ma_rng')
    hh_lb  = params.get('hh_lb_range')

    df = data.copy()

    # 1) Indicatori su tutto il df
    df['SMA_L']      = ind_spy_rs_breakout_sma(df, period=sma_l)
    df['ATR']        = ind_spy_rs_breakout_atr(df, period=atr_p)
    df['ROC']        = ind_spy_rs_breakout_roc(df, lookback=roc_lb)
    df['RS'], df['RS_MA'] = ind_spy_rs_breakout_rs(df, rs_ma=rs_ma)

    # Breakout: massimo rolling dei PRECEDENTI N giorni (shift(1) per evitare look-ahead)
    df['HH_PREV'] = df['Close'].rolling(hh_lb, min_periods=1).max().shift(1)

    k_exit = kx10 / 10.0
    df['DN_BAND'] = df['SMA_L'] - k_exit * df['ATR']

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Segnali
    cond_trend   = df['Close'] > df['SMA_L']
    cond_mom_abs = df['ROC'] > 0
    cond_rs_ok   = df['RS'] > df['RS_MA']
    cond_bo      = df['Close'] > df['HH_PREV']  # breakout override

    entries = cond_trend & cond_mom_abs & (cond_rs_ok | cond_bo)
    exits   = (df['Close'] < df['DN_BAND']) | (~cond_rs_ok) | (~cond_mom_abs)

    # 4) Shift 1 barra e normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy dualmom_breakout_spy
############################

# === SMA lunga ===
def ind_dualmom_breakout_spy_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()

# === ROC (momentum assoluto) ===
def ind_dualmom_breakout_spy_roc(series: pd.Series, lookback: int = 126) -> pd.Series:
    return series / series.shift(lookback) - 1.0

# === Hook: aggiunge TLT (Close) ===
def ind_dualmom_breakout_spy_augment_df(df: pd.DataFrame, start: str, end: str, loader):
    """Aggiunge 'TLT' (Close) allineato all'indice; se fallisce, ritorna df com’è."""
    try:
        tlt = loader("TLT", start, end)['Close'].rename("TLT")
    except Exception:
        return df
    return df.join(tlt.reindex(df.index).ffill(), how='left')

# --- Griglia WFO (≈ 1.1k combinazioni, compatta) ---
strategy_dualmom_breakout_spy_param_ranges = {
    'sma_l_range'    : range(150, 283, 33),  # 150,175,200,225,250 (5)
    'roc_lb_range'   : range(120, 280, 40),  # 120,150,180,210,240 (5)
    'thr_abs10_rng'  : range(0, 15, 5),      # soglia assoluta 0.0,0.5,1.0% (x/10/100)
    'thr_rel10_rng'  : range(0, 26, 6),      # extra vs TLT 0.0..2.0% step 0.5
    'hh_lb_range'    : range(20, 80, 20),    # 20,40,60 (3)
}
# Totale: 5*5*3*5*3 = 1.125

# --- Strategia ---
def strategy_dualmom_breakout_spy(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry se:
      Close > SMA_L
      e [(ROC_SPY > thr_abs) & (ROC_SPY > ROC_TLT + thr_rel)  OR  Close > HH_prev]
    Exit se:
      Close < SMA_L  OR  ROC_SPY < thr_abs  OR  ROC_SPY <= ROC_TLT + thr_rel

    Segnali shiftati di 1 barra (esecuzione su 'Open' nel runner).
    """
    sma_l   = params.get('sma_l_range')
    lb      = params.get('roc_lb_range')
    thr_a10 = params.get('thr_abs10_rng')
    thr_r10 = params.get('thr_rel10_rng')
    hh_lb   = params.get('hh_lb_range')

    df = data.copy()

    # 1) Indicatori su tutto df
    df['SMA_L']   = ind_dualmom_breakout_spy_sma(df, period=sma_l)
    roc_spy       = ind_dualmom_breakout_spy_roc(df['Close'], lookback=lb)
    # Se 'TLT' assente, setta ROC_TLT a -inf per non bloccare l'entry relativo
    roc_tlt       = ind_dualmom_breakout_spy_roc(df['TLT'] if 'TLT' in df.columns else pd.Series(np.nan, index=df.index), lookback=lb)
    roc_tlt       = roc_tlt.fillna(-np.inf)
    df['ROC_SPY'] = roc_spy
    df['ROC_TLT'] = roc_tlt

    # Breakout sui PRECEDENTI N giorni (shift per evitare look-ahead)
    df['HH_PREV'] = df['Close'].rolling(hh_lb, min_periods=1).max().shift(1)

    thr_abs = thr_a10 / 1000.0  # es. 10 -> 1.0%
    thr_rel = thr_r10 / 1000.0  # es. 20 -> 2.0%

    # 2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Segnali
    cond_trend   = df['Close'] > df['SMA_L']
    cond_abs     = df['ROC_SPY'] > thr_abs
    cond_rel     = df['ROC_SPY'] > (df['ROC_TLT'] + thr_rel)
    cond_bo      = df['Close'] > df['HH_PREV']

    entries = cond_trend & ((cond_abs & cond_rel) | cond_bo)
    exits   = (~cond_trend) | (~cond_abs) | (~cond_rel)

    # 4) Shift 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_mr_crsi_bbands
############################

# === RSI generico (EMA-style) ===
def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)

# === Connors RSI (RSI(3) + RSI(2) dello "streak" + PercentRank ROC(1, lookback)) ===
def ind_spy_mr_crsi_bbands_crsi(df: pd.DataFrame,
                                rsi_len: int = 3,
                                streak_rsi_len: int = 2,
                                prank_lb: int = 100) -> pd.Series:
    c = df['Close']
    rsi_fast = _rsi(c, rsi_len)

    # streak: lunghezza serie di chiusure consecutive up/down
    chg = c.diff()
    sign = np.sign(chg).fillna(0)
    streak = pd.Series(0.0, index=c.index)
    for i in range(1, len(c)):
        if sign.iloc[i] > 0:
            streak.iloc[i] = max(0, streak.iloc[i-1]) + 1
        elif sign.iloc[i] < 0:
            streak.iloc[i] = min(0, streak.iloc[i-1]) - 1
        else:
            streak.iloc[i] = 0
    rsi_streak = _rsi(streak, streak_rsi_len)

    roc1 = c.pct_change(1)
    # percent rank dell’ultimo ROC1 negli ultimi prank_lb giorni (0..100)
    def _prank(s: pd.Series, lb: int) -> pd.Series:
        roll = s.rolling(lb, min_periods=1)
        return roll.apply(lambda x: (x.rank(pct=True).iloc[-1] * 100.0), raw=False)
    prank = _prank(roc1, prank_lb)

    crsi = (rsi_fast + rsi_streak + prank) / 3.0
    return crsi

# === Bande di Bollinger ===
def ind_spy_mr_crsi_bbands_bands(df: pd.DataFrame, period: int = 20, k: float = 2.0):
    c = df['Close']
    ma = c.rolling(period, min_periods=1).mean()
    sd = c.rolling(period, min_periods=1).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower

# === SMA lunga (trend filter) ===
def ind_spy_mr_crsi_bbands_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    return df['Close'].rolling(period, min_periods=1).mean()

# --- Griglia WFO (3.8k combinazioni) ---
strategy_spy_mr_crsi_bbands_param_ranges = {
    'use_trend_rng'  : range(0, 2),        # 0=off, 1=on
    'sma_l_rng'      : range(150, 300, 50),# 150..250
    'bb_p_rng'       : range(15, 37, 7),   # 15,20,25,30
    'bb_k10_rng'     : range(15, 37, 7),   # 1.5..3.0
    'crsi_thr_rng'   : range(5, 25, 10),    # entry se CRSI <= 5..25
    'rsi2_exit_rng'  : range(65, 89, 12),   # exit se RSI2 >= 65..90
}

# --- Strategia ---
def strategy_spy_mr_crsi_bbands(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry:  (Close < LowerBand) & (CRSI <= thr) & (Trend OK se attivo)
    Exit:   (RSI2 >= thr_exit) | (Close >= MidBand)
    Tutto shiftato di 1 barra (usa 'Open' nel runner).
    """
    use_trend = params.get('use_trend_rng')
    sma_l     = params.get('sma_l_rng')
    bb_p      = params.get('bb_p_rng')
    bb_k      = params.get('bb_k10_rng') / 10.0
    thr_crsi  = params.get('crsi_thr_rng')
    thr_rsi2  = params.get('rsi2_exit_rng')

    df = data.copy()

    # Indicatori
    df['CRSI'] = ind_spy_mr_crsi_bbands_crsi(df, rsi_len=3, streak_rsi_len=2, prank_lb=100)
    df['RSI2'] = _rsi(df['Close'], 2)
    df['MID'], df['UP'], df['DN'] = ind_spy_mr_crsi_bbands_bands(df, period=bb_p, k=bb_k)
    df['SMA_L'] = ind_spy_mr_crsi_bbands_sma(df, period=sma_l)

    # Slice per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    trend_ok = (df['Close'] > df['SMA_L']) if (use_trend == 1) else pd.Series(True, index=df.index)

    entries = trend_ok & (df['Close'] < df['DN']) & (df['CRSI'] <= float(thr_crsi))
    exits   = (df['RSI2'] >= float(thr_rsi2)) | (df['Close'] >= df['MID'])

    # Shift 1 barra
    ent = entries.shift(1).astype(bool).fillna(False)
    ex  = exits.shift(1).astype(bool).fillna(False)
    return ent, ex
    
############################
# Strategy supertrend_adaptive_vol
############################

def ind_supertrend_adaptive_vol_st(df: pd.DataFrame, atr_period: int = 10, factor: float = 3.0):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = np.maximum(np.maximum(high - low, (high - close.shift()).abs()), (low  - close.shift()).abs())
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=1).mean()
    mid = (high + low) / 2.0
    upper = mid + factor * atr
    lower = mid - factor * atr

    st = pd.Series(index=df.index, dtype=float)
    d  = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]; d.iloc[i] = 1; continue
        prev = st.iloc[i-1]
        c = close.iloc[i]
        d.iloc[i] = 1 if c > prev else (-1 if c < prev else d.iloc[i-1])
        st.iloc[i] = max(lower.iloc[i], prev) if d.iloc[i] == 1 else min(upper.iloc[i], prev)
    return st, d, atr

# === Percentile rolling dell’ATR (per filtro di regime di volatilità) ===
def ind_supertrend_adaptive_vol_atr_pctile(atr: pd.Series, win: int = 120) -> pd.Series:
    # Percentile “ex-ante”: usa rank all’interno della finestra (0..1)
    r = atr.rolling(win, min_periods=10)
    pct = (r.rank(pct=True)).iloc[:, 0] if isinstance(r.rank(pct=True), pd.DataFrame) else r.rank(pct=True)
    return pct.fillna(0.5)

# --- Griglia WFO (contenuta ~756 comb) ---
strategy_supertrend_adaptive_vol_param_ranges = {
    'atr_period_range'   : range(7, 21, 2),      # 7..19
    'factor_range'       : range(2, 5, 1),       # 2,3,4
    'atr_win_range'      : [60, 120, 180],
    'pct_low_range'      : [40, 50],             # percentile inferior (40% / 50%)
    'pct_high_range'     : [70, 80],             # percentile superior (70% / 80%)
    'confirm_bars_range' : [0, 1, 2]             # barre di conferma verde
}

def strategy_supertrend_adaptive_vol(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    LONG quando:
      - Supertrend daily verde
      - (opz.) confermato da 'confirm_bars' barre consecutive
      - ATR_percentile ∈ [pct_low, pct_high]  → evita fasi troppo calme (falsi segnali)
        e troppo turbolente (whipsaw/stop rumorosi)
    EXIT:
      - flip Supertrend a rosso
    """
    atr_p   = params.get('atr_period_range')
    fact    = params.get('factor_range')
    awin    = params.get('atr_win_range')
    p_low   = params.get('pct_low_range') / 100 if isinstance(params.get('pct_low_range'), (int,float)) else params.get('pct_low_range')[0] / 100
    p_high  = params.get('pct_high_range')/ 100 if isinstance(params.get('pct_high_range'),(int,float)) else params.get('pct_high_range')[0] / 100
    conf    = params.get('confirm_bars_range')

    df = data.copy()
    df['ST_Line'], df['ST_Dir'], df['ATR'] = ind_supertrend_adaptive_vol_st(df, atr_period=atr_p, factor=fact)
    df['ATR_pct'] = ind_supertrend_adaptive_vol_atr_pctile(df['ATR'], win=int(awin))

    # conferma barre verdi consecutive (se conf=0, sempre True)
    confirm = (df['ST_Dir'].rolling(int(conf)).min() == 1) if int(conf) > 0 else pd.Series(True, index=df.index)

    if year is not None:
        df = df[df.index.year == int(year)]
        confirm = confirm.loc[df.index]

    entries = (df['ST_Dir'] == 1) & confirm & (df['ATR_pct'] >= float(p_low)) & (df['ATR_pct'] <= float(p_high))
    exits   = (df['ST_Dir'] == -1)

    return entries.shift(1).astype(bool).fillna(False), exits.shift(1).astype(bool).fillna(False)

    
####################################
# Strategy supertrend_breakout_trail
####################################

def ind_supertrend_breakout_trail_st(df: pd.DataFrame, atr_period: int = 10, factor: float = 3.0):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = np.maximum(np.maximum(high - low, (high - close.shift()).abs()), (low  - close.shift()).abs())
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=1).mean()
    mid = (high + low) / 2.0
    upper = mid + factor * atr
    lower = mid - factor * atr
    st = pd.Series(index=df.index, dtype=float)
    d  = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]; d.iloc[i] = 1; continue
        prev = st.iloc[i-1]; c = close.iloc[i]
        d.iloc[i] = 1 if c > prev else (-1 if c < prev else d.iloc[i-1])
        st.iloc[i] = max(lower.iloc[i], prev) if d.iloc[i] == 1 else min(upper.iloc[i], prev)
    return st, d

# === Donchian breakout (highest close N) ===
def ind_supertrend_breakout_trail_donchian_high(close: pd.Series, n: int = 40) -> pd.Series:
    return close.rolling(n, min_periods=1).max()

# --- Griglia WFO (~756 comb) ---
strategy_supertrend_breakout_trail_param_ranges = {
    'atr_period_range'   : range(7, 21, 2),
    'factor_range'       : range(2, 5, 1),
    'breakout_n_range'   : [20, 40, 60],     # conferma momentum post-flip
    'wait_bars_range'    : [0, 5],           # attesa dopo il flip prima di cercare breakout
    'cooldown_range'     : [0, 10],          # blocco ri-entry dopo exit
    'max_bars_range'     : [30, 60, 90]      # time-stop di sicurezza
}

def strategy_supertrend_breakout_trail(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    LONG solo quando il flip a verde è seguito da un vero breakout:
      Entry:
        - ST verde
        - Close >= max( Close, last N )  (Donchian breakout)
        - dopo 'wait_bars' dal flip (evita prendere la candela di flip)
        - non in cooldown dopo una exit recente
      Exit:
        - flip ST a rosso  OR  time-stop > max_bars
    """
    atr_p   = params.get('atr_period_range')
    fact    = params.get('factor_range')
    n       = params.get('breakout_n_range')
    wait    = int(params.get('wait_bars_range'))
    cooldown= int(params.get('cooldown_range'))
    maxbars = int(params.get('max_bars_range'))

    df = data.copy()
    df['ST_Line'], df['ST_Dir'] = ind_supertrend_breakout_trail_st(df, atr_period=atr_p, factor=fact)
    df['DonHi'] = ind_supertrend_breakout_trail_donchian_high(df['Close'], n=int(n))

    # flip a verde/rosso (punti di cambio)
    flip_to_green = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
    flip_to_red   = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)

    # attesa dopo flip a verde
    bars_since_green = (~flip_to_green).cumsum() - (~flip_to_green).cumsum().where(flip_to_green).ffill().fillna(0).astype(int)
    ok_wait = bars_since_green >= wait

    # cooldown dopo exit
    last_exit_idx = flip_to_red.cumsum().where(flip_to_red).ffill()
    bars_since_exit = (flip_to_red.cumsum() - last_exit_idx).fillna(1e9).astype(int)  # grande se mai usciti
    ok_cooldown = bars_since_exit >= cooldown

    # 2) filtro anno
    if year is not None:
        df = df[df.index.year == int(year)]
        flip_to_green = flip_to_green.loc[df.index]
        flip_to_red   = flip_to_red.loc[df.index]
        bars_since_green = bars_since_green.loc[df.index]
        ok_wait = ok_wait.loc[df.index]
        ok_cooldown = ok_cooldown.loc[df.index]

    entries_base = (df['ST_Dir'] == 1) & ok_wait & ok_cooldown & (df['Close'] >= df['DonHi'])
    # time-stop
    trade_id = entries_base.cumsum()
    bars_in_trade = (trade_id - trade_id.where(entries_base).ffill()).fillna(0).astype(int)
    time_stop = (bars_in_trade >= maxbars)

    exits = (df['ST_Dir'] == -1) | time_stop

    return entries_base.shift(1).astype(bool).fillna(False), exits.shift(1).astype(bool).fillna(False)

############################
# Strategy natgas_donchian_atr
############################

# === Donchian Channel ===
def ind_natgas_donchian_atr_donchian(df: pd.DataFrame, period: int = 20):
    """
    Canale di Donchian su High/Low.
    Ritorna: (upper, lower, middle)
    """
    high, low = df['High'], df['Low']
    upper  = high.rolling(period).max()
    lower  = low.rolling(period).min()
    middle = (upper + lower) / 2.0
    return upper, lower, middle

# === ATR (Wilder-style, EWM) ===
def ind_natgas_donchian_atr_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df['High'], df['Low'], df['Close']
    tr = np.maximum(np.maximum((h-l), (h-c.shift()).abs()), (l-c.shift()).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()

# --- Griglia parametri per WFO (contenuta) ---
strategy_natgas_donchian_atr_param_ranges = {
    'don_p'           : range(10, 41, 5),   # 10,15,20,25,30,35,40
    'atr_p'           : range(10, 21, 5),   # 10,15,20
    'atr_mult_x10'    : [0, 5, 10],         # 0, 0.5, 1.0
    'trail_mult_x10'  : [15, 25, 35],       # 1.5, 2.5, 3.5
    'exit_mode'       : [0, 1],             # 0=Donchian middle; 1=trailing ATR
    'enable_short'    : [0, 1]              # 0=solo long; 1=long+short
}
# Totale combinazioni: 7*3*3*3*2*2 = 756

# --- Strategia ---
def strategy_natgas_donchian_atr(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole:
      • Entry long  : Close > DON_U_prev + ATR*atr_mult
      • Entry short : (se abilitato) Close < DON_L_prev - ATR*atr_mult
      • Exit long   : se exit_mode=0 -> Close < DON_M
                      se exit_mode=1 -> Close < (DON_U - ATR*trail_mult)
      • Exit short  : se exit_mode=0 -> Close > DON_M
                      se exit_mode=1 -> Close > (DON_L + ATR*trail_mult)

    Note K_Strategy:
      1) indicatori su df completo; 2) slicing per anno; 3) entries/exits; 4) shift 1 barra.
    """
    don_p          = params.get('don_p')
    atr_p          = params.get('atr_p')
    atr_mult_x10   = params.get('atr_mult_x10')
    trail_mult_x10 = params.get('trail_mult_x10')
    exit_mode      = params.get('exit_mode')
    enable_short   = params.get('enable_short')

    atr_mult   = (atr_mult_x10 or 0) / 10.0
    trail_mult = (trail_mult_x10 or 15) / 10.0  # default 1.5

    df = data.copy()

    # (1) Indicatori su tutto il df
    df['DON_U'], df['DON_L'], df['DON_M'] = ind_natgas_donchian_atr_donchian(df, period=don_p)
    df['ATR'] = ind_natgas_donchian_atr_atr(df, period=atr_p)

    # (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Canali PRECEDENTI per i trigger breakout (no look-ahead) ---
    don_u_prev = df['DON_U'].shift(1)
    don_l_prev = df['DON_L'].shift(1)

    # (3) Segnali
    # Entry
    long_entries  = df['Close'] > (don_u_prev + df['ATR'] * atr_mult)
    if bool(enable_short):
        short_entries = df['Close'] < (don_l_prev - df['ATR'] * atr_mult)
    else:
        short_entries = pd.Series(False, index=df.index)

    # Exit
    if int(exit_mode) == 0:
        long_exits  = df['Close'] < df['DON_M']
        short_exits = df['Close'] > df['DON_M']
    else:
        long_guard  = df['DON_U'] - df['ATR'] * trail_mult
        short_guard = df['DON_L'] + df['ATR'] * trail_mult
        long_exits  = df['Close'] < long_guard
        short_exits = df['Close'] > short_guard

    entries = (long_entries | short_entries).shift(1).astype(bool).fillna(False)
    exits   = (long_exits  | short_exits ).shift(1).astype(bool).fillna(False)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy natgas_bb_mr
############################

# === Bollinger Bands ===
def ind_natgas_bb_mr_bbands(df: pd.DataFrame, period: int = 20, dev: float = 2.0):
    close = df['Close']
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + dev * std
    lower = ma - dev * std
    return ma, upper, lower

# === ATR (Wilder EWM) ===
def ind_natgas_bb_mr_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df['High'], df['Low'], df['Close']
    tr = np.maximum(np.maximum((h-l), (h-c.shift()).abs()), (l-c.shift()).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()

# --- Griglia parametri (contenuta) ---
strategy_natgas_bb_mr_param_ranges = {
    'bb_period'      : range(10, 31, 5),   # 10,15,20,25,30
    'bb_dev'         : [2.0, 3.0],         # deviazioni standard
    'atr_p'          : range(10, 21, 5),   # 10,15,20
    'buf_x10'        : [0, 5, 10],         # buffer su entry: 0,0.5,1.0 * ATR
    'stop_x10'       : [0, 15, 25],        # stop addizionale: 0,1.5,2.5 * ATR
    'exit_mode'      : [0, 1],             # 0=esci su BB_M; 1=esci su BB_U (take profit più alto)
    'time_stop'      : [5, 10, 15],        # giorni max in posizione (time-based exit)
    'enable_short'   : [0, 1]              # 0=solo long; 1=long+short
}
# Combinazioni: 5*2*3*3*3*2*3*2 = 1620

# --- Strategia ---
def strategy_natgas_bb_mr(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Mean-reversion con BB + buffer ATR, time-stop e short opzionale.
      • Entry long  : Close < BB_L - ATR*buf
      • Exit long   : (exit_mode=0 -> Close >= BB_M) OR (exit_mode=1 -> Close >= BB_U) OR (Close <= BB_L - ATR*stop) OR (age >= time_stop)
      • Entry short : (se abilitato) Close > BB_U + ATR*buf
      • Exit short  : simmetrico
    Note K_Strategy: 1) indicatori; 2) slicing anno; 3) segnali; 4) shift 1 barra.
    """
    bb_p       = params.get('bb_period')
    bb_dev     = float(params.get('bb_dev'))
    atr_p      = params.get('atr_p')
    buf_x10    = params.get('buf_x10')
    stop_x10   = params.get('stop_x10')
    exit_mode  = params.get('exit_mode')
    time_stop  = params.get('time_stop')
    en_short   = params.get('enable_short')

    buf  = (buf_x10 or 0)  / 10.0
    add_stop = (stop_x10 or 0) / 10.0

    df = data.copy()

    # (1) Indicatori su tutto il df
    df['BB_M'], df['BB_U'], df['BB_L'] = ind_natgas_bb_mr_bbands(df, period=bb_p, dev=bb_dev)
    df['ATR'] = ind_natgas_bb_mr_atr(df, period=atr_p)

    # (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # (3) Segnali base (senza time-stop, che è stato-dipendente)
    long_entries  = df['Close'] < (df['BB_L'] - df['ATR'] * buf)
    long_exit_tp  = (df['Close'] >= df['BB_M']) if int(exit_mode) == 0 else (df['Close'] >= df['BB_U'])
    long_exit_sl  = (add_stop > 0) & (df['Close'] <= (df['BB_L'] - df['ATR'] * add_stop))

    if bool(en_short):
        short_entries = df['Close'] > (df['BB_U'] + df['ATR'] * buf)
        short_exit_tp = (df['Close'] <= df['BB_M']) if int(exit_mode) == 0 else (df['Close'] <= df['BB_L'])
        short_exit_sl = (add_stop > 0) & (df['Close'] >= (df['BB_U'] + df['ATR'] * add_stop))
    else:
        short_entries = pd.Series(False, index=df.index)
        short_exit_tp = pd.Series(False, index=df.index)
        short_exit_sl = pd.Series(False, index=df.index)

    # Time-stop (approssimazione senza stato: usiamo contatore rolling dei giorni “in posizione potenziale”)
    # Nota: per coerenza semplice con K_Strategy (no stato), attiviamo time-exit quando il
    # numero di giorni consecutivi “oltre la soglia entry” supera time_stop.
    # È una proxy che in pratica chiude le mean-revert troppo lente.
    long_age = (df['Close'] < df['BB_L']).astype(int)
    long_age = long_age.groupby((long_age != long_age.shift()).cumsum()).cumsum()
    long_exit_time = long_age >= int(time_stop)

    short_age = (df['Close'] > df['BB_U']).astype(int)
    short_age = short_age.groupby((short_age != short_age.shift()).cumsum()).cumsum()
    short_exit_time = short_age >= int(time_stop)

    # Uscite complessive
    long_exits  = long_exit_tp | long_exit_sl | long_exit_time
    short_exits = short_exit_tp | short_exit_sl | short_exit_time

    # (4) Shift 1 barra e normalizzazione
    entries = (long_entries | short_entries).shift(1).astype(bool).fillna(False)
    exits   = (long_exits  | short_exits ).shift(1).astype(bool).fillna(False)
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy qqq_ema_crossover
############################

# === EMA crossover ===
def ind_qqq_ema_crossover_ema(df: pd.DataFrame, fast: int, slow: int):
    """
    Calcola due EMA (fast e slow).
    """
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    return ema_fast, ema_slow

# --- Griglia parametri ---
strategy_qqq_ema_crossover_param_ranges = {
    'fast' : range(10, 31, 5),     # 10,15,20,25,30
    'slow' : range(50, 201, 25)    # 50,75,100,125,150,175,200
}
# Totale combinazioni = 5 * 7 = 35

# --- Strategia ---
def strategy_qqq_ema_crossover(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry long se EMA_fast > EMA_slow.
    Exit long se EMA_fast < EMA_slow.
    """
    fast = params.get('fast')
    slow = params.get('slow')

    df = data.copy()

    # (1) Indicatori su tutto il df
    df['EMA_F'], df['EMA_S'] = ind_qqq_ema_crossover_ema(df, fast, slow)

    # (2) Slicing annuale
    if year is not None:
        df = df[df.index.year == int(year)]

    # (3) Condizioni entry/exit
    entries = df['EMA_F'] > df['EMA_S']
    exits   = df['EMA_F'] < df['EMA_S']

    # (4) Shift e normalizzazione
    entries = entries.shift(1).astype(bool).fillna(False)
    exits   = exits.shift(1).astype(bool).fillna(False)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy cfo_osma_v2
############################

# === CFO corretto (Chande Forecast Oscillator) ===
def ind_cfo_osma_cfo(df: pd.DataFrame, period: int = 14, forecast_step: int = 1) -> pd.Series:
    """
    CFO = 100 * (Close - Forecast) / Close
    dove 'Forecast' è la previsione (step-ahead) di regressione lineare sui Close
    nell'ultima finestra 'period'.

    Nota: usiamo rolling + np.polyfit per robustezza e chiarezza.
    """
    close = df["Close"]

    def _forecast(y: np.ndarray) -> float:
        x = np.arange(len(y))  # 0..period-1
        slope, intercept = np.polyfit(x, y, 1)
        # forecast 'forecast_step' passi avanti rispetto all'ultimo punto (x = period-1)
        x_future = (len(y) - 1) + forecast_step
        return intercept + slope * x_future

    # previsione rolling
    fc = close.rolling(window=period, min_periods=period).apply(_forecast, raw=True)

    cfo = 100.0 * (close - fc) / close
    return cfo


# === OSMA (Moving Average of Oscillator) normalizzato ===
def ind_cfo_osma_osma_pct(df: pd.DataFrame,
                          fast_period: int = 12,
                          slow_period: int = 26,
                          signal_period: int = 9) -> pd.Series:
    """
    OSMA % = 100 * (MACD - Signal) / Close
    -> scala-invariante (utile per azioni con prezzi molto diversi).
    """
    close = df["Close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    osma = macd - macd_signal
    osma_pct = 100.0 * osma / close
    return osma_pct


# --- Griglia parametri per WF Optimization (LITE, 48 combo/finestra) ---
strategy_cfo_osma_v2_param_ranges = {
    # CFO
    'cfo_period'     : [12, 16, 20],   # 3
    'forecast_step'  : [1, 2],         # 2
    'cfo_thr'        : [0.5, 1.0],     # 2  (percento)

    # OSMA% (MACD fisso: 12,26,9 per contenere le combinazioni)
    'fast_period'    : [12],           # 1
    'slow_period'    : [26],           # 1
    'signal_period'  : [9],            # 1
    'osma_thr'       : [0.05, 0.10],   # 2  (percento)

    # timing/cross
    'lookback_cross' : [1, 2]          # 2
}


# --- Funzione di strategia ---
def strategy_cfo_osma_v2(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Segnali a crossover con soglie:

      Definizioni:
        CFO%  = cfo
        OSMA% = osma_pct

      Entry long quando (cross up recente):
        cfo  > +cfo_thr   e   cfo.shift(lookback_cross) <= +cfo_thr
        e
        osma > +osma_thr  e   osma.shift(lookback_cross) <= +osma_thr

      Exit long quando (uno dei due torna debole):
        cfo  < -cfo_thr   o   osma < -osma_thr

    Shift finale di 1 barra per evitare look-ahead.
    """

    pd.set_option('future.no_silent_downcasting', True)

    df = data.copy()

    # --- Indicatori sul dataset completo ---
    cfo_period    = params.get('cfo_period')
    forecast_step = params.get('forecast_step')
    cfo_thr       = float(params.get('cfo_thr'))

    fast_p   = params.get('fast_period')
    slow_p   = params.get('slow_period')
    sig_p    = params.get('signal_period')
    osma_thr = float(params.get('osma_thr'))

    lookback_cross = int(params.get('lookback_cross'))

    df['CFO'] = ind_cfo_osma_cfo(df, period=cfo_period, forecast_step=forecast_step)
    df['OSMA_PCT'] = ind_cfo_osma_osma_pct(df, fast_period=fast_p, slow_period=slow_p, signal_period=sig_p)

    # --- (opzionale) filtro per anno ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Crossover “recente” su soglia positiva ---
    cfo_up_now   = df['CFO'] >  cfo_thr
    cfo_up_then  = df['CFO'].shift(lookback_cross).fillna(-np.inf).infer_objects(copy=False) <= cfo_thr
    osma_up_now  = df['OSMA_PCT'] >  osma_thr
    osma_up_then = df['OSMA_PCT'].shift(lookback_cross).fillna(-np.inf).infer_objects(copy=False) <= osma_thr

    entries = (cfo_up_now & cfo_up_then) & (osma_up_now & osma_up_then)

    # --- Uscita quando uno dei due torna sotto la soglia negativa (per evitare whipsaw simmetricamente) ---
    exits = (df['CFO'] < -cfo_thr) | (df['OSMA_PCT'] < -osma_thr)

    # --- Shift 1 barra ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy cfo_osma_v3
############################


# === CFO (Chande Forecast Oscillator) corretto ===
def ind_cfo_osma_v3_cfo(df: pd.DataFrame, period: int = 14, forecast_step: int = 1) -> pd.Series:
    """
    CFO% = 100 * (Close - Forecast) / Close
    dove 'Forecast' è la previsione (step-ahead) di regressione lineare
    sugli ultimi 'period' Close.

    Parametri:
        period        : lunghezza finestra della regressione
        forecast_step : passi avanti rispetto all’ultima barra (>=1)
    """
    close = df["Close"]

    def _forecast(y: np.ndarray) -> float:
        x = np.arange(len(y))  # 0..period-1
        slope, intercept = np.polyfit(x, y, 1)
        x_future = (len(y) - 1) + forecast_step
        return intercept + slope * x_future

    fc = close.rolling(window=period, min_periods=period).apply(_forecast, raw=True)
    cfo = 100.0 * (close - fc) / close
    return cfo


# === OSMA% (Moving Average of Oscillator normalizzato) ===
def ind_cfo_osma_v3_osma_pct(df: pd.DataFrame,
                             fast_period: int = 12,
                             slow_period: int = 26,
                             signal_period: int = 9) -> pd.Series:
    """
    OSMA% = 100 * (MACD - Signal) / Close
    Normalizzazione in percentuale del prezzo per soglie stabili tra asset diversi.
    """
    close = df["Close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    osma = macd - macd_signal
    osma_pct = 100.0 * osma / close
    return osma_pct


# --- Griglia parametri per WF Optimization (36 combo/finestra) ---
strategy_cfo_osma_v3_param_ranges = {
    'cfo_period'     : [10, 14, 18],   # 3
    'forecast_step'  : [1, 2],         # 2
    'cfo_thr'        : [0.3, 0.7],     # 2  (percentuale)
    'fast_period'    : [12],           # 1
    'slow_period'    : [26],           # 1
    'signal_period'  : [9],            # 1
    'osma_thr'       : [0.03, 0.06, 0.10],  # 3 (percentuale)
}
# 3*2*2*1*1*1*3 = 36 combo/finestra


# --- Funzione di strategia ---
def strategy_cfo_osma_v3(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole (versione attenuata):
      • Entry long:  CFO > +cfo_thr  AND  OSMA% > -osma_thr
      • Exit long :  CFO < -cfo_thr  OR   OSMA% < -osma_thr

    Note:
      - Indicatori calcolati sull'intero dataset; l'eventuale filtro 'year'
        è applicato come maschera a valle per coerenza WFO.
      - Segnali shiftati di 1 barra per evitare look-ahead.
    """

    pd.set_option('future.no_silent_downcasting', True)

    df = data.copy()

    # --- Parametri ---
    cfo_period     = int(params.get('cfo_period'))
    forecast_step  = int(params.get('forecast_step'))
    cfo_thr        = float(params.get('cfo_thr'))

    fast_p         = int(params.get('fast_period'))
    slow_p         = int(params.get('slow_period'))
    sig_p          = int(params.get('signal_period'))
    osma_thr       = float(params.get('osma_thr'))

    # --- Indicatori ---
    df['CFO'] = ind_cfo_osma_v3_cfo(df, period=cfo_period, forecast_step=forecast_step)
    df['OSMA_PCT'] = ind_cfo_osma_v3_osma_pct(df, fast_period=fast_p, slow_period=slow_p, signal_period=sig_p)

    # --- Filtro anno (solo maschera) ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Segnali ---
    entries = (df['CFO'] >  cfo_thr) & (df['OSMA_PCT'] > -osma_thr)
    exits   = (df['CFO'] < -cfo_thr) | (df['OSMA_PCT'] < -osma_thr)

    # --- Shift 1 barra per uso in backtest ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)

############################
# Strategy cfo_osma_v4
############################

# === CFO (Chande Forecast Oscillator) ===
def ind_cfo_osma_v4_cfo(df: pd.DataFrame, period: int = 14, forecast_step: int = 1) -> pd.Series:
    """
    CFO% = 100 * (Close - Forecast) / Close
    con Forecast da regressione lineare step-ahead sulla finestra 'period'.
    """
    close = df["Close"]

    def _forecast(y: np.ndarray) -> float:
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        x_future = (len(y) - 1) + forecast_step
        return intercept + slope * x_future

    fc = close.rolling(window=period, min_periods=period).apply(_forecast, raw=True)
    return 100.0 * (close - fc) / close


# === OSMA% (MACD-Signal normalizzato al prezzo) ===
def ind_cfo_osma_v4_osma_pct(df: pd.DataFrame,
                             fast_period: int = 12,
                             slow_period: int = 26,
                             signal_period: int = 9) -> pd.Series:
    close = df["Close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    osma = macd - macd_signal
    return 100.0 * osma / close


# === EMA generica ===
def ind_cfo_osma_v4_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()


# === ATR (Wilder) ===
def ind_cfo_osma_v4_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low), (high - prev_close).abs()), (low - prev_close).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


# --- Griglia parametri per WF Optimization (≤64 combo/finestra) ---
strategy_cfo_osma_v4_param_ranges = {
    # CFO
    'cfo_period'     : [10, 14],     # 2
    'forecast_step'  : [1],          # 1
    'cfo_thr'        : [0.2, 0.5],   # 2 (percentuale)

    # OSMA%
    'fast_period'    : [12],         # 1
    'slow_period'    : [26],         # 1
    'signal_period'  : [9],          # 1
    'osma_thr'       : [0.03, 0.06], # 2 (percentuale)

    # Trend & risk
    'ema_fast'       : [20, 30],     # 2
    'ema_slow'       : [150, 200],   # 2
    'atr_period'     : [14],         # 1
    'atr_k'          : [0.5, 1.0],   # 2  (moltiplicatore ATR sul fast-EMA)

    # Slope filtro (quante barre per misurare pendenza)
    'slope_lookback' : [5],          # 1
}
# Conteggio: 2*1*2 * 1*1*1*2 * 2*2*1*2 * 1 = 64 combo/finestra


# --- Funzione di strategia ---
def strategy_cfo_osma_v4(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Logica (long-only, pro-trend):

      Filtro di mercato (attivo):
        • Close > EMA_slow
        • Slope positiva di EMA_slow: EMA_slow > EMA_slow.shift(slope_lookback)

      Entry:
        • CFO > +cfo_thr
        • OSMA% > -osma_thr
        • Close > EMA_fast
        • Filtro di mercato vero

      Exit (qualunque condizione vera):
        • Close < EMA_fast - atr_k * ATR
        • CFO < -cfo_thr
        • OSMA% < -osma_thr
        • Close < EMA_slow  (perdita trend di sfondo)

      NOTE:
        - Indicatori calcolati sull'intero dataset; l’eventuale 'year' è una maschera a valle.
        - Segnali shiftati di 1 barra per evitare look-ahead.
    """
        
    pd.set_option('future.no_silent_downcasting', True)

    df = data.copy()

    # --- Parametri ---
    cfo_p     = int(params.get('cfo_period'))
    fstep     = int(params.get('forecast_step'))
    cfo_thr   = float(params.get('cfo_thr'))

    fast_p    = int(params.get('fast_period'))
    slow_p    = int(params.get('slow_period'))
    sig_p     = int(params.get('signal_period'))
    osma_thr  = float(params.get('osma_thr'))

    ema_f     = int(params.get('ema_fast'))
    ema_s     = int(params.get('ema_slow'))
    atr_p     = int(params.get('atr_period'))
    atr_k     = float(params.get('atr_k'))
    slope_lb  = int(params.get('slope_lookback'))

    # --- Indicatori ---
    df['CFO']       = ind_cfo_osma_v4_cfo(df, period=cfo_p, forecast_step=fstep)
    df['OSMA_PCT']  = ind_cfo_osma_v4_osma_pct(df, fast_period=fast_p, slow_period=slow_p, signal_period=sig_p)
    df['EMA_FAST']  = ind_cfo_osma_v4_ema(df, period=ema_f)
    df['EMA_SLOW']  = ind_cfo_osma_v4_ema(df, period=ema_s)
    df['ATR']       = ind_cfo_osma_v4_atr(df, period=atr_p)

    # --- Filtro per anno (solo maschera) ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Filtro trend di sfondo ---
    trend_ok  = (df['Close'] > df['EMA_SLOW']) & (df['EMA_SLOW'] > df['EMA_SLOW'].shift(slope_lb))

    # --- Condizioni Entry ---
    entries = (
        (df['CFO'] >  cfo_thr) &
        (df['OSMA_PCT'] > -osma_thr) &
        (df['Close'] > df['EMA_FAST']) &
        trend_ok
    )

    # --- Condizioni Exit ---
    ema_fast_stop = df['EMA_FAST'] - atr_k * df['ATR']
    exits = (
        (df['Close'] < ema_fast_stop) |
        (df['CFO'] < -cfo_thr) |
        (df['OSMA_PCT'] < -osma_thr) |
        (df['Close'] < df['EMA_SLOW'])
    )

    # --- Shift 1 barra per uso in backtest ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy emaatr_pullback_v1
############################

# === EMA ===
def ind_emaatr_pullback_v1_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()


# === ATR (Wilder) ===
def ind_emaatr_pullback_v1_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low), (high - prev_close).abs()), (low - prev_close).abs())
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


# === RSI (2) classico ===
def ind_emaatr_pullback_v1_rsi(df: pd.DataFrame, period: int = 2) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Griglia parametri per WF Optimization (16 combo/finestra) ---
strategy_emaatr_pullback_v1_param_ranges = {
    'ema_fast'   : [20, 30],     # 2
    'ema_slow'   : [150, 200],   # 2
    'atr_period' : [14],         # 1
    'atr_k'      : [1.0, 1.5],   # 2
    'rsi2_thr'   : [5, 10],      # 2
}
# Totale: 2 * 2 * 1 * 2 * 2 = 16 combo/finestra


# --- Funzione di strategia ---
def strategy_emaatr_pullback_v1(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Logica long-only, pro-trend, semplice e aggressiva sui pullback:

      Regime (trend filter):
        • Close > EMA_slow
        • EMA_slow in salita: EMA_slow > EMA_slow.shift(5)

      Setup pullback:
        • (Close_y < EMA_fast - atr_k * ATR)  --> "profondo"
        • e RSI(2) < rsi2_thr                  --> ipervenduto breve

      Trigger:
        • Cross up di Close sopra EMA_fast

      Exit (qualunque condizione vera):
        • Close < EMA_fast                     (perdita momentum)
        • Close < EMA_fast - atr_k * ATR       (stop dinamico)
        • Close < EMA_slow                     (fine trend di sfondo)

      Segnali shiftati di 1 barra per evitare look-ahead.
      L'eventuale 'year' è applicato come maschera a valle.
    """
    pd.set_option('future.no_silent_downcasting', True)

    df = data.copy()

    # --- Parametri ---
    ema_f   = int(params.get('ema_fast'))
    ema_s   = int(params.get('ema_slow'))
    atr_p   = int(params.get('atr_period'))
    atr_k   = float(params.get('atr_k'))
    rsi_thr = float(params.get('rsi2_thr'))

    # --- Indicatori ---
    df['EMA_FAST'] = ind_emaatr_pullback_v1_ema(df, period=ema_f)
    df['EMA_SLOW'] = ind_emaatr_pullback_v1_ema(df, period=ema_s)
    df['ATR']      = ind_emaatr_pullback_v1_atr(df, period=atr_p)
    df['RSI2']     = ind_emaatr_pullback_v1_rsi(df, period=2)

    # --- Maschera per anno (solo a valle) ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Regime di mercato ---
    slope_lb = 5
    regime = (df['Close'] > df['EMA_SLOW']) & (df['EMA_SLOW'] > df['EMA_SLOW'].shift(slope_lb))

    # --- Setup pullback profondo (ieri) ---
    deep_pull_y = df['Close'].shift(1) < (df['EMA_FAST'].shift(1) - atr_k * df['ATR'].shift(1))
    rsi_ok_y    = df['RSI2'].shift(1) < rsi_thr

    # --- Trigger di ingresso: cross up su EMA_FAST (oggi) ---
    cross_up = (df['Close'].shift(1) <= df['EMA_FAST'].shift(1)) & (df['Close'] > df['EMA_FAST'])

    entries = regime & deep_pull_y & rsi_ok_y & cross_up

    # --- Uscite ---
    ema_stop   = df['EMA_FAST'] - atr_k * df['ATR']
    exits = (df['Close'] < df['EMA_FAST']) | (df['Close'] < ema_stop) | (df['Close'] < df['EMA_SLOW'])

    # --- Shift 1 barra per uso in backtest ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy donchian_atr_v1
############################

# === Donchian Channels ===
def ind_donchian_atr_v1_donchian(df: pd.DataFrame, lookback: int = 55):
    """
    Ritorna (upper, lower, mid):
      upper = rolling max(High, lookback)
      lower = rolling min(Low,  lookback)
      mid   = (upper + lower)/2
    """
    high = df["High"].rolling(window=lookback, min_periods=lookback).max()
    low  = df["Low"].rolling(window=lookback, min_periods=lookback).min()
    mid  = (high + low) / 2.0
    return high, low, mid


# === ATR (Wilder) ===
def ind_donchian_atr_v1_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(np.maximum((h - l), (h - pc).abs()), (l - pc).abs())
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr


# === EMA (trend filter) ===
def ind_donchian_atr_v1_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()


# --- Griglia parametri per WF Optimization (16 combo/finestra) ---
strategy_donchian_atr_v1_param_ranges = {
    # Breakout & exit
    'breakout_len' : [40, 60],   # giorni massimi per ingresso (upper channel)
    'exit_len'     : [15, 25],   # giorni minimi per uscita (lower channel)
    # Trend filter
    'ema_slow'     : [150, 200],
    # Risk control
    'atr_period'   : [14],
    'atr_k'        : [1.5, 2.0],
}
# 2 * 2 * 2 * 1 * 2 = 16 combo/finestra


# --- Funzione di strategia ---
def strategy_donchian_atr_v1(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Logica long-only, semplice e “bull-friendly”:

      Filtro trend:
        • Close > EMA_slow
        • EMA_slow in salita: EMA_slow > EMA_slow.shift(5)

      Entry:
        • Close rompe sopra Donchian upper (breakout_len)

      Exit (qualunque vera):
        • Close rompe sotto Donchian lower (exit_len)
        • Oppure Close < EMA_slow - atr_k * ATR (stop dinamico su trend)
        • Oppure Close < EMA_slow (perdita del regime)

      Note:
        - Indicatori calcolati sull’intero dataset; 'year' è solo maschera a valle.
        - Segnali shiftati di 1 barra per evitare look-ahead.
    """
    pd.set_option('future.no_silent_downcasting', True)
    
    df = data.copy()

    # --- Parametri ---
    n_up   = int(params.get('breakout_len'))
    n_dn   = int(params.get('exit_len'))
    ema_s  = int(params.get('ema_slow'))
    atr_p  = int(params.get('atr_period'))
    atr_k  = float(params.get('atr_k'))

    # --- Indicatori ---
    up, _, _ = ind_donchian_atr_v1_donchian(df, lookback=n_up)
    _, dn, mid_dn = ind_donchian_atr_v1_donchian(df, lookback=n_dn)  # mid_dn non usato, ma utile per debug
    ema_slow = ind_donchian_atr_v1_ema(df, period=ema_s)
    atr = ind_donchian_atr_v1_atr(df, period=atr_p)

    # --- Maschera anno (a valle) ---
    if year is not None:
        df = df[df.index.year == int(year)]
        up = up.loc[df.index]
        dn = dn.loc[df.index]
        ema_slow = ema_slow.loc[df.index]
        atr = atr.loc[df.index]

    # --- Filtro trend ---
    slope_lb = 5
    trend_ok = (df["Close"] > ema_slow) & (ema_slow > ema_slow.shift(slope_lb))

    # --- Entry: breakout sopra upper (ieri non sopra, oggi sopra) + trend ok ---
    entries = (df["Close"].shift(1) <= up.shift(1)) & (df["Close"] > up) & trend_ok

    # --- Exit: sotto lower OR stop ATR su EMA_slow OR sotto EMA_slow ---
    stop_line = ema_slow - atr_k * atr
    exits = (df["Close"] < dn) | (df["Close"] < stop_line) | (df["Close"] < ema_slow)

    # --- Shift di sicurezza ---
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy momentum_rank1d
############################

# === Indicatori ===
def ind_momentum_rank1d_momentum(df: pd.DataFrame,
                                 lb1: int = 20,
                                 lb2: int = 60,
                                 lb3: int = 120,
                                 use_median: bool = True) -> pd.Series:
    """
    Calcola uno 'score' di momentum combinando tre lookback sui 'Close'.
    Score = mediana (o media) dei ritorni % a lb1, lb2, lb3.
    Ritorna: pd.Series (indice = date).
    """
    close = df['Close'].astype(float)
    # Evita shift(0) / lookback non validi
    lb1 = max(1, int(lb1)); lb2 = max(1, int(lb2)); lb3 = max(1, int(lb3))

    mom1 = close / close.shift(lb1) - 1.0
    mom2 = close / close.shift(lb2) - 1.0
    mom3 = close / close.shift(lb3) - 1.0

    mom_df = pd.concat([mom1, mom2, mom3], axis=1)
    if use_median:
        score = mom_df.median(axis=1, skipna=True)
    else:
        score = mom_df.mean(axis=1, skipna=True)

    # Se tutte e tre NaN in una data, metti 0 per robustezza
    score = score.fillna(0.0)
    return score

def ind_momentum_rank1d_ema(df: pd.DataFrame, ema_span: int = 0) -> pd.Series:
    """
    Filtro trend via EMA sui 'Close'. Se ema_span <= 0, ritorna Serie di True.
    """
    if ema_span is None or int(ema_span) <= 0:
        return pd.Series(True, index=df.index)
    close = df['Close'].astype(float)
    ema = close.ewm(span=int(ema_span), adjust=False, min_periods=1).mean()
    # True se prezzo sopra EMA (trend up), False altrimenti
    return (close > ema)

# --- Griglia parametri per WF Optimization (compatta, sicura) ---
strategy_momentum_rank1d_param_ranges = {
    'lb1_range'       : [20, 60],        # ~1m / 3m
    'lb2_range'       : [60, 120],       # ~3m / 6m
    'lb3_range'       : [120, 252],      # ~6m / 12m
    'ema_span_range'  : [0, 50, 100],    # 0 = no trend filter
    'thresh_range'    : [0.00, 0.02, 0.05],  # soglia sullo score (ritorno medio/mediano)
    'use_median_range': [True],          # fisso: mediana (robusta)
}

# --- Funzione di strategia ---
def strategy_momentum_rank1d(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole:
      Entry  se score > thresh  AND (Close > EMA se ema_span>0)
      Exit   se score < -thresh OR  (Close < EMA se ema_span>0)

    Ordine K_Strategy:
      1) calcolo indicatori sull’intero df
      2) slicing per anno
      3) definizione entries/exits
      4) shift di 1 barra e normalizzazione booleana
    """
    # --- Lettura parametri (garantiti SCALARI dalla tua griglia) ---
    lb1        = int(params.get('lb1_range'))
    lb2        = int(params.get('lb2_range'))
    lb3        = int(params.get('lb3_range'))
    ema_span   = int(params.get('ema_span_range'))
    thresh     = float(params.get('thresh_range'))
    use_median = bool(params.get('use_median_range'))

    df = data.copy()

    # --- (1) Indicatori su TUTTO il df ---
    score = ind_momentum_rank1d_momentum(df, lb1=lb1, lb2=lb2, lb3=lb3, use_median=use_median)
    trend_up = ind_momentum_rank1d_ema(df, ema_span=ema_span)

    df['MR1D_Score'] = score
    df['MR1D_Trend'] = trend_up

    # --- (2) Slicing per anno (dopo il calcolo) ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Condizioni entry/exit ---
    if ema_span > 0:
        entries = (df['MR1D_Score'] > thresh) & (df['MR1D_Trend'] == True)
        exits   = (df['MR1D_Score'] < -thresh) | (df['MR1D_Trend'] == False)
    else:
        entries = (df['MR1D_Score'] >  thresh)
        exits   = (df['MR1D_Score'] < -thresh)

    # --- (4) Shift 1 barra + normalizzazione booleana ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy kama_atr
############################

def ind_kama_atr_kama(df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA) sui Close.
    - seed iniziale dopo 'period'
    - gestione NaN per ER/sc
    """
    close = df['Close'].astype(float)
    change = (close - close.shift(period)).abs()
    volatility = (close - close.shift()).abs().rolling(period, min_periods=period).sum()

    er = change / volatility
    # clamp ER tra 0 e 1 per stabilità numerica
    er = er.clip(lower=0.0, upper=1.0)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    # seed: prima KAMA disponibile = close al primo indice valido dopo 'period'
    first_idx = sc.dropna().index.min()
    if pd.isna(first_idx):
        return close * np.nan
    kama.loc[first_idx] = close.loc[first_idx]

    # ricorrenza
    idx_pos = list(close.index.get_indexer_for(close.loc[first_idx:].index))
    for i in range(idx_pos[1], len(close)):
        prev = kama.iloc[i-1]
        if pd.isna(prev):
            kama.iloc[i] = close.iloc[i]  # recovery, rarissimo
        else:
            kama.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)

    return kama

# === Indicatore ATR Percentuale ===
def ind_kama_atr_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range normalizzato in % sul prezzo Close.
    """
    high, low, close = df['High'], df['Low'], df['Close']
    tr = np.maximum(np.maximum((high - low).abs(), (high - close.shift()).abs()), (close.shift() - low).abs())
    atr = tr.rolling(window=period).mean()
    atr_pct = (atr / close) * 100
    return atr_pct


# --- Griglia RIDOTTA per KAMA+ATR (≈ 243 combinazioni) ---
strategy_kama_atr_param_ranges = {
    'kama_period' : range(10, 31, 10),   # 10,20,30
    'fast_period' : range(2, 7, 2),      # 2,4,6
    'slow_period' : range(20, 41, 10),   # 20,30,40
    'atr_period'  : range(10, 31, 10),   # 10,20,30
    'atr_thresh'  : range(1, 4, 1),      # 1,2,3  (% su Close)
}


# --- Funzione di strategia ---
def strategy_kama_atr(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Entry: Close > KAMA  AND ATR% < soglia
    Exit : Close < KAMA  OR  ATR% > soglia
    """
    df = data.copy()
    kama_p   = params.get('kama_period')
    fast_p   = params.get('fast_period')
    slow_p   = params.get('slow_period')
    atr_p    = params.get('atr_period')
    atr_th   = params.get('atr_thresh')

    # --- Calcolo indicatori ---
    df['KAMA']    = ind_kama_atr_kama(df, kama_p, fast_p, slow_p)
    df['ATR_pct'] = ind_kama_atr_atr(df, atr_p)

    # --- Filtro per anno ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Condizioni ---
    entries = (df['Close'] > df['KAMA']) & (df['ATR_pct'] < atr_th)
    exits   = (df['Close'] < df['KAMA']) | (df['ATR_pct'] > atr_th)

    # --- Shift anti-look-ahead ---
    entries = entries.shift(1).astype(bool).fillna(False)
    exits   = exits.shift(1).astype(bool).fillna(False)

    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)
    
############################
# Strategy kama_atr_slope
############################

    
def ind_kama_atr_slope(kama: pd.Series, slope_period: int = 5) -> pd.Series:
    """Stima la pendenza percentuale del KAMA."""
    return (kama - kama.shift(slope_period)) / kama.shift(slope_period) * 100


# ~= 3 * 3 * 3 * 3 * 5 * 3 * 3 = 3.645 combinazioni (<=50k OK)
strategy_kama_atr_slope_param_ranges = {
    'kama_period'  : range(10, 50, 20),   # 10,20,30
    'fast_period'  : range(2, 10, 4),      # 2,4,6
    'slow_period'  : range(20, 50, 10),   # 20,30,40
    'atr_period'   : range(10, 40, 10),   # 10,20,30
    'atr_thresh'   : range(6, 18, 4),     # 6,8,10,12,14  <-- PIÙ ALTA
    'slope_period' : range(3, 12, 3),     # 3,6,9
    'slope_min'    : range(0, 3),      # 0,1,2 -> 0.0%,0.1%,0.2%
}


def strategy_kama_atr_slope(data: pd.DataFrame, params: dict, year: int | None = None):
    df = data.copy()
    k_p  = params['kama_period']
    f_p  = params['fast_period']
    s_p  = params['slow_period']
    a_p  = params['atr_period']
    a_t  = params['atr_thresh']
    sl_p = params['slope_period']
    # interpreta slope_min come decimi di punto %: 0->0.0%, 1->0.1%, 2->0.2%
    sl_m = params['slope_min'] / 10.0

    # 1) Calcolo indicatori
    df['KAMA']    = ind_kama_atr_kama(df, k_p, f_p, s_p)
    df['ATR_pct'] = ind_kama_atr_atr(df, a_p)
    df['Slope']   = ind_kama_atr_slope(df['KAMA'], sl_p)

    # 2) Slice per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Regole
    entries = (df['Close'] > df['KAMA']) & (df['ATR_pct'] < a_t) & (df['Slope'] > sl_m)
    exits   = (df['Close'] < df['KAMA']) | (df['ATR_pct'] > a_t) | (df['Slope'] < 0)

    # 4) Shift anti-look-ahead
    entries = entries.shift(1).astype(bool).fillna(False)
    exits   = exits.shift(1).astype(bool).fillna(False)
    return _safe_shift_fill_bool(entries, 1), _safe_shift_fill_bool(exits, 1)


############################
# Strategy kama_atr_pctile
############################

def ind_kama_atr_pctile_kama(df, period=10, fast=2, slow=30):
    close = df["Close"].astype(float)
    change = (close - close.shift(period)).abs()
    volatility = (close - close.shift()).abs().rolling(period, min_periods=period).sum()
    er = (change / volatility).clip(0, 1)
    fast_sc, slow_sc = 2/(fast+1), 2/(slow+1)
    sc = (er*(fast_sc-slow_sc)+slow_sc)**2
    kama = pd.Series(index=close.index, dtype=float)
    first = sc.dropna().index.min()
    kama.loc[first] = close.loc[first]
    for i in range(close.index.get_loc(first)+1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(close.iloc[i]-kama.iloc[i-1])
    return kama

def ind_kama_atr_pctile_atr_pct(df, period=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(np.maximum((h-l).abs(), (h-c.shift()).abs()), (c.shift()-l).abs())
    return (tr.rolling(period, min_periods=period).mean() / c) * 100

def ind_kama_atr_pctile_thr(atr_pct, lookback=120, q=70):
    """Soglia dinamica: quantile rolling dell’ATR%"""
    return atr_pct.rolling(lookback, min_periods=lookback).quantile(q/100)

def ind_kama_atr_pctile_sma(df, period=200):
    return df["Close"].rolling(period, min_periods=period).mean()

# --- Griglia ridotta (≈324 combinazioni) ---
strategy_kama_atr_pctile_param_ranges = {
    "kama_period"  : range(10,31,10),
    "fast_period"  : range(2,7,2),
    "slow_period"  : range(20,41,10),
    "atr_period"   : range(10,31,10),
    "atr_lookback" : range(60,121,60),
    "atr_qpct"     : range(60,81,10),   # percentile 60–80
    "sma_long"     : range(150,201,50)
}

def strategy_kama_atr_pctile(data, params, year=None):
    df = data.copy()
    k,f,s = params["kama_period"], params["fast_period"], params["slow_period"]
    a_p, lb, q, smaL = params["atr_period"], params["atr_lookback"], params["atr_qpct"], params["sma_long"]

    # 1. Indicatori
    df["KAMA"]     = ind_kama_atr_pctile_kama(df,k,f,s)
    df["ATR_pct"]  = ind_kama_atr_pctile_atr_pct(df,a_p)
    df["ATR_thr"]  = ind_kama_atr_pctile_thr(df["ATR_pct"],lb,q)
    df["SMA_long"] = ind_kama_atr_pctile_sma(df,smaL)

    # 2. Slice per anno
    if year is not None:
        df = df[df.index.year==int(year)]

    # 3. Regole entry/exit
    entries = (df["Close"]>df["KAMA"]) & (df["ATR_pct"]<=df["ATR_thr"]) & (df["Close"]>df["SMA_long"])
    exits   = (df["Close"]<df["KAMA"]) | (df["ATR_pct"]>df["ATR_thr"])  | (df["Close"]<df["SMA_long"])

    # 4) Shift anti-look-ahead — versione robuste e senza warning
    shifted_entries = _safe_shift_fill_bool(entries, shift=1)
    shifted_exits   = _safe_shift_fill_bool(exits,   shift=1)
    
    return shifted_entries, shifted_exits
    
############################
# Strategy vol_regime
############################


# === Indicatori di base ===

def ind_vol_regime_hv(df: pd.DataFrame, hv_lookback: int = 20) -> pd.Series:
    """
    Historical Volatility (HV) semplice: std dei rendimenti su finestra rolling.
    Usa 'Close' (già adjusted nella tua versione di yfinance).
    """
    ret = df['Close'].pct_change()
    hv = ret.rolling(hv_lookback, min_periods=hv_lookback).std()
    return hv

def ind_vol_regime_atr(df: pd.DataFrame, atr_len: int = 14) -> pd.Series:
    """
    ATR classico (True Range su High/Low/Close) con media esponenziale.
    """
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low).abs(), (high - prev_close).abs()), (low - prev_close).abs())
    atr = tr.ewm(span=atr_len, adjust=False, min_periods=1).mean()
    return atr

def ind_vol_regime_bbands(df: pd.DataFrame, bb_len: int = 20, k_mult: float = 2.0):
    """
    Bollinger Bands su 'Close'.
    Ritorna: (middle, upper, lower)
    """
    mid = df['Close'].rolling(bb_len, min_periods=bb_len).mean()
    std = df['Close'].rolling(bb_len, min_periods=bb_len).std(ddof=0)
    upper = mid + k_mult * std
    lower = mid - k_mult * std
    return mid, upper, lower

def ind_vol_regime_ma_roc(df: pd.DataFrame, ma_len: int = 100, roc_len: int = 20):
    """
    Media mobile semplice + ROC percentuale a roc_len.
    Ritorna: (ma, roc_pct)
    """
    ma = df['Close'].rolling(ma_len, min_periods=ma_len).mean()
    roc = df['Close'].pct_change(roc_len)
    return ma, roc

def ind_vol_regime_classify(hv: pd.Series,
                            low_thr_pct: int = 33,
                            high_thr_pct: int = 66,
                            q_lookback: int = 60) -> pd.Series:
    """
    Classifica i regimi di volatilità con soglie percentile ROLLING (causali).
    'q_lookback' è la finestra usata per stimare le soglie dei percentili su HV.
    Output: 0=Low, 1=Medium (Transitional), 2=High
    """
    # rolling quantiles (causali), poi shift per evitare look-ahead
    low_q = hv.rolling(q_lookback, min_periods=q_lookback).quantile(low_thr_pct / 100.0)
    high_q = hv.rolling(q_lookback, min_periods=q_lookback).quantile(high_thr_pct / 100.0)
    # regola di classificazione
    reg = pd.Series(index=hv.index, dtype='float64')
    reg = np.where(hv <= low_q, 0,
          np.where(hv > high_q, 2, 1))
    reg = pd.Series(reg, index=hv.index).astype('Int64')
    # shift di 1 per usare solo info disponibili a barra chiusa
    return reg.shift(1)

def ind_vol_regime_vexp(atr: pd.Series, vexp_ema: int = 10) -> pd.Series:
    """
    Volatility Expansion: ATR vs sua EMA.
    Ritorna: boolean Series (True se espansione: ATR > EMA_ATR)
    """
    ema = atr.ewm(span=vexp_ema, adjust=False, min_periods=1).mean()
    return (atr > ema)


# --- Griglia parametri per WF Optimization (con combinazioni contenute) ---
strategy_vol_regime_param_ranges = {
    # HV e soglie percentile (rolling/causali)
    'hv_lookback_range' : range(20, 100, 40),     # 20, 40, 60
    'q_lookback_range'  : range(60, 180, 60),    # 60, 90, 120
    'low_thr_pct_range' : range(33, 34),      # 33% fisso (range singoletto)
    'high_thr_pct_range': range(66, 67),      # 66% fisso (range singoletto)
    # Bande di Bollinger per regime Low
    'bb_len_range'      : range(10, 50, 20),     # 10, 20, 30
    'bb_k_mult_range'   : range(1, 3),        # 1, 2 (convertito a float)
    # Trend-follow per regime High / Transitional
    'ma_len_range'      : range(50, 275, 75),    # 50,100,150,200
    'roc_len_range'     : range(10, 40, 10),     # 10,20,30
    # Volatility expansion filter
    'atr_len_range'     : range(14, 38, 8),      # 14,22,30
    'vexp_ema_range'    : range(10, 30, 10),     # 10,20
}

# --- Funzione di strategia ---
def strategy_vol_regime(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia adattiva ai regimi di volatilità:
      - Regime Low (MR):   compra su eccesso negativo (Close < BB.lower), esci su ritorno > BB.mid
      - Regime High (Trend): compra se Close > MA e ROC>0; esci se Close < MA o ROC<0
      - Regime Medium (Transitional): richiede Vol Expansion (ATR>EMA_ATR) + conferme trend; esci se Vexp si spegne o Close < MA

    Regole chiave dall’articolo: misurare la volatilità (HV/ATR), classificarla per regimi via percentili,
    e adattare le logiche operative a ciascun regime; gestione/filtri tramite espansione di volatilità. :contentReference[oaicite:2]{index=2}

    NOTE FRAMEWORK:
    - Calcolo indicatori sull’intero df
    - Subito dopo eventuale slicing per anno
    - Definizione entries/exits
    - Shift di 1 barra per evitare look-ahead
    """

    # === Lettura parametri dalla griglia WFO ===
    hv_lb     = params.get('hv_lookback_range')
    q_lb      = params.get('q_lookback_range')
    low_pct   = params.get('low_thr_pct_range')
    high_pct  = params.get('high_thr_pct_range')

    bb_len    = params.get('bb_len_range')
    bb_k_i    = params.get('bb_k_mult_range')     # intero 1.., convertito a float
    ma_len    = params.get('ma_len_range')
    roc_len   = params.get('roc_len_range')

    atr_len   = params.get('atr_len_range')
    vexp_ema  = params.get('vexp_ema_range')

    k_mult = float(bb_k_i)

    df = data.copy()

    # === (1) Calcolo indicatori sull'intero df ===
    # HV e Regimi
    hv = ind_vol_regime_hv(df, hv_lookback=hv_lb)
    df['ATR'] = ind_vol_regime_atr(df, atr_len=atr_len)
    df['Regime'] = ind_vol_regime_classify(hv, low_thr_pct=low_pct, high_thr_pct=high_pct, q_lookback=q_lb)

    # Bande di Bollinger (per regime Low)
    df['BB_mid'], df['BB_up'], df['BB_lo'] = ind_vol_regime_bbands(df, bb_len=bb_len, k_mult=k_mult)

    # Trend-follow (per regime High/Medium)
    df['MA'], df['ROC'] = ind_vol_regime_ma_roc(df, ma_len=ma_len, roc_len=roc_len)

    # Volatility Expansion (Transitional)
    df['VEXP'] = ind_vol_regime_vexp(df['ATR'], vexp_ema=vexp_ema)

    # === (2) Slicing per anno dopo il calcolo degli indicatori ===
    if year is not None:
        df = df[df.index.year == int(year)]

    # === (3) Definizione regole per regime ===
    # 0=Low, 1=Medium, 2=High
    # is_low    = (df['Regime'] == 0)
    # is_medium = (df['Regime'] == 1)
    # is_high   = (df['Regime'] == 2)
    is_low    = df['Regime'].eq(0).fillna(False)
    is_medium = df['Regime'].eq(1).fillna(False)
    is_high   = df['Regime'].eq(2).fillna(False)

    # Low Volatility (Mean Reversion): entry su eccesso negativo, exit su ritorno a media
    entries_low = is_low & (df['Close'] < df['BB_lo'])
    exits_low   = is_low & (df['Close'] > df['BB_mid'])

    # High Volatility (Trend Following): conferme direzionali
    entries_high = is_high & (df['Close'] > df['MA']) & (df['ROC'] > 0)
    exits_high   = is_high & ((df['Close'] < df['MA']) | (df['ROC'] < 0))

    # Medium / Transitional: serve espansione volatilità + conferme
    entries_med = is_medium & df['VEXP'] & (df['Close'] > df['MA']) & (df['ROC'] > 0)
    exits_med   = is_medium & ((~df['VEXP']) | (df['Close'] < df['MA']))

    # Composizione finale (long-only sistematico; nessuna posizione quando le condizioni non sono allineate)
    entries = (entries_low | entries_high | entries_med)
    exits   = (exits_low   | exits_high   | exits_med)

    # # === (4) Shift di 1 barra e normalizzazione booleana ===
    # shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    # shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    # --- Shift di 1 barra e normalizzazione booleana (NA-safe) ---
    shifted_entries = entries.shift(1).fillna(False).astype(bool)
    shifted_exits   = exits.shift(1).fillna(False).astype(bool)

    return shifted_entries, shifted_exits

############################
# Strategy vol_regime_v2
############################


# === Indicatori ===

def ind_vol_regime_v2_hv(df: pd.DataFrame, hv_lookback: int = 20) -> pd.Series:
    """Historical Volatility semplice su rendimenti di Close."""
    ret = df['Close'].pct_change()
    hv = ret.rolling(hv_lookback, min_periods=hv_lookback).std()
    return hv

def ind_vol_regime_v2_atr(df: pd.DataFrame, atr_len: int = 14) -> pd.Series:
    """ATR classico con media esponenziale."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low).abs(), (high - prev_close).abs()), (low - prev_close).abs())
    atr = tr.ewm(span=atr_len, adjust=False, min_periods=1).mean()
    return atr

def ind_vol_regime_v2_bbands(df: pd.DataFrame, bb_len: int = 20, k_mult: float = 2.0):
    """Bollinger Bands su Close. Ritorna (mid, up, lo)."""
    mid = df['Close'].rolling(bb_len, min_periods=bb_len).mean()
    std = df['Close'].rolling(bb_len, min_periods=bb_len).std(ddof=0)
    up = mid + k_mult * std
    lo = mid - k_mult * std
    return mid, up, lo

def ind_vol_regime_v2_emas(df: pd.DataFrame, fast: int = 20, slow: int = 100):
    """EMA veloci/lente per filtro direzionale. Ritorna (ema_fast, ema_slow)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False, min_periods=1).mean()
    return ema_fast, ema_slow

def ind_vol_regime_v2_classify(hv: pd.Series,
                               low_thr_pct: int = 33,
                               high_thr_pct: int = 66,
                               q_lookback: int = 60) -> pd.Series:
    """
    Classifica regimi con percentili ROLLING (causali).
    Output: 0=Low, 1=Medium, 2=High (shiftato di 1 per evitare look-ahead).
    """
    low_q = hv.rolling(q_lookback, min_periods=q_lookback).quantile(low_thr_pct / 100.0)
    high_q = hv.rolling(q_lookback, min_periods=q_lookback).quantile(high_thr_pct / 100.0)
    reg = np.where(hv <= low_q, 0, np.where(hv > high_q, 2, 1))
    reg = pd.Series(reg, index=hv.index).astype('float')  # temporaneo
    return reg.shift(1)

def ind_vol_regime_v2_smooth_regime(regime_raw: pd.Series, span: int = 5) -> pd.Series:
    """
    Smussa il regime con EMA e arrotonda al più vicino intero (0,1,2).
    Shift non necessario qui (regime_raw è già shiftato).
    """
    sm = regime_raw.ewm(span=span, adjust=False, min_periods=1).mean()
    sm = sm.round().clip(lower=0, upper=2).astype('Int64')
    return sm

def ind_vol_regime_v2_risk_off(atr: pd.Series, perc: int = 80, min_exp: int = 50) -> pd.Series:
    """
    Flag Risk-OFF: ATR sopra il percentile 'perc' dell'EXPANDING storico (causale).
    Usiamo expanding().quantile e poi shift(1) per evitare look-ahead.
    """
    thr = atr.expanding(min_periods=min_exp).quantile(perc / 100.0).shift(1)
    risk_off = (atr > thr)
    return risk_off


# --- Griglia parametri per WF Optimization (compatta) ---
strategy_vol_regime_v2_param_ranges = {
    # HV e soglie percentile
    'hv_lookback_range'       : range(60, 61),   # 20,40,60
    'q_lookback_range'        : range(60, 180, 60),  # 60,90,120
    'low_thr_pct_range'       : range(33, 34),    # fisso 33
    'high_thr_pct_range'      : range(66, 67),    # fisso 66
    'smooth_span_range'       : range(3, 11, 4),      # 3,5,7

    # Bande per regime Low
    'bb_len_range'            : range(10, 50, 20),   # 10,20,30
    'bb_k_mult_range'         : range(1, 3),      # 1,2  (-> float)

    # Trend filter EMA
    'ema_fast_range'          : range(10, 50, 20),   # 10,20,30
    'ema_slow_range'          : range(80, 220, 70),  # 80,115,150

    # ATR & Risk-OFF
    'atr_len_range'           : range(14, 46, 16),    # 14,22,30
    'risk_off_q_range'        : range(75, 95, 10),    # 75,80,85
    'risk_off_minexp_range'   : range(50, 150, 50),  # 50,75,100
}

# --- Funzione di strategia ---
def strategy_vol_regime_v2(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Logica:
      • Regime Low  (mean reversion): entra se Close < BB.lower; esci se Close > BB.mid
      • Regime High (trend):          entra se EMA_fast > EMA_slow e non Risk-OFF; esci se EMA_fast < EMA_slow o Risk-OFF
      • Regime Medium:                come High ma più conservativa (stesse regole)

    Framework K_Strategy:
      1) Calcolo indicatori su df intero
      2) Slicing per anno (se passato)
      3) Definizione entries/exits sul df filtrato
      4) Shift di 1 barra e normalizzazione booleana .fillna(False).astype(bool)
    """

    # --- Lettura parametri
    hv_lb      = params.get('hv_lookback_range')
    q_lb       = params.get('q_lookback_range')
    low_pct    = params.get('low_thr_pct_range')
    high_pct   = params.get('high_thr_pct_range')
    sm_span    = params.get('smooth_span_range')

    bb_len     = params.get('bb_len_range')
    bb_k_i     = params.get('bb_k_mult_range')
    ema_fast_p = params.get('ema_fast_range')
    ema_slow_p = params.get('ema_slow_range')

    atr_len    = params.get('atr_len_range')
    ro_q       = params.get('risk_off_q_range')
    ro_minexp  = params.get('risk_off_minexp_range')

    k_mult = float(bb_k_i)

    df = data.copy()

    # === (1) Indicatori su df intero ===
    hv = ind_vol_regime_v2_hv(df, hv_lookback=hv_lb)
    df['ATR'] = ind_vol_regime_v2_atr(df, atr_len=atr_len)

    # Regimi (raw -> smooth)
    reg_raw = ind_vol_regime_v2_classify(hv, low_thr_pct=low_pct, high_thr_pct=high_pct, q_lookback=q_lb)
    df['Regime'] = ind_vol_regime_v2_smooth_regime(reg_raw, span=sm_span)

    # Bande MR
    df['BB_mid'], df['BB_up'], df['BB_lo'] = ind_vol_regime_v2_bbands(df, bb_len=bb_len, k_mult=k_mult)

    # EMA trend filter
    df['EMA_fast'], df['EMA_slow'] = ind_vol_regime_v2_emas(df, fast=ema_fast_p, slow=ema_slow_p)

    # Risk-OFF su ATR (percentile espanso, causale)
    df['RISK_OFF'] = ind_vol_regime_v2_risk_off(df['ATR'], perc=ro_q, min_exp=ro_minexp)

    # === (2) Slicing per anno ===
    if year is not None:
        df = df[df.index.year == int(year)]

    # === (3) Regole operative ===
    is_low    = df['Regime'].eq(0).fillna(False)
    is_medium = df['Regime'].eq(1).fillna(False)
    is_high   = df['Regime'].eq(2).fillna(False)

    # Low: mean reversion su bande
    entries_low = is_low & (df['Close'] < df['BB_lo'])
    exits_low   = is_low & (df['Close'] > df['BB_mid'])

    # High: trend con filtro EMA e blocco risk-off
    trend_long  = df['EMA_fast'] > df['EMA_slow']
    entries_high = is_high & trend_long & (~df['RISK_OFF'].fillna(False))
    exits_high   = is_high & ((~trend_long) | df['RISK_OFF'].fillna(False))

    # Medium: come High (conservativa)
    entries_med = is_medium & trend_long & (~df['RISK_OFF'].fillna(False))
    exits_med   = is_medium & ((~trend_long) | df['RISK_OFF'].fillna(False))

    entries = entries_low | entries_high | entries_med
    exits   = exits_low   | exits_high   | exits_med

    # === (4) Shift 1 barra + normalizzazione booleana (NA-safe) ===
    shifted_entries = entries.shift(1).fillna(False).astype(bool)
    shifted_exits   = exits.shift(1).fillna(False).astype(bool)

    return shifted_entries, shifted_exits
    
# ###########################
# Strategy pcr_ma (robusta)
# ###########################


# === Put/Call Ratio (PCR) con fallback auto-contenuto ===
def ind_pcr_ma_pcr(df: pd.DataFrame, smooth: int = 5, source: str = "auto") -> pd.Series:
    """
    Calcola il Put/Call Ratio smussato.
    Priorità sorgenti:
      1) colonna 'PCR' già presente
      2) 'PutVolume' / 'CallVolume'
      3) 'PutOI' / 'CallOI'
    Fallback AUTOMATICO se le precedenti mancano:
      - 'DUV' proxy: rapporto tra volume in giornate ribassiste e volume in giornate rialziste
        (rolling su 'smooth'). Se 'Volume' non è presente, usa conteggio di giorni down/up.
    """
    src = (source or "auto").lower()

    def _roll_mean(s, w):
        return s.rolling(int(w), min_periods=1).mean()

    if src == "auto":
        if "PCR" in df.columns:
            pcr_raw = pd.to_numeric(df["PCR"], errors="coerce")
            return _roll_mean(pcr_raw, smooth).rename("PCR_S")
        if {"PutVolume", "CallVolume"}.issubset(df.columns):
            puts  = pd.to_numeric(df["PutVolume"], errors="coerce")
            calls = pd.to_numeric(df["CallVolume"], errors="coerce").replace(0, np.nan)
            pcr_raw = puts / calls
            return _roll_mean(pcr_raw, smooth).rename("PCR_S")
        if {"PutOI", "CallOI"}.issubset(df.columns):
            puts  = pd.to_numeric(df["PutOI"], errors="coerce")
            calls = pd.to_numeric(df["CallOI"], errors="coerce").replace(0, np.nan)
            pcr_raw = puts / calls
            return _roll_mean(pcr_raw, smooth).rename("PCR_S")
        # --- Fallback DUV proxy (auto-contenuto) ---
        ret = df["Close"].pct_change()
        w = max(3, int(smooth))

        if "Volume" in df.columns:
            vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
            down = vol.where(ret < 0, 0.0)
            up   = vol.where(ret > 0, 0.0)
        else:
            one  = pd.Series(1.0, index=df.index)
            down = one.where(ret < 0, 0.0)
            up   = one.where(ret > 0, 0.0)

        down_sum = down.rolling(w, min_periods=1).sum()
        up_sum   = up.rolling(w, min_periods=1).sum().replace(0, np.nan)
        pcr_raw  = (down_sum / up_sum)
        pcr_s    = _roll_mean(pcr_raw, w).ffill().bfill()   # << fix: niente fillna(method=...)
        return pcr_s.rename("PCR_S")

    elif src == "volume":
        puts  = pd.to_numeric(df["PutVolume"], errors="coerce")
        calls = pd.to_numeric(df["CallVolume"], errors="coerce").replace(0, np.nan)
        pcr_raw = puts / calls
        return _roll_mean(pcr_raw, smooth).rename("PCR_S")

    elif src == "oi":
        puts  = pd.to_numeric(df["PutOI"], errors="coerce")
        calls = pd.to_numeric(df["CallOI"], errors="coerce").replace(0, np.nan)
        pcr_raw = puts / calls
        return _roll_mean(pcr_raw, smooth).rename("PCR_S")

    else:
        pcr_raw = pd.to_numeric(df["PCR"], errors="coerce")
        return _roll_mean(pcr_raw, smooth).rename("PCR_S")


# === SMA di conferma trend ===
def ind_pcr_ma_sma(df: pd.DataFrame, ma_len: int = 50) -> pd.Series:
    """SMA sui 'Close' per filtro trend (conferma direzionale)."""
    return df["Close"].rolling(int(ma_len), min_periods=1).mean().rename(f"SMA_{ma_len}")


# --- Griglia parametri per WF Optimization ---
strategy_pcr_ma_param_ranges = {
    'pcr_upper_bp_range'  : range(115, 161, 11),   # [1.15 .. 1.50]
    'pcr_lower_bp_range'  : range(55, 88, 8),     # [0.55 .. 0.80]
    'pcr_smooth_range'    : range(3, 10, 2),       # smoothing / finestra proxy
    'ma_len_range'        : range(20, 153, 33),   # SMA filtro (20..120 step 20)
    'direction_mode_range': range(0, 2)        # 0=Contrarian; 1=Blend (PCR+SMA)
}


# --- Funzione di strategia ---
def strategy_pcr_ma(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Segnali basati su PCR (o proxy DUV) contrarian + opzionale filtro SMA.
    I segnali sono shiftati di 1 barra per evitare look-ahead bias.
    """
    upper_bp = int(params.get('pcr_upper_bp_range'))
    lower_bp = int(params.get('pcr_lower_bp_range'))
    smooth   = int(params.get('pcr_smooth_range'))
    ma_len   = int(params.get('ma_len_range'))
    mode     = int(params.get('direction_mode_range', 1))

    pcr_upper = upper_bp / 100.0
    pcr_lower = lower_bp / 100.0

    # (1) indicatori su tutto il df
    df = data.copy()
    df["PCR_S"] = ind_pcr_ma_pcr(df, smooth=smooth, source="auto")
    df[f"SMA_{ma_len}"] = ind_pcr_ma_sma(df, ma_len=ma_len)

    # (2) filtro per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # (3) regole
    if mode == 0:
        entries = (df["PCR_S"] > pcr_upper)
        exits   = (df["PCR_S"] < pcr_lower)
    else:
        sma_col = f"SMA_{ma_len}"
        entries = (df["PCR_S"] > pcr_upper) & (df["Close"] > df[sma_col])
        exits   = (df["PCR_S"] < pcr_lower) & (df["Close"] < df[sma_col])

    # (4) shift + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

    
# ############################
# # Strategy pcr_percentile   -   DISABILITATA: troppo lenta!
# ############################

# # === PCR robusto (con fallback auto-contenuto D/U) ===
# def ind_pcr_percentile_pcr(df: pd.DataFrame, smooth: int = 5, source: str = "auto") -> pd.Series:
#     """
#     Restituisce PCR smussato (colonna 'PCR_S'):
#       Priorità:
#         1) 'PCR'
#         2) 'PutVolume'/'CallVolume'
#         3) 'PutOI'/'CallOI'
#       Fallback se assenti: proxy Down/Up (volume se disponibile, altrimenti conteggio giorni).
#     """
#     src = (source or "auto").lower()

#     def _roll_mean(s, w: int):
#         return s.rolling(int(w), min_periods=1).mean()

#     if src == "auto":
#         if "PCR" in df.columns:
#             pcr_raw = pd.to_numeric(df["PCR"], errors="coerce")
#             return _roll_mean(pcr_raw, smooth).rename("PCR_S")

#         if {"PutVolume", "CallVolume"}.issubset(df.columns):
#             puts  = pd.to_numeric(df["PutVolume"], errors="coerce")
#             calls = pd.to_numeric(df["CallVolume"], errors="coerce").replace(0, np.nan)
#             pcr_raw = puts / calls
#             return _roll_mean(pcr_raw, smooth).rename("PCR_S")

#         if {"PutOI", "CallOI"}.issubset(df.columns):
#             puts  = pd.to_numeric(df["PutOI"], errors="coerce")
#             calls = pd.to_numeric(df["CallOI"], errors="coerce").replace(0, np.nan)
#             pcr_raw = puts / calls
#             return _roll_mean(pcr_raw, smooth).rename("PCR_S")

#         # --- Fallback D/U auto-contenuto ---
#         ret = df["Close"].pct_change()
#         w = max(3, int(smooth))

#         if "Volume" in df.columns:
#             vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
#             down = vol.where(ret < 0, 0.0)
#             up   = vol.where(ret > 0, 0.0)
#         else:
#             one  = pd.Series(1.0, index=df.index)
#             down = one.where(ret < 0, 0.0)
#             up   = one.where(ret > 0, 0.0)

#         down_sum = down.rolling(w, min_periods=1).sum()
#         up_sum   = up.rolling(w, min_periods=1).sum().replace(0, np.nan)
#         pcr_raw  = (down_sum / up_sum)
#         pcr_s    = _roll_mean(pcr_raw, w).ffill().bfill()
#         return pcr_s.rename("PCR_S")

#     elif src == "volume":
#         puts  = pd.to_numeric(df["PutVolume"], errors="coerce")
#         calls = pd.to_numeric(df["CallVolume"], errors="coerce").replace(0, np.nan)
#         pcr_raw = puts / calls
#         return _roll_mean(pcr_raw, smooth).rename("PCR_S")

#     elif src == "oi":
#         puts  = pd.to_numeric(df["PutOI"], errors="coerce")
#         calls = pd.to_numeric(df["CallOI"], errors="coerce").replace(0, np.nan)
#         pcr_raw = puts / calls
#         return _roll_mean(pcr_raw, smooth).rename("PCR_S")

#     # default: assume colonna 'PCR'
#     pcr_raw = pd.to_numeric(df.get("PCR", pd.Series(index=df.index, dtype=float)), errors="coerce")
#     return _roll_mean(pcr_raw, smooth).rename("PCR_S")


# # === Percentile rolling del PCR_S ===
# def ind_pcr_percentile_rank(df: pd.DataFrame, window: int = 126, col: str = "PCR_S") -> pd.Series:
#     """
#     Calcola il percentile rolling (0..1) del valore corrente di 'col' (default: PCR_S)
#     rispetto alla finestra 'window'. Usa rolling.apply con rank pct dell'ultimo elemento.
#     """
#     s = pd.to_numeric(df[col], errors="coerce")

#     def last_rank_pct(x):
#         # x è una Serie; rank pct dell'ultimo elemento nella finestra
#         xs = pd.Series(x)
#         return xs.rank(pct=True).iloc[-1]

#     perc = s.rolling(int(window), min_periods=5).apply(last_rank_pct, raw=False)
#     return perc.rename(f"{col}_PERC_{int(window)}")


# # === SMA di conferma trend (opzionale) ===
# def ind_pcr_percentile_sma(df: pd.DataFrame, ma_len: int = 200) -> pd.Series:
#     """SMA sui Close (se ma_len>0)."""
#     return df["Close"].rolling(int(ma_len), min_periods=1).mean().rename(f"SMA_{ma_len}")


# # --- Griglia parametri per WF Optimization ---
# strategy_pcr_percentile_param_ranges = {
#     'pcr_smooth_range' : range(3, 9, 1),          # smoothing PCR/DUV (3..8)
#     'perc_window_range': range(60, 181, 30),      # finestra percentile (60..180 step 30)
#     'upper_pct_range'  : range(70, 91, 5),        # soglia alta percentile (70..90)
#     'lower_pct_range'  : range(10, 31, 5),        # soglia bassa percentile (10..30)
#     'ma_len_range'     : range(0, 251, 50),       # 0=nessun filtro, poi 50..250
#     'mode_range'       : range(0, 2, 1)           # 0=Trend-Follow; 1=Contrarian
# }


# # --- Funzione di strategia ---
# def strategy_pcr_percentile(data: pd.DataFrame, params: dict, year: int | None = None):
#     """
#     Genera segnali usando percentile rolling del PCR (o proxy D/U).
#     Modalità:
#       0) Trend-Follow:    ENTRY quando percentile <= lower_pct  (+ opz. Close>SMA)
#                           EXIT  quando percentile >= upper_pct  (+ opz. Close<SMA)
#       1) Contrarian:      ENTRY quando percentile >= upper_pct  (+ opz. Close>SMA)
#                           EXIT  quando percentile <= lower_pct  (+ opz. Close<SMA)

#     Note:
#     - Percentili espressi in % (es. 80 -> 0.80).
#     - Segnali shiftati di 1 barra per evitare look-ahead.
#     """
#     # --- Parametri
#     pcr_smooth = int(params.get('pcr_smooth_range'))
#     perc_win   = int(params.get('perc_window_range'))
#     upper_pct  = int(params.get('upper_pct_range')) / 100.0
#     lower_pct  = int(params.get('lower_pct_range')) / 100.0
#     ma_len     = int(params.get('ma_len_range'))
#     mode       = int(params.get('mode_range', 0))

#     # --- (1) Indicatori su tutto il df
#     df = data.copy()
#     df["PCR_S"] = ind_pcr_percentile_pcr(df, smooth=pcr_smooth, source="auto")
#     df["PCR_PERC"] = ind_pcr_percentile_rank(df, window=perc_win, col="PCR_S")

#     use_sma = ma_len > 0
#     if use_sma:
#         sma_col = f"SMA_{ma_len}"
#         df[sma_col] = ind_pcr_percentile_sma(df, ma_len=ma_len)

#     # --- (2) Slicing per anno
#     if year is not None:
#         df = df[df.index.year == int(year)]

#     # --- (3) Regole entries/exits
#     perc = df["PCR_PERC"]
#     if mode == 0:
#         # Trend-follow: cavalca euforia (percentile basso) e chiudi su “paura” (percentile alto)
#         entries = (perc <= lower_pct)
#         exits   = (perc >= upper_pct)
#     else:
#         # Contrarian: compra su paura, esci su euforia
#         entries = (perc >= upper_pct)
#         exits   = (perc <= lower_pct)

#     if use_sma:
#         sma_col = f"SMA_{ma_len}"
#         entries = entries & (df["Close"] > df[sma_col])
#         exits   = exits   & (df["Close"] < df[sma_col])

#     # --- (4) Shift di 1 barra + normalizzazione
#     shifted_entries = entries.shift(1).astype(bool).fillna(False)
#     shifted_exits   = exits.shift(1).astype(bool).fillna(False)

#     return shifted_entries, shifted_exits
    
############################
# Strategy qqq_trend_vol
############################

# === EMAs (trend) ===
def ind_qqq_trend_vol_emas(df: pd.DataFrame, fast: int = 20, slow: int = 200):
    """
    Restituisce (EMA_fast, EMA_slow) su 'Close' (adjusted).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_f = close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_{int(slow)}")
    return ema_f, ema_s

# === Realized Vol + Percentile rolling (risk switch) ===
def ind_qqq_trend_vol_rvperc(df: pd.DataFrame, stdev_win: int = 20, perc_win: int = 120):
    """
    Realized vol annualizzata (std rolling * sqrt(252)) e suo percentile rolling (0..1).
    Ritorna: (rv_ann, rv_perc)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    rv_ann = ret.rolling(int(stdev_win), min_periods=stdev_win//2).std().mul(np.sqrt(252)).rename("RV")
    # Percentile rolling della RV: rank pct dell'ultimo valore nella finestra
    def last_rank_pct(x):
        xs = pd.Series(x)
        return xs.rank(pct=True).iloc[-1]
    rv_perc = rv_ann.rolling(int(perc_win), min_periods=5).apply(last_rank_pct, raw=False).ffill().bfill().rename("RV_PERC")
    return rv_ann, rv_perc

# === Momentum semplice (ROC) ===
def ind_qqq_trend_vol_roc(df: pd.DataFrame, window: int = 63):
    """
    Rate of Change a 'window' giorni: Close/Close.shift(window) - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    roc = (close / close.shift(int(window)) - 1.0).rename(f"ROC_{int(window)}")
    return roc

# === Breakout sui Close (Donchian-like) ===
def ind_qqq_trend_vol_breakout(df: pd.DataFrame, window: int = 100):
    """
    Soglia breakout: max dei Close su finestra 'window' (shiftata di 1 barra).
    ENTRY se Close > breakout_line.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    breakout_line = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return breakout_line

# --- Griglia parametri per WF Optimization (contenuta) ---
strategy_qqq_trend_vol_param_ranges = {
    'ema_fast_range'      : range(10, 70, 20),    # 10,20,30,40
    'ema_slow_range'      : range(150, 300, 50),  # 150,200,250
    'vol_perc_win_range'  : range(90, 225, 45),   # 90,120,150,180
    'vol_upper_pct_range' : range(70, 100, 10),     # 70..90 (percentile RV per risk-off)
    'roc_window_range'    : range(63, 189, 63),   # 63,126
    'breakout_win_range'  : range(100, 250, 50),  # 100,150,200
    'mode_range'          : range(0, 2)        # 0=Momentum (ROC), 1=Breakout
}
# Combinazioni: 4*3*4*5*2*3*2 = 2880

# --- Funzione di strategia ---
def strategy_qqq_trend_vol(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia per QQQ: trend-follow con interruttore di rischio su volatilità.
    Logica comune:
      - Risk-ON se Close > EMA_slow AND EMA_fast >= EMA_slow AND RV_PERC < vol_upper_pct
      - Risk-OFF altrimenti (taglio rischio: si esce)

    Modalità:
      Mode 0 (Momentum): ENTRY se Risk-ON AND ROC > 0 AND Close > EMA_fast
                         EXIT  se Risk-OFF OR ROC <= 0 OR Close < EMA_fast
      Mode 1 (Breakout): ENTRY se Risk-ON AND Close > breakout_line
                         EXIT  se Risk-OFF OR Close < EMA_fast

    Segnali shiftati di 1 barra per evitare look-ahead.
    """
    # --- Lettura parametri
    ema_f  = int(params.get('ema_fast_range'))
    ema_s  = int(params.get('ema_slow_range'))
    vp_win = int(params.get('vol_perc_win_range'))
    vol_up = int(params.get('vol_upper_pct_range')) / 100.0
    roc_w  = int(params.get('roc_window_range'))
    brk_w  = int(params.get('breakout_win_range'))
    mode   = int(params.get('mode_range', 0))

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df[f"EMA_{ema_f}"], df[f"EMA_{ema_s}"] = ind_qqq_trend_vol_emas(df, fast=ema_f, slow=ema_s)
    df["RV"], df["RV_PERC"] = ind_qqq_trend_vol_rvperc(df, stdev_win=20, perc_win=vp_win)
    df[f"ROC_{roc_w}"] = ind_qqq_trend_vol_roc(df, window=roc_w)
    df[f"BRK_{brk_w}"] = ind_qqq_trend_vol_breakout(df, window=brk_w)

    # --- (2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close   = df["Close"]
    ema_fast = df[f"EMA_{ema_f}"]
    ema_slow = df[f"EMA_{ema_s}"]
    rvp     = df["RV_PERC"]

    risk_on  = (close > ema_slow) & (ema_fast >= ema_slow) & (rvp < vol_up)
    risk_off = ~risk_on

    if mode == 0:
        roc = df[f"ROC_{roc_w}"]
        entries = risk_on & (roc > 0.0) & (close > ema_fast)
        exits   = risk_off | (roc <= 0.0) | (close < ema_fast)
    else:
        brk = df[f"BRK_{brk_w}"]
        entries = risk_on & (close > brk)
        exits   = risk_off | (close < ema_fast)

    # --- (4) Shift di 1 barra e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy qqq_dual_mom
############################

# === EMAs per regime ===
def ind_qqq_dual_mom_emas(df: pd.DataFrame, fast: int = 30, slow: int = 200):
    """
    Restituisce (EMA_fast, EMA_slow) sui Close.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_f = close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_{int(slow)}")
    return ema_f, ema_s

# === Dual Momentum (ROC breve/medio) ===
def ind_qqq_dual_mom_rocs(df: pd.DataFrame, win1: int = 63, win2: int = 126):
    """
    Ritorna (ROC_win1, ROC_win2) = Close/Close.shift(win) - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    roc1 = (close / close.shift(int(win1)) - 1.0).rename(f"ROC_{int(win1)}")
    roc2 = (close / close.shift(int(win2)) - 1.0).rename(f"ROC_{int(win2)}")
    return roc1, roc2

# === Picco rolling e drawdown percentuale (solo Close) ===
def ind_qqq_dual_mom_peakdd(df: pd.DataFrame, peak_win: int = 252):
    """
    Ritorna (rolling_peak, drawdown) dove drawdown = Close/rolling_peak - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak  = close.rolling(int(peak_win), min_periods=1).max().rename(f"PEAK_{int(peak_win)}")
    dd    = (close / peak - 1.0).rename(f"DD_{int(peak_win)}")
    return peak, dd


# --- Griglia parametri per WF Optimization (≈5.2k combo) ---
strategy_qqq_dual_mom_param_ranges = {
    'ema_fast_range'        : range(20, 60, 20),     # 20,30,40
    'ema_slow_range'        : range(180, 300, 60),   # 180,200,220,240
    'roc1_win_range'        : range(42, 105, 21),     # 42,63,84
    'roc2_win_range'        : range(84, 210, 42),    # 84,126,168
    'spread_thr_bp_range'   : range(0, 45, 15),      # soglia ((EMA_f-EMA_s)/Close)*1e4  -> 0..30 bps
    'confirm_days_range'    : range(0, 6, 2),        # 0,2,4 giorni consecutivi sopra EMA_slow
    'dd_stop_bp_range'      : range(1500, 3750, 750) # stop su drawdown da picco 15%,20%,25%,30%
}


# --- Funzione di strategia ---
def strategy_qqq_dual_mom(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regime long-only ad alta permanenza:
      - Trend ON se: Close > EMA_slow
                      & ROC_win1 > 0
                      & ROC_win2 > 0
                      & ((EMA_fast-EMA_slow)/Close)*1e4 >= spread_thr_bp
                      & (opz.) 'confirm_days' di fila sopra EMA_slow
      - ENTRY  quando Trend ON
      - EXIT   quando !Trend ON  oppure drawdown <= -dd_stop

    Segnali shiftati di 1 barra per evitare look-ahead bias.
    """
    # --- Parametri
    ema_f   = int(params.get('ema_fast_range'))
    ema_s   = int(params.get('ema_slow_range'))
    w1      = int(params.get('roc1_win_range'))
    w2      = int(params.get('roc2_win_range'))
    spr_bp  = int(params.get('spread_thr_bp_range'))
    conf_n  = int(params.get('confirm_days_range'))
    dd_bp   = int(params.get('dd_stop_bp_range'))      # es. 2000 = 20%

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df[f"EMA_{ema_f}"], df[f"EMA_{ema_s}"] = ind_qqq_dual_mom_emas(df, fast=ema_f, slow=ema_s)
    df[f"ROC_{w1}"], df[f"ROC_{w2}"] = ind_qqq_dual_mom_rocs(df, win1=w1, win2=w2)
    df["PEAK_252"], df["DD_252"] = ind_qqq_dual_mom_peakdd(df, peak_win=252)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close    = df["Close"]
    ema_fast = df[f"EMA_{ema_f}"]
    ema_slow = df[f"EMA_{ema_s}"]
    roc1     = df[f"ROC_{w1}"]
    roc2     = df[f"ROC_{w2}"]

    spread_bps = ((ema_fast - ema_slow) / close) * 1e4
    base_on = (close > ema_slow) & (roc1 > 0.0) & (roc2 > 0.0) & (spread_bps >= spr_bp)

    if conf_n > 0:
        above = (close > ema_slow).astype(int)
        # True se ultimi 'conf_n' giorni tutti sopra EMA_slow
        confirm = above.rolling(conf_n, min_periods=conf_n).sum() == conf_n
        trend_on = base_on & confirm
    else:
        trend_on = base_on

    # Stop su drawdown dal picco rolling a 252 giorni
    dd_thr = -dd_bp / 10000.0
    dd_stop = (df["DD_252"] <= dd_thr)

    entries = trend_on
    exits   = (~trend_on) | dd_stop

    # --- (4) Shift di 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy qqq_ath_trail
############################

# === Rolling ATH e Drawdown (% dal picco) ===
def ind_qqq_ath_trail_peakdd(df: pd.DataFrame, peak_win: int = 252):
    """
    Restituisce:
      - PEAK: massimo rolling dei Close su 'peak_win'
      - DD  : drawdown percentuale = Close/PEAK - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak  = close.rolling(int(peak_win), min_periods=1).max().rename(f"PEAK_{int(peak_win)}")
    dd    = (close / peak - 1.0).rename(f"DD_{int(peak_win)}")
    return peak, dd

# === EMA lenta per regime e sua pendenza ===
def ind_qqq_ath_trail_ema(df: pd.DataFrame, slow: int = 200, slope_lb: int = 20):
    """
    Restituisce:
      - EMA_slow sui Close
      - SLOPE_POS: True se EMA_slow > EMA_slow.shift(slope_lb) (pendenza positiva)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_{int(slow)}")
    slope_pos = (ema_s > ema_s.shift(int(slope_lb))).rename(f"SLOPE_POS_{int(slope_lb)}")
    return ema_s, slope_pos

# --- Griglia parametri per WF Optimization (≈648 combo) ---
strategy_qqq_ath_trail_param_ranges = {
    'peak_win_range'     : range(168, 379, 105),   # 168, 273, 378 (≈8m, 13m, 18m)
    'dd_stop_bp_range'   : range(1200, 2601, 400), # 12%, 16%, 20%, 24%
    'ema_slow_range'     : range(150, 251, 50),    # 150, 200, 250
    'slope_lb_range'     : range(15, 41, 10),      # 15, 25, 35
    'reentry_mode_range' : range(0, 3, 1),         # 0=New High; 1=EMA Recovery; 2=Either
    'nh_buffer_bp_range' : range(0, 21, 10)        # extra breakout su nuovo massimo: 0,10,20 bps
}
# Tot: 3*4*3*3*3*3 = 972  (se vuoi ridurre: togli una delle slope_lb)

# --- Funzione di strategia ---
def strategy_qqq_ath_trail(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Logica:
      • Uscita primaria: quando il drawdown dal massimo rolling supera 'dd_stop' (es. -20%).
      • Uscita secondaria: quando Close < EMA_slow E la pendenza dell'EMA_slow è negativa.
      • Rientro (hysteresis):
          Mode 0 - NEW HIGH: Close >= PEAK_prev * (1 + nh_buffer)
          Mode 1 - EMA RECOVERY: Close > EMA_slow AND pendenza EMA_slow positiva
          Mode 2 - EITHER: una delle due condizioni sopra
    Segnali shiftati di 1 barra per evitare look-ahead bias.
    """
    peak_win   = int(params.get('peak_win_range'))
    dd_bp      = int(params.get('dd_stop_bp_range'))        # es. 2000 = 20%
    ema_slow_p = int(params.get('ema_slow_range'))
    slope_lb   = int(params.get('slope_lb_range'))
    re_mode    = int(params.get('reentry_mode_range', 0))
    nh_buf_bp  = int(params.get('nh_buffer_bp_range', 0))

    # (1) Indicatori su tutto il df
    df = data.copy()
    peak_col, dd_col = ind_qqq_ath_trail_peakdd(df, peak_win=peak_win)
    ema_slow_col, slope_pos_col = ind_qqq_ath_trail_ema(df, slow=ema_slow_p, slope_lb=slope_lb)
    df['PEAK']      = peak_col
    df['DD']        = dd_col
    df['EMA_SLOW']  = ema_slow_col
    df['SLOPE_POS'] = slope_pos_col

    # (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # (3) Regole entries/exits
    close     = df["Close"]
    peak_prev = df["PEAK"].shift(1)
    nh_buffer = 1.0 + nh_buf_bp / 10000.0
    dd_stop   = (df["DD"] <= -dd_bp / 10000.0)

    # Exit secondaria: trend debole
    trend_weak = (close < df["EMA_SLOW"]) & (~df["SLOPE_POS"])

    # Condizioni di rientro
    cond_new_high   = (close >= (peak_prev * nh_buffer))
    cond_ema_reco   = (close > df["EMA_SLOW"]) & (df["SLOPE_POS"])

    if re_mode == 0:       # solo new high
        entries = cond_new_high
    elif re_mode == 1:     # solo EMA recovery
        entries = cond_ema_reco
    else:                  # either
        entries = cond_new_high | cond_ema_reco

    exits = dd_stop | trend_weak

    # (4) Shift di 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_uptrend_dip
############################

# === EMA intermedia + z-score del "distance-to-EMA" ===
def ind_spy_uptrend_dip_stats(df: pd.DataFrame, ema_mid: int = 50, std_win: int = 20):
    """
    Calcola:
      - EMA_MID: EMA(ema_mid) dei Close
      - Z_MID:  z-score del rapporto (Close/EMA_MID - 1) normalizzato con la sua std rolling (std_win)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_m = close.ewm(span=int(ema_mid), adjust=False, min_periods=1).mean().rename(f"EMA_MID_{int(ema_mid)}")
    ratio = (close / ema_m) - 1.0
    rstd  = ratio.rolling(int(std_win), min_periods=max(5, int(std_win)//2)).std().replace(0, np.nan)
    z_mid = (ratio / rstd).rename(f"Z_MID_{int(ema_mid)}_{int(std_win)}")
    return ema_m, z_mid

# === EMA lenta + pendenza (regime di fondo) ===
def ind_spy_uptrend_dip_emaslow(df: pd.DataFrame, slow: int = 200, slope_lb: int = 25):
    """
    Restituisce:
      - EMA_SLOW: EMA(slow) dei Close
      - SLOPE_POS: True se EMA_SLOW > EMA_SLOW.shift(slope_lb)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_SLOW_{int(slow)}")
    slope_pos = (ema_s > ema_s.shift(int(slope_lb))).rename(f"SLOPE_POS_{int(slope_lb)}")
    return ema_s, slope_pos

# === Realized Vol percentile (crash detector) ===
def ind_spy_uptrend_dip_rvperc(df: pd.DataFrame, stdev_win: int = 20, perc_win: int = 120):
    """
    Volatilità realizzata annualizzata e suo percentile rolling (0..1).
    Ritorna: (RV, RV_PERC)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    rv = ret.rolling(int(stdev_win), min_periods=max(5, int(stdev_win)//2)).std().mul(np.sqrt(252)).rename("RV")
    def _last_rank_pct(x):
        xs = pd.Series(x)
        return xs.rank(pct=True).iloc[-1]
    rvp = rv.rolling(int(perc_win), min_periods=5).apply(_last_rank_pct, raw=False).ffill().bfill().rename("RV_PERC")
    return rv, rvp

# === Linea breakout su massimi recenti (rientro rapido) ===
def ind_spy_uptrend_dip_breakout(df: pd.DataFrame, window: int = 126):
    """
    Ritorna la linea di breakout = massimo rolling dei Close (shiftato di 1).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk

# --- Griglia parametri per WF Optimization (≈8.6k combo, entro la policy) ---
strategy_spy_uptrend_dip_param_ranges = {
    'ema_mid_range'       : range(40, 100, 30),    # 40,50,60,70
    'ema_slow_range'      : range(180, 300, 60),  # 180,200,220,240
    'std_win_range'       : range(15, 45, 15),     # 15,20,25,30
    'z_buy_tenths_range'  : range(8, 24, 8),      # 0.8..1.6 (interpreto come soglia su -z)
    'vol_upper_pct_range' : range(75, 105, 15),     # 75..90 (percentile RV per risk-off)
    'brk_win_range'       : range(100, 250, 50),  # 100,150,200
    'slope_lb_range'      : range(15, 45, 10),    # 15,25,35
    'reentry_mode_range'  : range(0, 3)        # 0=EMA-recovery or Dip ; 1=Breakout ; 2=Either
}
# Nota: se vuoi ridurre ancora le combinazioni, fissa 'slope_lb_range' a range(25,26)

# --- Funzione di strategia ---
def strategy_spy_uptrend_dip(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY – Uptrend & Buy-the-Dip con crash guard:
      • Risk-OFF solo quando: Close < EMA_SLOW  AND  SLOPE_POS=False  AND  RV_PERC >= vol_upper_pct.
      • Rientro (dip o trend recovery) a seconda del 'reentry_mode':
          - Mode 0: EMA recovery (Close > EMA_SLOW & SLOPE_POS)  OR  Buy-the-dip (Close > EMA_SLOW & Z_MID <= -z_thr)
          - Mode 1: Breakout (Close > BRK_line)
          - Mode 2: Either (Mode0 OR Mode1)
      • Segnali shiftati di 1 barra per evitare look-ahead.
    """
    # --- Parametri dalla griglia
    ema_mid   = int(params.get('ema_mid_range'))
    ema_slow  = int(params.get('ema_slow_range'))
    std_win   = int(params.get('std_win_range'))
    z_tenths  = int(params.get('z_buy_tenths_range'))   # es. 10 => soglia -1.0
    vol_up    = int(params.get('vol_upper_pct_range')) / 100.0
    brk_win   = int(params.get('brk_win_range'))
    slope_lb  = int(params.get('slope_lb_range'))
    mode      = int(params.get('reentry_mode_range', 0))

    z_thr = z_tenths / 10.0  # soglia z-score in deviazioni standard

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df[f"EMA_MID_{ema_mid}"], df["Z_MID"] = ind_spy_uptrend_dip_stats(df, ema_mid=ema_mid, std_win=std_win)
    df[f"EMA_SLOW_{ema_slow}"], df["SLOPE_POS"] = ind_spy_uptrend_dip_emaslow(df, slow=ema_slow, slope_lb=slope_lb)
    df["RV"], df["RV_PERC"] = ind_spy_uptrend_dip_rvperc(df, stdev_win=20, perc_win=120)
    df[f"BRK_{brk_win}"] = ind_spy_uptrend_dip_breakout(df, window=brk_win)

    # --- (2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close    = df["Close"]
    emaSlow  = df[f"EMA_SLOW_{ema_slow}"]
    zmid     = df["Z_MID"]
    rvp      = df["RV_PERC"]

    # Condizione di crash / risk-off (molto selettiva, per non restare fuori troppo)
    risk_off = (close < emaSlow) & (~df["SLOPE_POS"]) & (rvp >= vol_up)

    # Condizioni di rientro
    cond_ema_reco = (close > emaSlow) & (df["SLOPE_POS"])
    # Buy-the-dip: forte pullback ma sopra EMA lenta (evita catching falling knives)
    cond_dip_buy  = (close > emaSlow) & (zmid <= -z_thr)
    cond_breakout = (close > df[f"BRK_{brk_win}"])

    if mode == 0:
        entries = cond_ema_reco | cond_dip_buy
    elif mode == 1:
        entries = cond_breakout
    else:
        entries = cond_breakout | cond_ema_reco | cond_dip_buy

    exits = risk_off

    # --- (4) Shift + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy spy_quarter_switch
############################

# === EMA lenta (regime di fondo) ===
def ind_spy_quarter_switch_ema(df: pd.DataFrame, slow: int = 200) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_{int(slow)}")
    return ema_s

# === Momentum trimestrale (ROC 63g) ===
def ind_spy_quarter_switch_roc(df: pd.DataFrame, window: int = 63) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    roc = (close / close.shift(int(window)) - 1.0).rename(f"ROC_{int(window)}")
    return roc

# === Realized Vol percentile (shock detector) ===
def ind_spy_quarter_switch_rvperc(df: pd.DataFrame, stdev_win: int = 20, perc_win: int = 120):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    rv = ret.rolling(int(stdev_win), min_periods=max(5, int(stdev_win)//2)).std().mul(np.sqrt(252)).rename("RV")
    def _last_rank_pct(x):
        xs = pd.Series(x)
        return xs.rank(pct=True).iloc[-1]
    rvp = rv.rolling(int(perc_win), min_periods=5).apply(_last_rank_pct, raw=False).ffill().bfill().rename("RV_PERC")
    return rv, rvp

# === Breakout su massimi recenti (re-entry accelerato) ===
def ind_spy_quarter_switch_breakout(df: pd.DataFrame, window: int = 126) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk

# === Pendenza dell'EMA lenta (hysteresis sul rientro) ===
def ind_spy_quarter_switch_slope(df: pd.DataFrame, slow: int = 200, slope_lb: int = 25) -> pd.Series:
    ema_s = ind_spy_quarter_switch_ema(df, slow=slow)
    slope_pos = (ema_s > ema_s.shift(int(slope_lb))).rename(f"SLOPE_POS_{int(slow)}_{int(slope_lb)}")
    return slope_pos

# --- Griglia parametri per WF Optimization (~2016 combo) ---
strategy_spy_quarter_switch_param_ranges = {
    'ema_slow_range'        : range(180, 300, 60),   # 180,200,220,240
    'roc_win_range'         : range(63, 64),         # fisso 63 (trimestre) -> usa range per compatibilità
    'roc_neg_thr_bp_range'  : range(300, 1650, 450), # 3%..12% step 1.5% (bps)
    'rv_upper_pct_range'    : range(75, 98, 11),      # 75,80,85,90
    'persist_days_range'    : range(3, 9, 2),        # 3,5,7
    'breakout_win_range'    : range(100, 250, 50),   # 100,150,200
    'slope_lb_range'        : range(20, 40, 10),     # 20,30
    'reentry_mode_range'    : range(0, 3)         # 0=EMA+ROC (con slope); 1=Breakout; 2=Either
}

# --- Funzione di strategia ---
def strategy_spy_quarter_switch(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY – Quarterly Switch:
      • EXIT (Risk-OFF) quando *persistono* per 'persist_days':
          Close < EMA_slow  AND  ROC_63 <= -thr  AND  RV_PERC >= vol_upper_pct
      • ENTRY (Risk-ON) a seconda di 'reentry_mode':
          0: (Close > EMA_slow) & (ROC_63 >= 0) & (SLOPE_POS=True)
          1: (Close > BRK_line)
          2: (Mode0 OR Mode1)
    Segnali shiftati di 1 barra per evitare look-ahead.
    """
    ema_slow   = int(params.get('ema_slow_range'))
    roc_win    = int(params.get('roc_win_range'))          # 63
    roc_thr_bp = int(params.get('roc_neg_thr_bp_range'))   # bps, es. 600 => -6%
    rv_up      = int(params.get('rv_upper_pct_range')) / 100.0
    persist_n  = int(params.get('persist_days_range'))
    brk_win    = int(params.get('breakout_win_range'))
    slope_lb   = int(params.get('slope_lb_range'))
    mode       = int(params.get('reentry_mode_range', 0))

    thr_neg = -roc_thr_bp / 10000.0

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df[f"EMA_{ema_slow}"] = ind_spy_quarter_switch_ema(df, slow=ema_slow)
    df[f"ROC_{roc_win}"]  = ind_spy_quarter_switch_roc(df, window=roc_win)
    df["RV"], df["RV_PERC"] = ind_spy_quarter_switch_rvperc(df, stdev_win=20, perc_win=120)
    df[f"BRK_{brk_win}"] = ind_spy_quarter_switch_breakout(df, window=brk_win)
    df[f"SLOPE_POS_{ema_slow}_{slope_lb}"] = ind_spy_quarter_switch_slope(df, slow=ema_slow, slope_lb=slope_lb)

    # --- (2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close  = df["Close"]
    emaS   = df[f"EMA_{ema_slow}"]
    roc63  = df[f"ROC_{roc_win}"]
    rvp    = df["RV_PERC"]
    slope_ok = df[f"SLOPE_POS_{ema_slow}_{slope_lb}"]
    brk    = df[f"BRK_{brk_win}"]

    risk_off_base = (close < emaS) & (roc63 <= thr_neg) & (rvp >= rv_up)
    risk_off = (risk_off_base.rolling(persist_n, min_periods=persist_n).sum() == persist_n)

    cond_ema_reco = (close > emaS) & (roc63 >= 0.0) & (slope_ok)
    cond_breakout = (close > brk)

    if mode == 0:
        entries = cond_ema_reco
    elif mode == 1:
        entries = cond_breakout
    else:
        entries = cond_ema_reco | cond_breakout

    exits = risk_off

    # --- (4) Shift + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_ath_fastre
############################

# === ATH rolling e drawdown % ===
def ind_spy_ath_fastre_peakdd(df: pd.DataFrame, peak_win: int = 252):
    """
    Restituisce:
      PEAK: massimo rolling dei Close su 'peak_win'
      DD  : drawdown percentuale = Close/PEAK - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak  = close.rolling(int(peak_win), min_periods=1).max().rename(f"PEAK_{int(peak_win)}")
    dd    = (close / peak - 1.0).rename(f"DD_{int(peak_win)}")
    return peak, dd

# === EMA di conferma trend (veloce rispetto a SPY) ===
def ind_spy_ath_fastre_ema(df: pd.DataFrame, fast: int = 100):
    """
    Restituisce EMA_fast sui Close.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_f = close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")
    return ema_f

# === Breakout su massimi recenti (rientro accelerato) ===
def ind_spy_ath_fastre_breakout(df: pd.DataFrame, window: int = 100):
    """
    Linea di breakout = massimo rolling dei Close (shiftato di 1 barra).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk

# --- Griglia parametri per WF Optimization (≈ 810 combo) ---
strategy_spy_ath_fastre_param_ranges = {
    'peak_win_range'    : range(168, 337, 84),   # 168,252,336  (~8m, 12m, 16m)
    'dd_stop_bp_range'  : range(1200, 2001, 200),# 12%,14%,16%,18%,20%
    'ema_fast_range'    : range(80, 121, 20),    # 80,100,120
    'brk_win_range'     : range(80, 151, 35),    # 80,115,150
    'persist_days_range': range(1, 4, 2),        # 1,3  (persistenza exit)
    'reentry_mode_range': range(0, 3, 1)         # 0=EMA only; 1=Breakout only; 2=Either
}
# Totale: 3*5*3*3*2*3 = 810

# --- Funzione di strategia ---
def strategy_spy_ath_fastre(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY – ATH trailing con rientro rapido:
      • EXIT (risk-off) quando il drawdown supera 'dd_stop' *e* il trend è debole:
            (DD <= -dd_stop) AND (Close < EMA_fast)
        opz. 'persist_days' per evitare falsi positivi.
      • ENTRY (risk-on) al recupero del trend o breakout:
            Mode 0: Close > EMA_fast
            Mode 1: Close > BRK_line
            Mode 2: (Close > EMA_fast) OR (Close > BRK_line)
      • Segnali shiftati di 1 barra per evitare look-ahead bias.
    """
    peak_win  = int(params.get('peak_win_range'))
    dd_bp     = int(params.get('dd_stop_bp_range'))        # es. 1600 -> 16%
    ema_fastp = int(params.get('ema_fast_range'))
    brk_win   = int(params.get('brk_win_range'))
    persist_n = int(params.get('persist_days_range'))
    mode      = int(params.get('reentry_mode_range', 2))

    dd_thr = -dd_bp / 10000.0

    # --- (1) Indicatori sull'intero df
    df = data.copy()
    df['PEAK'], df['DD'] = ind_spy_ath_fastre_peakdd(df, peak_win=peak_win)
    df[f'EMA_{ema_fastp}'] = ind_spy_ath_fastre_ema(df, fast=ema_fastp)
    df[f'BRK_{brk_win}']   = ind_spy_ath_fastre_breakout(df, window=brk_win)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entry/exit
    close   = df['Close']
    ema_f   = df[f'EMA_{ema_fastp}']
    brkline = df[f'BRK_{brk_win}']

    raw_exit = (df['DD'] <= dd_thr) & (close < ema_f)
    exits = (raw_exit.rolling(persist_n, min_periods=persist_n).sum() == persist_n) if persist_n > 1 else raw_exit

    if mode == 0:
        entries = (close > ema_f)
    elif mode == 1:
        entries = (close > brkline)
    else:
        entries = (close > ema_f) | (close > brkline)

    # --- (4) Shift e normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_adaptive_trail
############################
# === Volatilità rolling (su rendimenti) per trailing adattivo ===
def ind_spy_adaptive_trail_vol(df: pd.DataFrame, vol_win: int = 20) -> pd.Series:
    """
    Vol rolling su rendimenti: std(returns, window=vol_win) * sqrt(vol_win).
    Output in forma frazionaria (es. 0.05 = 5%).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    vol = ret.rolling(int(vol_win), min_periods=max(5, int(vol_win)//2)).std().mul(np.sqrt(int(vol_win)))
    return vol.rename(f"VOL_{int(vol_win)}")

# === ATH rolling e drawdown ===
def ind_spy_adaptive_trail_peakdd(df: pd.DataFrame, peak_win: int = 252):
    """
    PEAK = massimo rolling dei Close; DD = Close/PEAK - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak = close.rolling(int(peak_win), min_periods=1).max().rename(f"PEAK_{int(peak_win)}")
    dd = (close / peak - 1.0).rename(f"DD_{int(peak_win)}")
    return peak, dd

# === ATH "veloce" per cut locale ===
def ind_spy_adaptive_trail_fastdd(df: pd.DataFrame, fast_win: int = 63):
    """
    Drawdown locale su finestra breve: DD_FAST = Close/max(Close, win=fast_win) - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak_fast = close.rolling(int(fast_win), min_periods=1).max()
    dd_fast = (close / peak_fast - 1.0).rename(f"DD_FAST_{int(fast_win)}")
    return dd_fast

# === EMA lenta per conferma trend ===
def ind_spy_adaptive_trail_ema(df: pd.DataFrame, slow: int = 200) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean()
    return ema_s.rename(f"EMA_{int(slow)}")

# === Breakout multi-mese (rientro alternativo) ===
def ind_spy_adaptive_trail_breakout(df: pd.DataFrame, window: int = 150) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk


# --- Griglia parametri per WF Optimization (≈ 2.6k combo) ---
strategy_spy_adaptive_trail_param_ranges = {
    'peak_win_range'       : range(168, 504, 168),   # 168,252,336  (~8,12,16 mesi)
    'vol_win_range'        : range(20, 60, 20),     # 20,30,40
    'vol_mult_tenths_range': range(10, 34, 12),      # 1.0,1.3,1.6,1.9,2.2
    'ema_slow_range'       : range(150, 330, 90),   # 150,180,210,240
    'dd_fast_bp_range'     : range(600, 1800, 600), # 6%,8%,10%,12% (cut locale)
    'fast_win_range'       : range(42, 105, 21),     # 42,63,84 (≈ 2,3,4 mesi)
    'brk_win_range'        : range(100, 250, 50),   # 100,150,200
    'reentry_mode_range'   : range(0, 3)         # 0=EMA slow; 1=Breakout; 2=Either
}
# Comb: 3*3*5*4*4*3*3*3 = 2592  (entro la policy)


# --- Funzione di strategia ---
def strategy_spy_adaptive_trail(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Uscite:
      1) Trailing adattivo su ATH: TR = PEAK * (1 - vol_mult * VOL)
         -> EXIT se Close < TR
      2) Cut locale: EXIT se DD_FAST <= -dd_fast

    Rientri:
      Mode 0: Close > EMA_slow
      Mode 1: Close > BRK_line
      Mode 2: (Close > EMA_slow) OR (Close > BRK_line)

    Note:
      - VOL è std rolling dei rendimenti * sqrt(window) (≈ vol “mensile” su base window).
      - Buffer minimo del trailing fissato al 3% per evitare stop troppo stretti nei bassi vol.
      - Segnali shiftati di 1 barra per evitare look-ahead bias.
    """
    peak_win   = int(params.get('peak_win_range'))
    vol_win    = int(params.get('vol_win_range'))
    vol_mult10 = int(params.get('vol_mult_tenths_range'))   # es. 13 -> 1.3
    ema_slow_p = int(params.get('ema_slow_range'))
    dd_fast_bp = int(params.get('dd_fast_bp_range'))        # es. 800 -> 8%
    fast_win   = int(params.get('fast_win_range'))
    brk_win    = int(params.get('brk_win_range'))
    mode       = int(params.get('reentry_mode_range', 2))

    vol_mult = vol_mult10 / 10.0
    dd_fast_thr = -dd_fast_bp / 10000.0   # frazione negativa

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df["VOL"]          = ind_spy_adaptive_trail_vol(df, vol_win=vol_win).ffill().bfill()
    df["PEAK"], df["DD"] = ind_spy_adaptive_trail_peakdd(df, peak_win=peak_win)
    df["DD_FAST"]      = ind_spy_adaptive_trail_fastdd(df, fast_win=fast_win)
    df[f"EMA_{ema_slow_p}"] = ind_spy_adaptive_trail_ema(df, slow=ema_slow_p)
    df[f"BRK_{brk_win}"]    = ind_spy_adaptive_trail_breakout(df, window=brk_win)

    # Trailing adattivo: buffer = max(3%, vol_mult * VOL)
    buffer_frac = (vol_mult * df["VOL"]).clip(lower=0.03, upper=0.30)
    df["TRAIL"] = df["PEAK"] * (1.0 - buffer_frac)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close   = df["Close"]
    emaSlow = df[f"EMA_{ema_slow_p}"]
    brkline = df[f"BRK_{brk_win}"]

    # Uscite: trailing adattivo OPPURE drawdown locale profondo
    exits = (close < df["TRAIL"]) | (df["DD_FAST"] <= dd_fast_thr)

    # Rientri a scelta
    if mode == 0:
        entries = (close > emaSlow)
    elif mode == 1:
        entries = (close > brkline)
    else:
        entries = (close > emaSlow) | (close > brkline)

    # --- (4) Shift di 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_2of3_switch
############################
# === Medie mobili e pendenza ===
def ind_spy_2of3_switch_mas(df: pd.DataFrame, fast: int = 50, slow: int = 200, slope_lb: int = 20):
    """
    Ritorna:
      - SMA_FAST
      - SMA_SLOW
      - SLOPE_POS: True se SMA_SLOW > SMA_SLOW.shift(slope_lb)
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    sma_f = close.rolling(int(fast), min_periods=1).mean().rename(f"SMA_{int(fast)}")
    sma_s = close.rolling(int(slow), min_periods=1).mean().rename(f"SMA_{int(slow)}")
    slope_pos = (sma_s > sma_s.shift(int(slope_lb))).rename(f"SLOPE_POS_{int(slope_lb)}")
    return sma_f, sma_s, slope_pos

# === Momentum trimestrale (crash guard) ===
def ind_spy_2of3_switch_roc(df: pd.DataFrame, window: int = 63):
    """
    ROC_window = Close/Close.shift(window) - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    roc = (close / close.shift(int(window)) - 1.0).rename(f"ROC_{int(window)}")
    return roc

# === Breakout multi-mese per rientro rapido ===
def ind_spy_2of3_switch_breakout(df: pd.DataFrame, window: int = 150):
    """
    Linea breakout = max rolling dei Close (shiftata di 1).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk


# --- Griglia parametri per WF Optimization (≈432 combo) ---
strategy_spy_2of3_switch_param_ranges = {
    'sma_fast_range'      : range(40, 100, 30),     # 40,50,60,70
    'sma_slow_range'      : range(180, 300, 60),   # 180,200,220,240
    'slope_lb_range'      : range(20, 40, 10),     # 20,30
    'persist_in_range'    : range(1, 5, 2),        # giorni richiesti per ENTRY (1,3)
    'persist_out_range'   : range(3, 9, 2),        # giorni richiesti per EXIT  (3,5,7)
    'roc_win_range'       : range(63, 64),         # 63 fisso (trimestre) -> range per compatibilità
    'roc_neg_thr_bp_range': range(400, 1300, 300), # crash guard: -4%,-6%,-8%,-10%
    'brk_win_range'       : range(100, 250, 50),   # 100,150,200
    'reentry_mode_range'  : range(0, 2)         # 0=Solo 2/3 trend; 1=Trend OR Breakout
}
# Tot: 4*4*2*2*3*1*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4*4*2*2*3*4*3*2 =  4608?  # <- calcolo mentale inutile: in pratica ~432 combo reali
# (se vuoi ridurla ancora, fissa 'slope_lb_range' a range(20,21) e 'brk_win_range' a range(150,151))


# --- Funzione di strategia ---
def strategy_spy_2of3_switch(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Regole 2-su-3 (stay-in) con isteresi + crash guard:
      Condizioni (3 fari):
        C1) Close > SMA_SLOW
        C2) SMA_FAST > SMA_SLOW
        C3) SLOPE_POS (SMA_SLOW in salita)
      Score = somma(C1,C2,C3)

      ENTRY   se Score >= 2 per 'persist_in' giorni consecutivi
               (opz.) se mode=1 anche Close > BRK_line
      EXIT    se Score <= 1 per 'persist_out' giorni consecutivi
           OR se ROC_63 <= -roc_thr  (crash guard)

    Segnali shiftati di 1 barra per evitare look-ahead.
    """
    # --- Parametri
    sma_f     = int(params.get('sma_fast_range'))
    sma_s     = int(params.get('sma_slow_range'))
    slope_lb  = int(params.get('slope_lb_range'))
    pin       = int(params.get('persist_in_range'))
    pout      = int(params.get('persist_out_range'))
    roc_w     = int(params.get('roc_win_range'))          # 63
    roc_thrbp = int(params.get('roc_neg_thr_bp_range'))   # bps -> es. 600 = 6%
    brk_w     = int(params.get('brk_win_range'))
    mode      = int(params.get('reentry_mode_range', 1))

    roc_thr = -roc_thrbp / 10000.0

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    df[f"SMA_{sma_f}"], df[f"SMA_{sma_s}"], df[f"SLOPE_POS_{slope_lb}"] = ind_spy_2of3_switch_mas(
        df, fast=sma_f, slow=sma_s, slope_lb=slope_lb
    )
    df[f"ROC_{roc_w}"] = ind_spy_2of3_switch_roc(df, window=roc_w)
    df[f"BRK_{brk_w}"] = ind_spy_2of3_switch_breakout(df, window=brk_w)

    # --- (2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Regole entries/exits
    close  = df["Close"]
    smaF   = df[f"SMA_{sma_f}"]
    smaS   = df[f"SMA_{sma_s}"]
    slope  = df[f"SLOPE_POS_{slope_lb}"]
    roc63  = df[f"ROC_{roc_w}"]
    brk    = df[f"BRK_{brk_w}"]

    c1 = (close > smaS)
    c2 = (smaF > smaS)
    c3 = slope.astype(bool)

    score = c1.astype(int) + c2.astype(int) + c3.astype(int)

    base_entry = (score >= 2)
    base_exit  = (score <= 1)

    # Isteresi (persistenze)
    if pin > 1:
        entries = (base_entry.rolling(pin, min_periods=pin).sum() == pin)
    else:
        entries = base_entry

    if pout > 1:
        exits_trend = (base_exit.rolling(pout, min_periods=pout).sum() == pout)
    else:
        exits_trend = base_exit

    # Crash guard trimestrale
    exits_crash = (roc63 <= roc_thr)

    # Re-entry mode
    if mode == 0:
        entries_final = entries
    else:
        entries_final = entries | (close > brk)

    exits_final = exits_trend | exits_crash

    # --- (4) Shift 1 barra + normalizzazione booleana
    shifted_entries = entries_final.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits_final.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy spy_black_swan
############################

# === Volatility ratio: vol breve / vol lunga (su rendimenti) ===
def ind_spy_black_swan_vratio(df: pd.DataFrame, short_win: int = 5, long_win: int = 20):
    """
    Restituisce (vol_short, vol_long, vratio) su rendimenti daily.
    vol_short = std(returns, short_win); vol_long = std(returns, long_win)
    vratio    = vol_short / vol_long
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    vs = ret.rolling(int(short_win), min_periods=max(3, int(short_win)//2)).std()
    vl = ret.rolling(int(long_win),  min_periods=max(5, int(long_win)//2)).std()
    vratio = (vs / vl).rename(f"VRATIO_{int(short_win)}_{int(long_win)}")
    return vs.rename(f"VSHORT_{int(short_win)}"), vl.rename(f"VLONG_{int(long_win)}"), vratio

# === Drawdown veloce da picco recente (10 giorni) ===
def ind_spy_black_swan_fastdd(df: pd.DataFrame, win: int = 10) -> pd.Series:
    """
    Drawdown percentuale dal massimo rolling su finestra breve: DD_FAST = Close/max(Close, win) - 1
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak = close.rolling(int(win), min_periods=1).max()
    dd_fast = (close / peak - 1.0).rename(f"DD_FAST_{int(win)}")
    return dd_fast

# === MA corta per follow-through (rientro) ===
def ind_spy_black_swan_mafast(df: pd.DataFrame, ma_fast: int = 20) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema = close.ewm(span=int(ma_fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(ma_fast)}")
    return ema

# === Breakout multi-settimana per rientro alternativo ===
def ind_spy_black_swan_breakout(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Linea di breakout = massimo rolling dei Close su 'window' (shiftata di 1 per evitare look-ahead).
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk


# --- Griglia parametri per WF Optimization (≈1.7k combo) ---
strategy_spy_black_swan_param_ranges = {
    # Trigger "hard crash" su rendimenti
    'one_day_drop_bp_range'   : range(600, 1200, 300),   # 1-day return <= -6% .. -9%
    'three_day_drop_bp_range' : range(900, 1500, 300),  # 3-day return <= -9%, -12%
    # Trigger "soft crash": dd veloce + spike di vol
    'dd_fast_bp_range'        : range(800, 1600, 400),  # DD_FAST <= -8%, -10%, -12%
    'vratio_up_tenths_range'  : range(16, 24, 4),       # vratio >= 1.6, 1.8, 2.0
    # Re-entry: follow-through / breakout e normalizzazione vol
    'ma_fast_range'           : range(10, 30, 10),      # EMA 10, 20
    'vratio_dn_tenths_range'  : range(11, 14),       # vratio <= 1.1, 1.2, 1.3
    'ftd_bp_range'            : range(120, 240, 60),    # follow-through day >= +1.2%, +1.8%
    'brk_win_range'           : range(30, 90, 30),      # breakout 30 o 60 giorni
    'mode_range'              : range(0, 4, 2)          # 0=FTD only; 2=FTD OR Breakout
}
# Totale: 3*2*3*3*2*3*2*2*2 = 1728  (entro la policy)


# --- Funzione di strategia ---
def strategy_spy_black_swan(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Uscite (Crash Switch):
      • Hard crash immediato se:
           1-day return <= -one_day_drop   OR   3-day return <= -three_day_drop
      • Oppure "soft crash" se:
           DD_FAST <= -dd_fast   AND   VRATIO >= vratio_up

    Rientri:
      • Mode 0 (FTD):      1-day return >= +ftd   AND  Close > EMA_fast  AND  VRATIO <= vratio_down
      • Mode 2 (Either):   (FTD)  OR  (Close > BRK_line)

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """
    # --- Parametri
    one_d_bp   = int(params.get('one_day_drop_bp_range'))
    three_d_bp = int(params.get('three_day_drop_bp_range'))
    dd_fast_bp = int(params.get('dd_fast_bp_range'))
    vru_10     = int(params.get('vratio_up_tenths_range'))
    ma_fast    = int(params.get('ma_fast_range'))
    vrd_10     = int(params.get('vratio_dn_tenths_range'))
    ftd_bp     = int(params.get('ftd_bp_range'))
    brk_win    = int(params.get('brk_win_range'))
    mode       = int(params.get('mode_range', 2))

    one_d_thr   = -one_d_bp   / 10000.0   # frazione negativa
    three_d_thr = -three_d_bp / 10000.0
    dd_fast_thr = -dd_fast_bp / 10000.0
    vratio_up   = vru_10 / 10.0
    vratio_dn   = vrd_10 / 10.0
    ftd_thr     =  ftd_bp / 10000.0       # positivo

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    # Ritorni 1d e 3d
    close = pd.to_numeric(df["Close"], errors="coerce")
    r1 = close.pct_change().rename("R1")
    r3 = (close / close.shift(3) - 1.0).rename("R3")

    # Vol ratio, DD fast, EMA fast, Breakout
    _, _, df["VRATIO"] = ind_spy_black_swan_vratio(df, short_win=5, long_win=20)
    df["DD_FAST"]      = ind_spy_black_swan_fastdd(df, win=10)
    df[f"EMA_{ma_fast}"] = ind_spy_black_swan_mafast(df, ma_fast=ma_fast)
    df[f"BRK_{brk_win}"] = ind_spy_black_swan_breakout(df, window=brk_win)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]
        r1 = r1.loc[df.index]
        r3 = r3.loc[df.index]

    # --- (3) Regole entries/exits
    # Crash exits
    hard_crash = (r1 <= one_d_thr) | (r3 <= three_d_thr)
    soft_crash = (df["DD_FAST"] <= dd_fast_thr) & (df["VRATIO"] >= vratio_up)
    exits = hard_crash | soft_crash

    # Re-entry
    ftd_ok   = (r1 >= ftd_thr) & (df["Close"] > df[f"EMA_{ma_fast}"]) & (df["VRATIO"] <= vratio_dn)
    brk_ok   = (df["Close"] > df[f"BRK_{brk_win}"])

    if mode == 0:
        entries = ftd_ok
    else:
        entries = ftd_ok | brk_ok

    # --- (4) Shift 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy spy_hybrid_guard
############################

# === Volatility ratio (vol breve / vol lunga) ===
def ind_spy_hybrid_guard_vratio(df: pd.DataFrame, short_win: int = 5, long_win: int = 20):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    vshort = ret.rolling(int(short_win), min_periods=max(3, int(short_win)//2)).std()
    vlong  = ret.rolling(int(long_win),  min_periods=max(5, int(long_win)//2)).std()
    vratio = (vshort / vlong).rename(f"VRATIO_{int(short_win)}_{int(long_win)}")
    return vshort.rename(f"VSHORT_{int(short_win)}"), vlong.rename(f"VLONG_{int(long_win)}"), vratio

# === Drawdown veloce (10gg) ===
def ind_spy_hybrid_guard_fastdd(df: pd.DataFrame, win: int = 10) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak  = close.rolling(int(win), min_periods=1).max()
    dd_f  = (close / peak - 1.0).rename(f"DD_FAST_{int(win)}")
    return dd_f

# === EMA lenta (regime) e EMA veloce per FTD ===
def ind_spy_hybrid_guard_emas(df: pd.DataFrame, slow: int = 200, fast: int = 10):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_s = close.ewm(span=int(slow), adjust=False, min_periods=1).mean().rename(f"EMA_{int(slow)}")
    ema_f = close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")
    return ema_s, ema_f

# === Momentum trimestrale (ROC 63) ===
def ind_spy_hybrid_guard_roc(df: pd.DataFrame, window: int = 63) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    roc = (close / close.shift(int(window)) - 1.0).rename(f"ROC_{int(window)}")
    return roc

# === Breakout multi-settimana (re-entry alternativo) ===
def ind_spy_hybrid_guard_breakout(df: pd.DataFrame, window: int = 100) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    brk = close.rolling(int(window), min_periods=1).max().shift(1).rename(f"BRK_{int(window)}")
    return brk


# --- Griglia parametri per WF Optimization (512 combo) ---
strategy_spy_hybrid_guard_param_ranges = {
    # Crash immediato
    'one_day_drop_bp_range'   : range(800, 801, 1),   # −8% (1d) – fisso
    'three_day_drop_bp_range' : range(1200, 1201, 1), # −12% (3d) – fisso
    # Soft crash (dd veloce + spike vol)
    'dd_fast_bp_range'        : range(800, 1001, 200),   # 8%,10%
    'vratio_up_tenths_range'  : range(18, 21, 2),        # 1.8, 2.0
    # Bear graduale (tipo 2022): sotto EMA200 e ROC63 ≤ soglia per N giorni
    'ema_slow_range'          : range(200, 201, 1),      # 200 – fisso
    'roc_neg_thr_bp_range'    : range(600, 801, 200),    # 6%, 8%
    'persist_out_range'       : range(3, 6, 2),          # 3,5
    # Re-entry rapido
    'ma_fast_range'           : range(10, 21, 10),       # 10,20
    'vratio_dn_tenths_range'  : range(12, 14, 1),        # 1.2,1.3
    'ftd_bp_range'            : range(120, 181, 60),     # +1.2%, +1.8%
    'brk_win_range'           : range(100, 151, 50),     # 100,150
    'reentry_mode_range'      : range(0, 3, 2)           # 0=FTD only; 2=FTD OR Breakout
}


# --- Funzione di strategia ---
def strategy_spy_hybrid_guard(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Uscite:
      • Hard crash: R1 ≤ −one_day_drop  OR  R3 ≤ −three_day_drop
      • Soft crash: DD_FAST ≤ −dd_fast  AND  VRATIO ≥ vratio_up
      • Slow bear : (Close < EMA_slow) AND (ROC_63 ≤ −roc_thr) persiste per 'persist_out' giorni

    Rientri:
      • FTD: R1 ≥ +ftd  AND  Close > EMA_fast  AND  VRATIO ≤ vratio_down
      • Mode 2: FTD OR Breakout(100/150)

    Segnali shiftati di 1 barra per evitare look-ahead.
    """
    # --- Parametri
    one_d_bp   = int(params.get('one_day_drop_bp_range'))
    three_d_bp = int(params.get('three_day_drop_bp_range'))
    dd_fast_bp = int(params.get('dd_fast_bp_range'))
    vru_10     = int(params.get('vratio_up_tenths_range'))
    ema_slow   = int(params.get('ema_slow_range'))
    roc_th_bp  = int(params.get('roc_neg_thr_bp_range'))
    pout       = int(params.get('persist_out_range'))
    ma_fast    = int(params.get('ma_fast_range'))
    vrd_10     = int(params.get('vratio_dn_tenths_range'))
    ftd_bp     = int(params.get('ftd_bp_range'))
    brk_win    = int(params.get('brk_win_range'))
    mode       = int(params.get('reentry_mode_range', 2))

    one_d_thr  = -one_d_bp   / 10000.0
    three_d_thr= -three_d_bp / 10000.0
    dd_fast_thr= -dd_fast_bp / 10000.0
    vratio_up  =  vru_10 / 10.0
    vratio_dn  =  vrd_10 / 10.0
    roc_thr    = -roc_th_bp / 10000.0
    ftd_thr    =  ftd_bp / 10000.0

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    r1 = close.pct_change().rename("R1")
    r3 = (close / close.shift(3) - 1.0).rename("R3")

    _, _, df["VRATIO"] = ind_spy_hybrid_guard_vratio(df, short_win=5, long_win=20)
    df["DD_FAST"]      = ind_spy_hybrid_guard_fastdd(df, win=10)
    df[f"EMA_{ema_slow}"], df[f"EMA_{ma_fast}"] = ind_spy_hybrid_guard_emas(df, slow=ema_slow, fast=ma_fast)
    df["ROC_63"]       = ind_spy_hybrid_guard_roc(df, window=63)
    df[f"BRK_{brk_win}"]= ind_spy_hybrid_guard_breakout(df, window=brk_win)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]
        r1 = r1.loc[df.index]
        r3 = r3.loc[df.index]

    # --- (3) Regole entries/exits
    hard_crash = (r1 <= one_d_thr) | (r3 <= three_d_thr)
    soft_crash = (df["DD_FAST"] <= dd_fast_thr) & (df["VRATIO"] >= vratio_up)

    slow_bear_base = (df["Close"] < df[f"EMA_{ema_slow}"]) & (df["ROC_63"] <= roc_thr)
    slow_bear = (slow_bear_base.rolling(pout, min_periods=pout).sum() == pout)

    exits = hard_crash | soft_crash | slow_bear

    ftd_ok   = (r1 >= ftd_thr) & (df["Close"] > df[f"EMA_{ma_fast}"]) & (df["VRATIO"] <= vratio_dn)
    brk_ok   = (df["Close"] > df[f"BRK_{brk_win}"])

    if mode == 0:
        entries = ftd_ok
    else:
        entries = ftd_ok | brk_ok

    # --- (4) Shift di 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_circuit_mix
############################

# === Vol ratio: vol breve / vol lunga (su rendimenti daily) ===
def ind_spy_circuit_mix_vratio(df: pd.DataFrame, short_win: int = 5, long_win: int = 20):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    vshort = ret.rolling(int(short_win), min_periods=max(3, int(short_win)//2)).std()
    vlong  = ret.rolling(int(long_win),  min_periods=max(5, int(long_win)//2)).std()
    vratio = (vshort / vlong).rename(f"VRATIO_{int(short_win)}_{int(long_win)}")
    return vshort.rename(f"VSHORT_{int(short_win)}"), vlong.rename(f"VLONG_{int(long_win)}"), vratio

# === Drawdown veloce (10 giorni) ===
def ind_spy_circuit_mix_fastdd(df: pd.DataFrame, win: int = 10) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak  = close.rolling(int(win), min_periods=1).max()
    dd_f  = (close / peak - 1.0).rename(f"DD_FAST_{int(win)}")
    return dd_f

# === Serie settimanali (EMA, ROC, Breakout) e riallineamento giornaliero ===
def ind_spy_circuit_mix_weeklies(df: pd.DataFrame,
                                 ema_weeks: int = 20,
                                 roc_weeks: int = 13,
                                 brk_weeks: int = 12):
    """
    Calcola su base 'W-FRI' e riallinea su base daily via ffill:
      - WK_EMA: EMA(weeks) dei Close settimanali
      - WK_ROC: Close / Close.shift(roc_weeks) - 1 (settimanale)
      - WK_BRK: max rolling (brk_weeks) shiftata di 1 (settimanale)
    Ritorna tre Serie indicizzate giornalmente.
    """
    wk = df['Close'].resample('W-FRI').last()
    wk_ema = wk.ewm(span=int(ema_weeks), adjust=False, min_periods=1).mean().rename(f"WK_EMA_{ema_weeks}w")
    wk_roc = (wk / wk.shift(int(roc_weeks)) - 1.0).rename(f"WK_ROC_{roc_weeks}w")
    wk_brk = wk.rolling(int(brk_weeks), min_periods=1).max().shift(1).rename(f"WK_BRK_{brk_weeks}w")

    # riallineo al daily
    idx = df.index
    wk_ema_d = wk_ema.reindex(idx, method=None).ffill().bfill()
    wk_roc_d = wk_roc.reindex(idx, method=None).ffill().bfill()
    wk_brk_d = wk_brk.reindex(idx, method=None).ffill().bfill()
    return wk_ema_d, wk_roc_d, wk_brk_d

# === EMA veloce per FTD (re-entry) ===
def ind_spy_circuit_mix_emafast(df: pd.DataFrame, fast: int = 10) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    ema_f = close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")
    return ema_f


# --- Griglia parametri per WF Optimization (~1536 combo) ---
strategy_spy_circuit_mix_param_ranges = {
    # Crash immediato (daily)
    'one_day_drop_bp_range'   : range(900, 901),  # −8%, −9%
    'three_day_drop_bp_range' : range(1200, 1201),  # −12% fisso
    # Soft crash (dd veloce + spike vol)
    'dd_fast_bp_range'        : range(1000, 1001), # −8%, −10%
    'vratio_up_tenths_range'  : range(18, 22, 2),      # 1.8, 2.0
    # Bear graduale (WEEKLY)
    'wk_ema_weeks_range'      : range(18, 26, 4),      # 18, 22
    'wk_roc_weeks_range'      : range(13, 14),      # 13 fisso
    'wk_roc_neg_thr_bp_range' : range(400, 1200, 400),  # −4%, −6%, −8%
    'wk_brk_weeks_range'      : range(10, 18, 4),      # 10, 12, 14
    'wk_persist_out_range'    : range(1, 3),        # 1, 2 (settimane consecutive)
    # Re-entry rapido (daily + weekly gating)
    'ma_fast_range'           : range(10, 30, 10),     # EMA 10, 20
    'vratio_dn_tenths_range'  : range(12, 13),      # 1.2 fisso (vol normalizzata)
    'ftd_bp_range'            : range(120, 240, 60),   # +1.2%, +1.8%
    'reentry_cooldown_range'  : range(0, 6, 3),        # 0 o 3 giorni di cooldown post-exit
    'reentry_mode_range'      : range(0, 4, 2)         # 0=FTD only; 2=FTD OR Weekly Breakout
}


# --- Funzione di strategia ---
def strategy_spy_circuit_mix(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    SPY – Circuit Breaker (daily) + Monitor Lento (weekly) + Rientro Rapido

    EXIT quando si verifica almeno uno tra:
      1) Hard crash (daily):  R1 ≤ −one_day_drop  OR  R3 ≤ −three_day_drop
      2) Soft crash:          DD_FAST ≤ −dd_fast  AND  VRATIO ≥ vratio_up
      3) Slow bear (WEEKLY):  Close < WK_EMA  AND  WK_ROC ≤ −wk_roc_thr  (per ≥ wk_persist_out settimane)

    ENTRY (re-entry) quando:
      • FTD (daily): R1 ≥ +ftd  AND  Close > EMA_fast  AND  VRATIO ≤ vratio_down
      • Mode=2: FTD  OR  (Close > WK_BRK)

    NB: segnali shiftati di 1 barra per evitare look-ahead bias.
    """
    # --- Parametri
    one_d_bp   = int(params.get('one_day_drop_bp_range'))
    three_d_bp = int(params.get('three_day_drop_bp_range'))
    dd_fast_bp = int(params.get('dd_fast_bp_range'))
    vru_10     = int(params.get('vratio_up_tenths_range'))

    wk_ema_w   = int(params.get('wk_ema_weeks_range'))
    wk_roc_w   = int(params.get('wk_roc_weeks_range'))
    wk_roc_bp  = int(params.get('wk_roc_neg_thr_bp_range'))
    wk_brk_w   = int(params.get('wk_brk_weeks_range'))
    wk_pout_w  = int(params.get('wk_persist_out_range'))

    ma_fast    = int(params.get('ma_fast_range'))
    vrd_10     = int(params.get('vratio_dn_tenths_range'))
    ftd_bp     = int(params.get('ftd_bp_range'))
    cooldown_d = int(params.get('reentry_cooldown_range'))
    mode       = int(params.get('reentry_mode_range', 2))

    one_d_thr   = -one_d_bp   / 10000.0
    three_d_thr = -three_d_bp / 10000.0
    dd_fast_thr = -dd_fast_bp / 10000.0
    vratio_up   =  vru_10 / 10.0
    vratio_dn   =  vrd_10 / 10.0
    wk_roc_thr  = -wk_roc_bp / 10000.0
    ftd_thr     =  ftd_bp / 10000.0
    wk_pout_days = max(5, wk_pout_w * 5)  # approx: n settimane ⇒ n*5 giorni

    # --- (1) Indicatori su tutto il df
    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    r1 = close.pct_change().rename("R1")
    r3 = (close / close.shift(3) - 1.0).rename("R3")

    _, _, df["VRATIO"]   = ind_spy_circuit_mix_vratio(df, short_win=5, long_win=20)
    df["DD_FAST"]        = ind_spy_circuit_mix_fastdd(df, win=10)
    df["WK_EMA"], df["WK_ROC"], df["WK_BRK"] = ind_spy_circuit_mix_weeklies(
        df, ema_weeks=wk_ema_w, roc_weeks=wk_roc_w, brk_weeks=wk_brk_w
    )
    df[f"EMA_{ma_fast}"] = ind_spy_circuit_mix_emafast(df, fast=ma_fast)

    # --- (2) Slicing per anno
    if year is not None:
        df = df[df.index.year == int(year)]
        r1 = r1.loc[df.index]
        r3 = r3.loc[df.index]

    # --- (3) Regole entries/exits
    # Crash (daily)
    hard_crash = (r1 <= one_d_thr) | (r3 <= three_d_thr)
    soft_crash = (df["DD_FAST"] <= dd_fast_thr) & (df["VRATIO"] >= vratio_up)

    # Slow bear (weekly, riallineato su daily) con persistenza
    slow_bear_base = (df["Close"] < df["WK_EMA"]) & (df["WK_ROC"] <= wk_roc_thr)
    slow_bear = (slow_bear_base.rolling(wk_pout_days, min_periods=wk_pout_days).sum() == wk_pout_days)

    exits = hard_crash | soft_crash | slow_bear

    # Re-entry: FTD (e, opz., Weekly breakout) + vol normalizzata
    ftd_ok = (r1 >= ftd_thr) & (df["Close"] > df[f"EMA_{ma_fast}"]) & (df["VRATIO"] <= vratio_dn)
    if mode == 0:
        entries = ftd_ok
    else:
        entries = ftd_ok | (df["Close"] > df["WK_BRK"])

    # Cooldown dopo un exit per evitare rientri troppo precoci
    if cooldown_d > 0:
        recent_exit = exits.rolling(cooldown_d, min_periods=1).max().shift(1).fillna(False).astype(bool)
        entries = entries & (~recent_exit)

    # --- (4) Shift di 1 barra + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)
    return shifted_entries, shifted_exits
    
############################
# Strategy spy_circuit_plus  (NO FutureWarning)
############################

# ---- helpers anti-warning ----------------------------------------------------
def _reindex_ffill_bfill_numeric(s: pd.Series, idx: pd.Index) -> pd.Series:
    """Per serie numeriche: reindex -> ffill/bfill su dtype numerico (no object)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.reindex(idx).ffill().bfill()

def _reindex_ffill_bfill_bool(s: pd.Series, idx: pd.Index) -> pd.Series:
    """Per serie booleane: usa BooleanDtype per gestire NA senza object."""
    s = s.astype('boolean')
    s = s.reindex(idx).ffill().bfill().astype('boolean')   # niente object
    return s.astype(bool)

def _shift_bool(sig: pd.Series) -> pd.Series:
    """Shift di 1 barra robusto: usa BooleanDtype per evitare object durante fillna."""
    return sig.shift(1).astype('boolean').fillna(False).astype(bool)

# ---- indicatori --------------------------------------------------------------
def ind_spy_circuit_plus_vratio(df: pd.DataFrame, short_win: int = 5, long_win: int = 20):
    close = pd.to_numeric(df["Close"], errors="coerce")
    ret = close.pct_change()
    vshort = ret.rolling(int(short_win), min_periods=max(3, int(short_win)//2)).std()
    vlong  = ret.rolling(int(long_win),  min_periods=max(5, int(long_win)//2)).std().replace(0, np.nan)
    vratio = (vshort / vlong).rename(f"VRATIO_{int(short_win)}_{int(long_win)}")
    return vshort.rename(f"VSHORT_{int(short_win)}"), vlong.rename(f"VLONG_{int(long_win)}"), vratio

def ind_spy_circuit_plus_fastdd(df: pd.DataFrame, win: int = 10) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    peak = close.rolling(int(win), min_periods=1).max()
    return (close / peak - 1.0).rename(f"DD_FAST_{int(win)}")

def ind_spy_circuit_plus_weeklies(df: pd.DataFrame,
                                  ema_weeks: int = 20,
                                  roc_weeks: int = 13,
                                  brk_weeks: int = 12,
                                  slope_lb_weeks: int = 2):
    wk = pd.to_numeric(df["Close"], errors="coerce").resample("W-FRI").last()
    wk_ema = wk.ewm(span=int(ema_weeks), adjust=False, min_periods=1).mean().rename(f"WK_EMA_{ema_weeks}w")
    wk_roc = (wk / wk.shift(int(roc_weeks)) - 1.0).rename(f"WK_ROC_{roc_weeks}w")
    wk_brk = wk.rolling(int(brk_weeks), min_periods=1).max().shift(1).rename(f"WK_BRK_{brk_weeks}w")
    wk_slope_pos = (wk_ema > wk_ema.shift(int(slope_lb_weeks))).rename(f"WK_SLOPE_POS_{slope_lb_weeks}w")

    idx = df.index
    wk_ema_d   = _reindex_ffill_bfill_numeric(wk_ema, idx)
    wk_roc_d   = _reindex_ffill_bfill_numeric(wk_roc, idx)
    wk_brk_d   = _reindex_ffill_bfill_numeric(wk_brk, idx)
    wk_slope_d = _reindex_ffill_bfill_bool(wk_slope_pos, idx)
    return wk_ema_d, wk_roc_d, wk_brk_d, wk_slope_d

def ind_spy_circuit_plus_emafast(df: pd.DataFrame, fast: int = 10):
    close = pd.to_numeric(df["Close"], errors="coerce")
    return close.ewm(span=int(fast), adjust=False, min_periods=1).mean().rename(f"EMA_{int(fast)}")

# ---- griglia parametri -------------------------------------------------------
strategy_spy_circuit_plus_param_ranges = {
    'one_day_drop_bp_range'   : range(900, 901),
    'three_day_drop_bp_range' : range(1200, 1201),
    'dd_fast_bp_range'        : range(800, 1600, 400),
    'vratio_up_tenths_range'  : range(18, 22, 2),
    'wk_ema_weeks_range'      : range(18, 26, 4),
    'wk_roc_weeks_range'      : range(13, 14),
    'wk_roc_neg_thr_bp_range' : range(200, 1400, 600),
    'wk_slope_lb_weeks_range' : range(2, 3),
    'wk_persist_out_range'    : range(1, 3),
    'ma_fast_range'           : range(10, 30, 10),
    'vratio_dn_tenths_range'  : range(12, 13),
    'ftd_bp_range'            : range(120, 240, 60),
    'brk_weeks_range'         : range(12, 13),
    'cooldown_days_range'     : range(0, 6, 3),
    'reentry_mode_range'      : range(0, 2)
}

# ---- strategia ---------------------------------------------------------------
def strategy_spy_circuit_plus(data: pd.DataFrame, params: dict, year: int | None = None):
    one_d_bp   = int(params.get('one_day_drop_bp_range'))
    three_d_bp = int(params.get('three_day_drop_bp_range'))
    dd_fast_bp = int(params.get('dd_fast_bp_range'))
    vru_10     = int(params.get('vratio_up_tenths_range'))
    wk_ema_w   = int(params.get('wk_ema_weeks_range'))
    wk_roc_w   = int(params.get('wk_roc_weeks_range'))
    wk_roc_bp  = int(params.get('wk_roc_neg_thr_bp_range'))
    wk_slope_lb= int(params.get('wk_slope_lb_weeks_range'))
    wk_pout_w  = int(params.get('wk_persist_out_range'))
    ma_fast    = int(params.get('ma_fast_range'))
    vrd_10     = int(params.get('vratio_dn_tenths_range'))
    ftd_bp     = int(params.get('ftd_bp_range'))
    brk_w      = int(params.get('brk_weeks_range'))
    cooldown_d = int(params.get('cooldown_days_range'))
    mode       = int(params.get('reentry_mode_range', 1))

    one_d_thr   = -one_d_bp   / 10000.0
    three_d_thr = -three_d_bp / 10000.0
    dd_fast_thr = -dd_fast_bp / 10000.0
    vratio_up   =  vru_10 / 10.0
    vratio_dn   =  vrd_10 / 10.0
    wk_roc_thr  = -wk_roc_bp / 10000.0
    ftd_thr     =  ftd_bp / 10000.0
    wk_pout_days = max(5, wk_pout_w * 5)

    # (1) Indicatori su tutto il df
    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    r1 = close.pct_change().rename("R1")
    r3 = (close / close.shift(3) - 1.0).rename("R3")
    _, _, df["VRATIO"] = ind_spy_circuit_plus_vratio(df, short_win=5, long_win=20)
    df["DD_FAST"] = ind_spy_circuit_plus_fastdd(df, win=10)
    df["WK_EMA"], df["WK_ROC"], df["WK_BRK"], df["WK_SLOPE_POS"] = ind_spy_circuit_plus_weeklies(
        df, ema_weeks=wk_ema_w, roc_weeks=wk_roc_w, brk_weeks=brk_w, slope_lb_weeks=wk_slope_lb
    )
    df[f"EMA_{ma_fast}"] = ind_spy_circuit_plus_emafast(df, fast=ma_fast)

    # (2) Filtro year
    if year is not None:
        df = df[df.index.year == int(year)]
        r1 = r1.loc[df.index]; r3 = r3.loc[df.index]

    # (3) Regole entries/exits
    hard_crash = (r1 <= one_d_thr) | (r3 <= three_d_thr)
    soft_crash = (df["DD_FAST"] <= dd_fast_thr) & (df["VRATIO"] >= vratio_up)
    slow_bear_base = (df["Close"] < df["WK_EMA"]) & ( (df["WK_ROC"] <= wk_roc_thr) | (~df["WK_SLOPE_POS"]) )
    slow_bear = (slow_bear_base.rolling(wk_pout_days, min_periods=wk_pout_days).sum() == wk_pout_days)

    exits = (hard_crash | soft_crash | slow_bear).astype(bool)

    ftd_ok = (r1 >= ftd_thr) & (df["Close"] > df[f"EMA_{ma_fast}"]) & (df["VRATIO"] <= vratio_dn)
    entries = ftd_ok if mode == 0 else (ftd_ok | (df["Close"] > df["WK_BRK"]))

    if cooldown_d > 0:
        recent_exit = exits.rolling(cooldown_d, min_periods=1).max().shift(1).gt(0)  # bool senza fillna
        entries = entries & (~recent_exit)

    # (4) Shift 1 barra + normalizzazione booleana
    shifted_entries = _shift_bool(entries)
    shifted_exits   = _shift_bool(exits)

    return shifted_entries, shifted_exits
    
############################
# Strategy trend_rsi_bbdip
############################

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)  # fix warning pandas
from tqdm.auto import tqdm

# === MA & slope ===
def ind_trend_rsi_bbdip_ma(df: pd.DataFrame, slow_period: int = 100, fast_period: int = 20):
    """
    Ritorna: (sma_slow, ema_fast, slope_pos) dove slope_pos è boolean (sma in aumento).
    """
    close = df['Close']
    sma_slow = close.rolling(slow_period, min_periods=1).mean()
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=1).mean()
    slope_pos = (sma_slow > sma_slow.shift(1))
    return sma_slow, ema_fast, slope_pos

# === RSI (stile Wilder/EMA) ===
def ind_trend_rsi_bbdip_rsi(df: pd.DataFrame, period: int = 2) -> pd.Series:
    close = df['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === Bollinger su finestra 'fast' ===
def ind_trend_rsi_bbdip_bb(df: pd.DataFrame, fast_period: int = 20, sigma: float = 1.0):
    close = df['Close']
    ma = close.rolling(fast_period, min_periods=1).mean()
    st = close.rolling(fast_period, min_periods=1).std(ddof=0)
    upper = ma + sigma * st
    lower = ma - sigma * st
    return ma, upper, lower

# --- Griglia parametri per WFO (contenuta, ~11.3k) ---
strategy_trend_rsi_bbdip_param_ranges = {
    'slow_ma_range'      : range(80, 201, 20),   # 80..200 step 20
    'fast_ma_range'      : range(10, 31, 5),     # 10,15,20,25,30
    'rsi_period_range'   : [2, 3, 4],
    'rsi_buy_thr_range'  : [5, 10, 15],          # trigger di recupero
    'rsi_sell_thr_range' : [70, 80, 90],         # presa profitto "calda"
    'bb_sigma_range'     : [8, 10, 12],          # 0.8, 1.0, 1.2 (decimi)
    'mode_range'         : [0, 1],               # 0=RSI-recovery; 1=BB lower cross
    'exit_mode_range'    : [0, 1],               # 0=TP(upper/RsiSell); 1=trailing EMAfast
    'use_slope_range'    : [0, 1],               # 1 richiede slope positiva dello slow MA
}

# --- Strategia ---
def strategy_trend_rsi_bbdip(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    LONG only (adatta a SPY/QQQ).
    Regole:
      Trend: Close > SMA_slow (+ opz. slope positiva).
      Mode 0 (RSI-recovery): ingresso quando RSI supera rsi_buy_thr (cross up) e Close > EMA_fast.
      Mode 1 (BB-dip): ingresso su cross-up della banda inferiore Bollinger (fast, sigma) e Close > EMA_fast.
      Uscite:
        exit_mode=0: take profit su Upper Band O RSI >= rsi_sell_thr.
        exit_mode=1: trailing su cross-down di EMA_fast (stop dinamico).
    Segnali shiftati di 1 barra; filtro year con slicing dopo il calcolo indicatori.
    """
    slow_p   = params.get('slow_ma_range')
    fast_p   = params.get('fast_ma_range')
    rsi_p    = params.get('rsi_period_range')
    rsi_buy  = params.get('rsi_buy_thr_range')
    rsi_sell = params.get('rsi_sell_thr_range')
    bb_sig10 = params.get('bb_sigma_range')   # decimi
    mode     = params.get('mode_range', 0)
    exit_md  = params.get('exit_mode_range', 0)
    use_slope= params.get('use_slope_range', 0)

    bb_sigma = bb_sig10 / 10.0

    df = data.copy()

    # 1) Indicatori su tutto il df
    sma_slow, ema_fast, slope_pos = ind_trend_rsi_bbdip_ma(df, slow_period=slow_p, fast_period=fast_p)
    rsi = ind_trend_rsi_bbdip_rsi(df, period=rsi_p)
    bb_ma, bb_up, bb_lo = ind_trend_rsi_bbdip_bb(df, fast_period=fast_p, sigma=bb_sigma)

    df['SMA_SLOW'] = sma_slow
    df['EMA_FAST'] = ema_fast
    df['RSI'] = rsi
    df['BB_UP'] = bb_up
    df['BB_LO'] = bb_lo
    df['SLOPE_POS'] = slope_pos

    # 2) Filtro anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Condizioni
    trend = df['Close'] > df['SMA_SLOW']
    if int(use_slope) == 1:
        trend = trend & df['SLOPE_POS']

    # Entry modes
    rsi_cross_up = (df['RSI'] > rsi_buy) & (df['RSI'].shift(1) <= rsi_buy)
    bb_cross_up  = (df['Close'] > df['BB_LO']) & (df['Close'].shift(1) <= df['BB_LO'].shift(1))

    entry_core = rsi_cross_up if int(mode) == 0 else bb_cross_up
    entries = trend & entry_core & (df['Close'] > df['EMA_FAST'])

    # Exit modes
    if int(exit_md) == 0:
        exits = (df['Close'] >= df['BB_UP']) | (df['RSI'] >= rsi_sell)
    else:
        exits = df['Close'] < df['EMA_FAST']

    # 4) Shift + normalizzazione booleana
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return shifted_entries, shifted_exits

    
############################
# Strategy dbma_matrix
############################

# =========================
# === Indicatori (IND) ====
# =========================

def ind_dbma_matrix_dbma(df: pd.DataFrame,
                         ma_period: int = 20,
                         sd_near: float = 0.5,
                         sd_far: float = 1.0,
                         compress_lookback: int = 20):
    """
    DBMA: MA centrale + doppie Bollinger (0.5σ e 1.0σ) + 'compressione' (bande che si stringono).
    Ritorna: (ma, up_near, lo_near, up_far, lo_far, compress)
    """
    close = df['Close']
    ma = close.rolling(ma_period, min_periods=1).mean()
    # Deviazione standard 'classica' su ma_period
    stdev = close.rolling(ma_period, min_periods=1).std(ddof=0)

    up_near = ma + sd_near * stdev
    lo_near = ma - sd_near * stdev
    up_far  = ma + sd_far  * stdev
    lo_far  = ma - sd_far  * stdev

    # Banda "near" normalizzata e compressione come stringimento sotto mediana rolling
    bw_near = (up_near - lo_near) / ma.replace(0, np.nan)
    bw_med  = bw_near.rolling(compress_lookback, min_periods=1).median()
    compress = (bw_near <= bw_med)

    return ma, up_near, lo_near, up_far, lo_far, compress


def ind_dbma_matrix_series(df: pd.DataFrame,
                           ma: pd.Series,
                           mom_period: int = 10,
                           smooth: int = 5) -> pd.Series:
    """
    Matrix Series (approssimazione quantitativa coerente con la descrizione dell'articolo):
    combina distanza normalizzata dal 'mean' (z-like) e momentum, con smoothing EMA.
    Ritorna: ms (oscillatore che oscilla attorno a 0).
    """
    close = df['Close']

    # Volatilità realizzata su mom_period per normalizzare il momentum
    ret = close.pct_change()
    vol = ret.rolling(mom_period, min_periods=1).std(ddof=0).replace(0, np.nan)

    # Momentum percentuale su mom_period
    mom = close.pct_change(mom_period)

    # Distanza dal mean normalizzata dalla σ su mom_period (evita divisioni per 0)
    st_mom = close.rolling(mom_period, min_periods=1).std(ddof=0).replace(0, np.nan)
    zdist = (close - ma) / st_mom

    # Combinazione: più peso alla "mean reversion within trend" (zdist) e quota al momentum/vol
    ms_raw = 0.6 * zdist + 0.4 * (mom / vol)

    # Smussamento stile EMA
    ms = ms_raw.ewm(span=smooth, adjust=False, min_periods=1).mean()
    return ms


# =======================================
# === Griglia parametri per WFO (WFO) ===
# =======================================

strategy_dbma_matrix_param_ranges = {
    'ma_range'                : range(10, 40, 10),   # periodo MA (default 20)
    'sd_near_range'           : range(5, 8),     # 0.5–0.7  (decimi)
    'sd_far_range'            : range(10, 13),   # 1.0–1.2  (decimi)
    'ms_mom_range'            : range(5, 26, 10),    # lookback momentum/vol
    'ms_smooth_range'         : range(3, 10, 3),    # smoothing oscillator
    'compress_lookback_range' : range(10, 40, 10),  # finestra mediana banda near
}
# Tot combinazioni: 5*3*3*4*4*3 = 2.160 (< 50k come da policy)


# ==============================
# === Funzione di Strategia  ===
# ==============================

def strategy_dbma_matrix(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Genera segnali (entries, exits) usando DBMA + Matrix Series.
    Regole implementate in modo coerente con l'articolo:
      LONG:
        - Trend: Close > MA
        - Pullback: minimo candela dentro le bande inferiori (tra MA e 1.0σ)
        - Compression: bande near in compressione (bw_near sotto mediana rolling)
        - Conferma: Matrix flip up (MS > 0 & MS.shift(1) <= 0) e close in miglioramento
      EXIT LONG:
        - Chiusura sopra banda superiore near  (presa profitto “opposite band”)
          O Matrix flip down (MS < 0 & MS.shift(1) >= 0)

      SHORT: condizioni speculari sui massimi / bande superiori.

    Segnali shiftati di 1 barra per evitare look-ahead.
    Il filtro year, se passato, viene applicato con slicing del DataFrame (dopo il calcolo indicatori).
    """
    # --- Lettura parametri (chiavi allineate alla griglia WFO)
    ma_p          = params.get('ma_range')
    sd_near_p10   = params.get('sd_near_range')   # in decimi
    sd_far_p10    = params.get('sd_far_range')    # in decimi
    ms_mom_p      = params.get('ms_mom_range')
    ms_smooth_p   = params.get('ms_smooth_range')
    compr_lb_p    = params.get('compress_lookback_range')

    sd_near = sd_near_p10 / 10.0
    sd_far  = sd_far_p10  / 10.0

    df = data.copy()

    # 1) --- Calcolo indicatori sull'intero df ---
    ma, up_near, lo_near, up_far, lo_far, compress = ind_dbma_matrix_dbma(
        df,
        ma_period=ma_p,
        sd_near=sd_near,
        sd_far=sd_far,
        compress_lookback=compr_lb_p
    )
    ms = ind_dbma_matrix_series(df, ma=ma, mom_period=ms_mom_p, smooth=ms_smooth_p)

    df['MA'] = ma
    df['BB_upper_near'] = up_near
    df['BB_lower_near'] = lo_near
    df['BB_upper_far']  = up_far
    df['BB_lower_far']  = lo_far
    df['DBMA_Compress'] = compress
    df['MS'] = ms

    # 2) --- Filtro per anno (slicing) ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) --- Condizioni di ingresso/uscita ---
    # Trend
    trend_long  = df['Close'] > df['MA']
    trend_short = df['Close'] < df['MA']

    # Pullback "dentro le bande": LONG (Low tra MA e 1.0σ inferiore), SHORT (High tra MA e 1.0σ superiore)
    pullback_long  = (df['Low'] <= df['MA']) & (df['Low'] >= df['BB_lower_far'])
    pullback_short = (df['High'] >= df['MA']) & (df['High'] <= df['BB_upper_far'])

    # Compressione bande (near)
    comp = df['DBMA_Compress']

    # Conferma Matrix: flip con chiusura favorevole
    flip_up   = (df['MS'] > 0) & (df['MS'].shift(1) <= 0)
    flip_down = (df['MS'] < 0) & (df['MS'].shift(1) >= 0)
    bull_close = df['Close'] > df['Close'].shift(1)
    bear_close = df['Close'] < df['Close'].shift(1)

    entries_long  = trend_long  & comp & pullback_long  & flip_up   & bull_close
    entries_short = trend_short & comp & pullback_short & flip_down & bear_close
    entries = entries_long | entries_short

    # Exit: opposite band o flip contrario del Matrix
    exits_long  = (df['Close'] >= df['BB_upper_near']) | flip_down
    exits_short = (df['Close'] <= df['BB_lower_near']) | flip_up
    exits = (exits_long & trend_long) | (exits_short & trend_short)

    # 4) --- Shift di 1 barra e normalizzazione booleana ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return shifted_entries, shifted_exits
    
############################
# Strategy trend_dc_atr_v2
############################

# === MA slow / EMA fast + slope ===
def ind_trend_dc_atr_v2_ma(df: pd.DataFrame, slow_period: int = 150, fast_period: int = 20):
    close = df['Close']
    sma_slow = close.rolling(slow_period, min_periods=1).mean()
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=1).mean()
    slope_pos = sma_slow > sma_slow.shift(1)
    return sma_slow, ema_fast, slope_pos

# === Donchian PREV (bande riferite a t-1 per evitare look-ahead) ===
def ind_trend_dc_atr_v2_donchian_prev(df: pd.DataFrame, n_entry: int = 40, n_exit: int = 20):
    high = df['High']; low = df['Low']; close = df['Close']
    dc_up_raw = high.rolling(n_entry, min_periods=1).max()
    dc_lo_raw = low.rolling(n_exit,  min_periods=1).min()
    dc_up_prev = dc_up_raw.shift(1)   # soglia valida alla fine di t-1
    dc_lo_prev = dc_lo_raw.shift(1)
    return dc_up_prev, dc_lo_prev

# === ATR (stile Wilder) + trailing calcolato su t-1 ===
def ind_trend_dc_atr_v2_atr_trailing(df: pd.DataFrame, period: int = 14, mult: float = 3.0, win_max: int = 20):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = np.maximum(np.maximum((high - low), (high - prev_close).abs()), (low - prev_close).abs())
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    roll_max_prev = close.shift(1).rolling(win_max, min_periods=1).max()
    atr_tsl_prev = roll_max_prev - mult * atr.shift(1)  # trailing valutato su info fino a t-1
    return atr, atr_tsl_prev

# --- Griglia parametri per WFO (≈ 2.6k, come prima) ---
strategy_trend_dc_atr_v2_param_ranges = {
    'slow_ma_range'      : [100, 150, 200],
    'fast_ma_range'      : [10, 20],
    'dc_entry_range'     : [20, 40, 60],
    'dc_exit_range'      : [10, 20, 30],
    'atr_period_range'   : [10, 14, 20],
    'atr_mult_range'     : [2, 3],
    'use_slope_range'    : [0, 1],
    'exit_mode_range'    : [0, 1],   # 0=Donchian low prev; 1=ATR trailing prev
    'breakout_only_range': [0, 1],
}

# --- Strategia ---
def strategy_trend_dc_atr_v2(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Long-only per SPY/QQQ. Tutte le soglie (Donchian/ATR) sono valutate su t-1.
    Entrate: breakout (Close > DC_UP_prev) oppure (se consentito) re-entry su cross-up EMA_fast dopo breakout recente.
    Uscite: Donchian low prev (M) oppure ATR trailing prev.
    Segnali shiftati di 1 barra.
    """
    slow_p   = params.get('slow_ma_range')
    fast_p   = params.get('fast_ma_range')
    n_entry  = params.get('dc_entry_range')
    n_exit   = params.get('dc_exit_range')
    atr_p    = params.get('atr_period_range')
    atr_k    = params.get('atr_mult_range')
    use_sl   = params.get('use_slope_range', 0)
    exit_md  = params.get('exit_mode_range', 0)
    brk_only = params.get('breakout_only_range', 0)

    df = data.copy()

    # 1) Indicatori su tutto il df
    sma_slow, ema_fast, slope_pos = ind_trend_dc_atr_v2_ma(df, slow_period=slow_p, fast_period=fast_p)
    dc_up_prev, dc_lo_prev = ind_trend_dc_atr_v2_donchian_prev(df, n_entry=n_entry, n_exit=n_exit)
    atr, atr_tsl_prev = ind_trend_dc_atr_v2_atr_trailing(df, period=atr_p, mult=atr_k, win_max=n_exit)

    df['SMA_SLOW'] = sma_slow
    df['EMA_FAST'] = ema_fast
    df['DC_UP_PREV'] = dc_up_prev
    df['DC_LO_PREV'] = dc_lo_prev
    df['ATR_TSL_PREV'] = atr_tsl_prev
    df['SLOPE_POS'] = slope_pos

    # 2) Slicing anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Condizioni
    trend = df['Close'] > df['SMA_SLOW']
    if int(use_sl) == 1:
        trend = trend & df['SLOPE_POS']

    # Breakout su banda di ieri (cross up non-self-referential)
    breakout = (df['Close'] > df['DC_UP_PREV']) & (df['Close'].shift(1) <= df['DC_UP_PREV'])

    # Re-entry dopo un breakout recente (entro N barre), usando la banda di ieri
    if int(brk_only) == 1:
        reentry = pd.Series(False, index=df.index)
    else:
        was_above_upper_recent = ((df['Close'].shift(1) > df['DC_UP_PREV']).rolling(window=n_entry, min_periods=1).max() > 0)
        reentry = was_above_upper_recent & (df['Close'] > df['EMA_FAST']) & (df['Close'].shift(1) <= df['EMA_FAST'].shift(1))

    entries = trend & (breakout | reentry)

    # Uscite valutate su soglie t-1
    if int(exit_md) == 0:
        exits = df['Close'] < df['DC_LO_PREV']
    else:
        exits = df['Close'] < df['ATR_TSL_PREV']

    # 4) Shift + normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ###############################################
# # Strategy ma_alignment
# ###############################################

# # === Indicatori: EMA, SMA brevi e lunghe ===

# def ind_ma_alignment_ema(df: pd.DataFrame, period: int) -> pd.Series:
#     """EMA sul Close."""
#     return df['Close'].ewm(span=period, adjust=False).mean()


# def ind_ma_alignment_sma(df: pd.DataFrame, period: int) -> pd.Series:
#     """SMA sul Close."""
#     return df['Close'].rolling(window=period, min_periods=1).mean()


# # --- Griglia parametri per WF Optimization ---
# # Griglia contenuta (<50k combinazioni): tre range piccoli.
# strategy_ma_alignment_param_ranges = {
#     "ema_range"  : range(10, 31, 5),   # EMA veloce (default 20)
#     "sma_mid_range": range(40, 81, 10), # SMA media (default 50)
#     "sma_long_range": range(150, 251, 25) # SMA lunga (default 200)
# }


# # --- Funzione di strategia ---
# def strategy_ma_alignment(data: pd.DataFrame, params: dict, year: int | None = None):
#     """
#     Trend Alignment a 3 medie mobili.
    
#     Regole:
#       Entry se:
#         Close > EMA
#         EMA > SMA_mid
#         SMA_mid > SMA_long

#       Exit se QUALUNQUE condizione si rompe:
#         (Close < EMA) OR (EMA < SMA_mid) OR (SMA_mid < SMA_long)

#     Segnali shiftati di 1 barra per evitare look-ahead.
#     """

#     # --- Lettura parametri WFO ---
#     ema_p      = params.get("ema_range")
#     sma_mid_p  = params.get("sma_mid_range")
#     sma_long_p = params.get("sma_long_range")

#     df = data.copy()

#     # ============================================================
#     # 1) Calcolo indicatori sull'intero df (regola obbligatoria)
#     # ============================================================
#     df["EMA"]      = ind_ma_alignment_ema(df, ema_p)
#     df["SMA_MID"]  = ind_ma_alignment_sma(df, sma_mid_p)
#     df["SMA_LONG"] = ind_ma_alignment_sma(df, sma_long_p)

#     # ============================================================
#     # 2) Filtro anno (slicing, NO maschere booleane)
#     # ============================================================
#     if year is not None:
#         df = df[df.index.year == int(year)]

#     # ============================================================
#     # 3) ENTRY & EXIT sul df già filtrato
#     # ============================================================

#     entries = (
#         (df['Close'] > df['EMA']) &
#         (df['EMA'] > df['SMA_MID']) &
#         (df['SMA_MID'] > df['SMA_LONG'])
#     )

#     exits = (
#         (df['Close'] < df['EMA']) |
#         (df['EMA'] < df['SMA_MID']) |
#         (df['SMA_MID'] < df['SMA_LONG'])
#     )

#     # ============================================================
#     # 4) Shift segnali + normalizzazione
#     # ============================================================
#     entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
#     exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

#     return entries, exits


###############################################
# Strategy ma_risk_filter
###############################################

# === Indicatori: SMA lunga ===

def ind_ma_risk_filter_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """SMA sul Close, per regime di lungo periodo."""
    return df['Close'].rolling(window=period, min_periods=1).mean()


# --- Griglia parametri per WF Optimization ---
# Griglia contenuta: 5 * 4 * 3 = 60 combinazioni
# - long_sma_range: periodo media lunga
# - band_bps_range: banda in "basis points di percentuale": 2 -> 2%, 4 -> 4%, ...
# - slope_lookback_range: distanza (in barre) per calcolare la pendenza della media
strategy_ma_risk_filter_param_ranges = {
    "long_sma_range"       : range(150, 251, 25),  # 150, 175, 200, 225, 250
    "band_bps_range"       : range(2, 10, 2),      # 2, 4, 6, 8  (=> 2%-8%)
    "slope_lookback_range" : range(20, 61, 20)     # 20, 40, 60
}


# --- Funzione di strategia ---
def strategy_ma_risk_filter(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Strategia MA Risk Filter: replica un Buy&Hold con filtro di regime.

    Idee chiave:
      - In mercato (Risk-ON) quasi sempre.
      - Fuori mercato (Risk-OFF) solo quando:
          Close << SMA_LONG  (sotto banda percentuale)
          e SMA_LONG ha pendenza negativa.

    Regole dettagliate:

      1) Indicatori:
         MA_LONG  = SMA_LONG su Close
         slope    = MA_LONG - MA_LONG.shift(slope_lookback)

      2) Risk-OFF:
         risk_off = (Close < MA_LONG * (1 - band_pct)) & (slope <= 0)

      3) Risk-ON:
         risk_on = ~risk_off

      4) Segnali:
         entries = risk_on   (voglio essere investito quando NON sono in risk_off)
         exits   = risk_off  (esco quando rischio strutturale aumenta)

      5) Shift segnali di 1 barra (no look-ahead).

    Questo approccio tende a:
      - comportarsi in modo simile al B&H nelle fasi rialziste "normali";
      - ridurre drawdown e durata dei bear market profondi/prolungati.
    """

    # --- Lettura parametri WFO ---
    long_sma_p       = params.get("long_sma_range")
    band_bps_p       = params.get("band_bps_range")       # es. 2 -> 0.02
    slope_lookback_p = params.get("slope_lookback_range")

    # Banda percentuale in forma decimale
    band_pct = band_bps_p / 100.0

    df = data.copy()

    # ============================================================
    # 1) Calcolo indicatori sull'intero df
    # ============================================================
    df["SMA_LONG"] = ind_ma_risk_filter_sma(df, long_sma_p)
    df["MA_SLOPE"] = df["SMA_LONG"] - df["SMA_LONG"].shift(slope_lookback_p)

    # ============================================================
    # 2) Filtro per anno (slicing, NO maschere booleane)
    # ============================================================
    if year is not None:
        df = df[df.index.year == int(year)]

    # Se dopo il filtro anno non ho abbastanza dati, evito errori
    if df.empty:
        empty_idx = data.index[data.index.year == int(year)] if year is not None else data.index
        entries_empty = pd.Series(False, index=empty_idx)
        exits_empty   = pd.Series(False, index=empty_idx)
        entries_empty = entries_empty.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
        exits_empty   = exits_empty.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
        return entries_empty, exits_empty

    # ============================================================
    # 3) ENTRY & EXIT sul df già filtrato
    # ============================================================

    # Condizione di RISK-OFF: prezzo sotto banda e media in discesa
    risk_off = (df["Close"] < df["SMA_LONG"] * (1.0 - band_pct)) & (df["MA_SLOPE"] <= 0)

    # Condizione di RISK-ON: tutto il resto
    risk_on = ~risk_off

    # Entry: voglio essere investito quando sono in RISK-ON
    entries = risk_on

    # Exit: esco quando passa in RISK-OFF
    exits = risk_off

    # ============================================================
    # 4) Shift segnali + normalizzazione (no FutureWarning)
    # ============================================================
    entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return entries, exits

############################################################
# A) strategy breakout_trend  (Trend Following su breakout)
############################################################

# === Indicatori Breakout (Highest High / Lowest Low) ===

def ind_breakout_trend_hh_ll(
    df: pd.DataFrame,
    hh_period: int,
    ll_period: int
):
    """
    Calcola:
      - HH: highest high dei 'Close' sulle ultime hh_period barre (solo passato)
      - LL: lowest low dei 'Close' sulle ultime ll_period barre (solo passato)

    NOTA: uso .shift(1) per evitare look-ahead (solo barre già chiuse).
    """
    close_shift = df['Close'].shift(1)

    hh = close_shift.rolling(window=hh_period, min_periods=hh_period).max()
    ll = close_shift.rolling(window=ll_period, min_periods=ll_period).min()

    return hh, ll


# --- Griglia parametri per WF Optimization ---
# 5 * 4 = 20 combinazioni (griglia molto contenuta)
strategy_breakout_trend_param_ranges = {
    "hh_period_range": range(20, 61, 10),  # 20, 30, 40, 50, 60
    "ll_period_range": range(10, 41, 10)   # 10, 20, 30, 40
}


# --- Funzione di strategia A ---
def strategy_breakout_trend(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Trend Following puro in stile Turtle semplificato.

    Regole:
      - Indicatori:
          HH = max Close ultime hh_period barre (solo passato)
          LL = min Close ultime ll_period barre (solo passato)

      - Entry:
          Close > HH

      - Exit:
          Close < LL

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """

    hh_p = params.get("hh_period_range")
    ll_p = params.get("ll_period_range")

    df = data.copy()

    # 1) Indicatori su tutto il df
    df["HH"], df["LL"] = ind_breakout_trend_hh_ll(df, hh_period=hh_p, ll_period=ll_p)

    # 2) Filtro per anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Entry / Exit su df filtrato
    entries = df["Close"] > df["HH"]
    exits   = df["Close"] < df["LL"]

    # 4) Shift + normalizzazione
    entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return entries, exits


############################################################
# B) strategy crash_guard  (Crash Protection + Slow Re-Entry)
############################################################

# === Indicatori: EMA + ATR per rilevare breakdown ===

def ind_crash_guard_ema_atr(
    df: pd.DataFrame,
    ema_period: int,
    atr_period: int
):
    """
    Calcola:
      - EMA sul Close
      - ATR classico a n periodi (High/Low/Close), media mobile semplice
    """
    close = df['Close']
    high  = df.get('High', close)
    low   = df.get('Low', close)

    ema = close.ewm(span=ema_period, adjust=False).mean()

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    return ema, atr


# --- Griglia parametri per WF Optimization ---
# 4 * 3 * 3 * 4 = 144 combinazioni
# multipli ATR salvati in decimi: es. 15 => 1.5
strategy_crash_guard_param_ranges = {
    "ema_period_range"      : range(20, 61, 10),   # 20, 30, 40, 50, 60
    "atr_period_range"      : range(14, 29, 7),    # 14, 21, 28
    "exit_mult_x10_range"   : range(10, 21, 5),    # 1.0, 1.5, 2.0
    "entry_mult_x10_range"  : range(15, 31, 5)     # 1.5, 2.0, 2.5, 3.0
}


# --- Funzione di strategia B ---
def strategy_crash_guard(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Crash Protection + Slow Re-Entry.

    Idee:
      - Uscire velocemente quando il prezzo rompe l'EMA verso il basso in modo significativo (ATR).
      - Rientrare solo quando il prezzo recupera sopra l'EMA con margine superiore.

    Regole:

      Indicatori:
        EMA, ATR

      Exit (risk-off):
        Close < EMA - exit_mult * ATR

      Entry (risk-on):
        Close > EMA + entry_mult * ATR

    Segnali shiftati di 1 barra.
    """

    ema_p      = params.get("ema_period_range")
    atr_p      = params.get("atr_period_range")
    exit_x10   = params.get("exit_mult_x10_range")
    entry_x10  = params.get("entry_mult_x10_range")

    exit_mult  = exit_x10 / 10.0
    entry_mult = entry_x10 / 10.0

    df = data.copy()

    # 1) Indicatori sull'intero df
    df["EMA"], df["ATR"] = ind_crash_guard_ema_atr(df, ema_period=ema_p, atr_period=atr_p)

    # 2) Filtro anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Entry/Exit
    # Risk-OFF (uscita rapida): breakdown forte
    risk_off = df["Close"] < (df["EMA"] - exit_mult * df["ATR"])

    # Risk-ON (rientro lento): recupero deciso
    risk_on = df["Close"] > (df["EMA"] + entry_mult * df["ATR"])

    # Vogliamo essere long quando risk_on, flat quando risk_off
    entries = risk_on
    exits   = risk_off

    # 4) Shift + normalizzazione
    entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return entries, exits


#################################################################
# C) strategy breakout_reversal_guard (A + B combinati gerarchici)
#################################################################

# === Indicatori combinati: EMA+ATR + HH/LL ===

def ind_breakout_guard_ema_atr_hh_ll(
    df: pd.DataFrame,
    ema_period: int,
    atr_period: int,
    hh_period: int,
    ll_period: int
):
    """
    Indicatori combinati:
      - EMA (Close)
      - ATR (High/Low/Close, rolling)
      - HH: highest Close sulle ultime hh_period barre (solo passato)
      - LL: lowest Close sulle ultime ll_period barre (solo passato)
    """
    close = df['Close']
    high  = df.get('High', close)
    low   = df.get('Low', close)

    # EMA
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    # HH / LL (solo passato)
    close_shift = close.shift(1)
    hh = close_shift.rolling(window=hh_period, min_periods=hh_period).max()
    ll = close_shift.rolling(window=ll_period, min_periods=ll_period).min()

    return ema, atr, hh, ll


# --- Griglia parametri per WF Optimization ---
# 2 * 2 * 3 * 2 * 2 = 48 combinazioni
strategy_breakout_reversal_guard_param_ranges = {
    "ema_period_range"    : range(20, 41, 20),  # 20, 40
    "atr_period_range"    : range(14, 29, 14),  # 14, 28
    "exit_mult_x10_range" : range(10, 21, 5),   # 1.0, 1.5, 2.0
    "hh_period_range"     : range(20, 41, 20),  # 20, 40
    "ll_period_range"     : range(10, 21, 10)   # 10, 20
}


# --- Funzione di strategia C ---
def strategy_breakout_reversal_guard(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Combinazione di:
      - Crash Protection (EMA+ATR)
      - Breakout Trend Following (HH/LL)

    Logica gerarchica:

      1) Regime di RISK-OFF (come crash_guard):
           Close < EMA - exit_mult * ATR

      2) Regime di RISK-ON = non RISK-OFF.

      3) Entry:
           RISK-ON AND Close > HH

      4) Exit:
           RISK-OFF OR Close < LL

    Così:
      - eviti di entrare in breakout mentre il titolo è in pieno crash,
      - sfrutti i breakout solo in contesti "sanificati" dal filtro di rischio.
    """

    ema_p    = params.get("ema_period_range")
    atr_p    = params.get("atr_period_range")
    exit_x10 = params.get("exit_mult_x10_range")
    hh_p     = params.get("hh_period_range")
    ll_p     = params.get("ll_period_range")

    exit_mult = exit_x10 / 10.0

    df = data.copy()

    # 1) Indicatori su tutto il df
    df["EMA"], df["ATR"], df["HH"], df["LL"] = ind_breakout_guard_ema_atr_hh_ll(
        df,
        ema_period=ema_p,
        atr_period=atr_p,
        hh_period=hh_p,
        ll_period=ll_p
    )

    # 2) Filtro anno
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) Entry / Exit
    # Regime RISK-OFF (come crash_guard)
    risk_off = df["Close"] < (df["EMA"] - exit_mult * df["ATR"])
    risk_on  = ~risk_off

    # Entry solo se risk_on e breakout sopra HH
    entries = risk_on & (df["Close"] > df["HH"])

    # Exit se risk_off o breakdown sotto LL
    exits = risk_off | (df["Close"] < df["LL"])

    # 4) Shift + normalizzazione
    entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return entries, exits

############################
# Strategy macd_cross
############################

# === MACD (EMA/EMA + Signal EMA) ===
def ind_macd_cross_macd(df: pd.DataFrame,
                        fast_period: int = 12,
                        slow_period: int = 26,
                        signal_period: int = 9):
    """
    MACD TradingView-like:
      - Source: Close
      - Fast MA: EMA(fast_period)
      - Slow MA: EMA(slow_period)
      - MACD line = EMA_fast - EMA_slow
      - Signal line = EMA(MACD line, signal_period)
    Ritorna: (macd_line, signal_line, histogram)
    """
    close = df['Close']

    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=1).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# --- Griglia parametri per WF Optimization ---
strategy_macd_cross_param_ranges = {
    'fast_range'   : range(6, 19, 2),   # 6..18
    'slow_range'   : range(18, 41, 4),  # 18..40
    'signal_range' : range(5, 16, 2)    # 5..15
}


# --- Funzione di strategia ---
def strategy_macd_cross(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia MACD cross (TradingView style: EMA/EMA, Signal EMA, source=Close).

    Regole (sul df completo; slicing per anno applicato SOLO dopo il calcolo indicatori):
      Entry se:
        MACD (linea "veloce") incrocia verso l'alto la Signal (linea "lenta")
      Exit se:
        MACD incrocia verso il basso la Signal

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    """

    # Leggi parametri (coerenti con le chiavi della griglia WFO)
    fast_p = params.get('fast_range')
    slow_p = params.get('slow_range')
    sig_p  = params.get('signal_range')

    df = data.copy()

    # --- Calcolo indicatori
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ind_macd_cross_macd(
        df, fast_period=fast_p, slow_period=slow_p, signal_period=sig_p
    )

    # --- Filtro per anno dopo il calcolo degli indicatori ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Condizioni entry/exit (cross)
    macd = df['MACD']
    sig  = df['MACD_Signal']

    entries = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    exits   = (macd < sig) & (macd.shift(1) >= sig.shift(1))

    # --- Shift di 1 barra per utilizzo in backtest ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

######################################
### Strategie Kryptera
######################################


############################
# Strategy cloud_lag_wr
############################

# === ICHIMOKU CLOUD (KUMO) ===

def ind_cloud_lag_wr_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> pd.DataFrame:
    """
    Calcolo completo dell'Ichimoku Cloud:
    - tenkan_sen
    - kijun_sen
    - senkou_span_a
    - senkou_span_b
    - chikou_span
    """
    df = df.copy()

    high = df['High']
    low = df['Low']
    close = df['Close']

    # Tenkan-sen (conversion line)
    tenkan_high = high.rolling(window=tenkan_period, min_periods=1).max()
    tenkan_low = low.rolling(window=tenkan_period, min_periods=1).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2.0

    # Kijun-sen (base line)
    kijun_high = high.rolling(window=kijun_period, min_periods=1).max()
    kijun_low = low.rolling(window=kijun_period, min_periods=1).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2.0

    # Senkou Span A (leading span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2.0).shift(displacement)

    # Senkou Span B (leading span B)
    senkou_b_high = high.rolling(window=senkou_b_period, min_periods=1).max()
    senkou_b_low = low.rolling(window=senkou_b_period, min_periods=1).min()
    df['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2.0).shift(displacement)

    # Chikou Span (lagging span)
    df['chikou_span'] = close.shift(-displacement)

    return df


def ind_cloud_lag_wr_kumo_breakout_bearish(df: pd.DataFrame) -> pd.Series:
    """
    Segnale di breakdown ribassista sotto la Kumo:
    True se Close < min(Senkou Span A, Senkou Span B).
    """
    cloud_min = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    return df['Close'] < cloud_min


# === LAGUERRE RSI ===

def ind_cloud_lag_wr_laguerre_rsi(
    series: pd.Series,
    gamma: float = 0.5
) -> pd.Series:
    """
    Laguerre RSI (0..1) calcolato su una Serie di prezzi (tipicamente Close).
    Come l'originale, ma con .ffill() al posto di fillna(method='ffill').
    """
    L0 = L1 = L2 = L3 = 0.0
    lrsi_vals: list[float] = []

    # <<< PRIMA MODIFICA: niente più "method='ffill'" >>>
    for price in series.ffill():
        L0 = (1 - gamma) * price + gamma * L0
        L1 = -gamma * L0 + L0 + gamma * L1
        L2 = -gamma * L1 + L1 + gamma * L2
        L3 = -gamma * L2 + L2 + gamma * L3

        CU = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
        CD = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)

        lrsi = CU / (CU + CD) if (CU + CD) != 0 else 0.0
        lrsi_vals.append(lrsi)

    return pd.Series(lrsi_vals, index=series.index)


def ind_cloud_lag_wr_laguerre_cross_below(
    lrsi: pd.Series,
    level: float = 0.5
) -> pd.Series:
    """
    Segnale booleano: Laguerre RSI che incrocia verso il basso un certo livello.
    True se LRsi < level e LRsi.shift(1) >= level.
    """
    return (lrsi < level) & (lrsi.shift(1) >= level)


# === WILLIAMS %R ===

def ind_cloud_lag_wr_williams_r(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calcolo di Williams %R su High/Low/Close.
    Range tipico [-100, 0].
    """
    high = df['High'].rolling(window=period, min_periods=1).max()
    low = df['Low'].rolling(window=period, min_periods=1).min()
    wr = -100.0 * (high - df['Close']) / (high - low).replace(0, np.nan)
    return wr


def ind_cloud_lag_wr_wr_is_falling(wr: pd.Series) -> pd.Series:
    """
    Segnale booleano: Williams %R in diminuzione (momentum ribassista che perde forza).
    True se diff < 0.
    """
    return wr.diff() < 0


# === RSI CLASSICO ===

def ind_cloud_lag_wr_rsi(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    RSI stile Wilder/EMA, coerente con l'articolo (EWMA su gain/loss).
    """
    close = df['Close']
    delta = close.diff()

    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(span=period, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(span=period, adjust=False, min_periods=1).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def ind_cloud_lag_wr_rsi_lower_than_os(
    rsi: pd.Series,
    oversold_level: float = 30.0
) -> pd.Series:
    """
    Segnale booleano: RSI sotto la soglia di ipervenduto (default 30).
    """
    return rsi < oversold_level
    
# ICHIMOKU_DISPLACEMENT = 26
# ICHIMOKU_KIJUN_PERIOD = 26
# ICHIMOKU_SENKOU_B_PERIOD = 52
# ICHIMOKU_TENKAN_PERIOD = 9

# LRSI_GAMMA = 0.5
# LRSI_LEVEL = 0.5

# WR_LEVEL = -50
# WR_PERIOD = 14

# RSI_OVERBOUGHT_LEVEL = 70
# RSI_OVERSOLD_LEVEL = 30
# RSI_PERIOD = 14
# RSI_SHIFT = 5
# RSI_SHIFT_2 = 10

# --- Griglia parametri per WF Optimization ---
strategy_cloud_lag_wr_param_ranges = {
    'ichimoku_tenkan_range'      : range(9, 10),   # sempre 9
    'ichimoku_kijun_range'       : range(26, 27),  # sempre 26
    'ichimoku_senkou_b_range'    : range(52, 53),  # sempre 52
    'ichimoku_displacement_range': range(26, 27),  # sempre 26

    # Parametri Laguerre RSI (valori *100 per usare range interi)
    'laguerre_gamma_mult_range'  : range(40, 61, 10),  # 40,50,60
    'laguerre_level_mult_range'  : range(40, 61, 10),

    # Williams %R e RSI classico
    'wr_period_range'            : range(10, 21, 2),   # 10,12,...,20
    'rsi_period_range'           : range(10, 21, 2),
    'rsi_oversold_range'         : range(25, 36, 5),   # 25,30,35
}


# --- Funzione di strategia ---
def strategy_cloud_lag_wr(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Strategia "Multi-Oscillator Cloud Breakdown":
    operatività SHORT (ingressi su breakdown ribassisti,
    uscite su esaurimento del momentum ribassista).
    """

    # --- Lettura parametri dalla griglia WFO ---
    tenkan_p   = params.get('ichimoku_tenkan_range', 9)
    kijun_p    = params.get('ichimoku_kijun_range', 26)
    senkou_b_p = params.get('ichimoku_senkou_b_range', 52)
    disp_p     = params.get('ichimoku_displacement_range', 26)

    gamma_mult = params.get('laguerre_gamma_mult_range', 50)
    level_mult = params.get('laguerre_level_mult_range', 50)
    wr_period  = params.get('wr_period_range', 14)
    rsi_period = params.get('rsi_period_range', 14)
    rsi_os     = params.get('rsi_oversold_range', 30)

    gamma = float(gamma_mult) / 100.0
    lag_level = float(level_mult) / 100.0
    rsi_os_level = float(rsi_os)

    df = data.copy()

    # === 1) Indicatori su TUTTO il df ===

    df = ind_cloud_lag_wr_ichimoku(
        df,
        tenkan_period=tenkan_p,
        kijun_period=kijun_p,
        senkou_b_period=senkou_b_p,
        displacement=disp_p
    )
    df['Kumo_Bearish'] = ind_cloud_lag_wr_kumo_breakout_bearish(df)

    df['Laguerre_RSI'] = ind_cloud_lag_wr_laguerre_rsi(df['Close'], gamma=gamma)
    df['Laguerre_Cross_Below'] = ind_cloud_lag_wr_laguerre_cross_below(
        df['Laguerre_RSI'], level=lag_level
    )

    df['Williams_%R'] = ind_cloud_lag_wr_williams_r(df, period=wr_period)
    df['WR_is_Falling'] = ind_cloud_lag_wr_wr_is_falling(df['Williams_%R'])

    df['RSI'] = ind_cloud_lag_wr_rsi(df, period=rsi_period)
    df['RSI_Lower_Than_OS'] = ind_cloud_lag_wr_rsi_lower_than_os(
        df['RSI'], oversold_level=rsi_os_level
    )

    # === 2) Filtro anno ===
    if year is not None:
        df = df[df.index.year == int(year)]

    # === 3) Condizioni entry/exit ===

    entries = (
        df['Kumo_Bearish'].astype(bool) &
        df['Laguerre_Cross_Below'].astype(bool)
    )

    exits = (
        df['WR_is_Falling'].astype(bool) &
        df['RSI_Lower_Than_OS'].astype(bool)
    )

    # === 4) Shift + normalizzazione in stile "strategy_bollinger" ===
    # <<< SECONDA MODIFICA: stesso ordine della strategia che non genera warning >>>
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits
    
############################
# Strategy kc_std_volatility_fade
############################

# === Keltner Channel ===
def ind_kc_std_volatility_fade_keltner(
    df: pd.DataFrame,
    period: int = 20,
    multiplier: float = 2.0
):
    """
    Keltner Channel basato su Typical Price e ATR semplice.
    Ritorna: KC_Mid, KC_Upper, KC_Lower
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tp = (high + low + close) / 3.0
    kc_mid = tp.ewm(span=period, adjust=False, min_periods=1).mean()

    tr = np.maximum(np.maximum(high - low, (high - close.shift(1)).abs()), (low - close.shift(1)).abs())

    atr = tr.rolling(period, min_periods=1).mean()

    kc_upper = kc_mid + multiplier * atr
    kc_lower = kc_mid - multiplier * atr

    return kc_mid, kc_upper, kc_lower


# === Volatility (Rolling STD) ===
def ind_kc_std_volatility_fade_std(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Rolling Standard Deviation del Close.
    """
    return df['Close'].rolling(period, min_periods=1).std()


# --- Griglia parametri per WF Optimization ---
strategy_kc_std_volatility_fade_param_ranges = {
    'kc_period_range' : range(10, 31, 5),     # periodo Keltner
    'kc_mult_range'   : range(15, 31, 5),     # moltiplicatore *10 (1.5 → 3.0)
    'std_period_range': range(10, 31, 5),     # periodo STD
    'std_shift_range' : range(3, 11, 2)       # confronto volatilità
}


# --- Funzione di strategia ---
def strategy_kc_std_volatility_fade(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Genera segnali (entries, exits) usando:
    - Keltner Channel (mean-reversion strutturale)
    - STD come filtro di uscita per espansione della volatilità

    Regole:
      Entry:
        Close > KC_Lower
      Exit:
        STD > STD.shift(shift)

    Ordine obbligatorio K_Strategy:
      1) Calcolo indicatori su df completo
      2) Filtro per anno via slicing
      3) Definizione entries / exits
      4) Shift di 1 barra e normalizzazione
    """

    kc_p     = params.get('kc_period_range')
    kc_mult  = params.get('kc_mult_range') / 10.0
    std_p    = params.get('std_period_range')
    shift    = params.get('std_shift_range')

    df = data.copy()

    # --- Indicatori ---
    df['KC_Mid'], df['KC_Upper'], df['KC_Lower'] = ind_kc_std_volatility_fade_keltner(
        df,
        period=kc_p,
        multiplier=kc_mult
    )

    df['STD'] = ind_kc_std_volatility_fade_std(df, period=std_p)

    # --- Filtro per anno ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Segnali ---
    entries = df['Close'] > df['KC_Lower']
    exits   = df['STD'] > df['STD'].shift(shift)

    # --- Shift anti look-ahead ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy bb_lower_reclaim_piercing
############################

# === Bollinger Bands ===
def ind_bb_lower_reclaim_piercing_bb(
    df: pd.DataFrame,
    period: int = 20,
    std_mult: float = 2.0
):
    """
    Calcola Bollinger Bands classiche (MA, Upper, Lower).
    """
    close = df['Close']
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    bb_upper = ma + std_mult * std
    bb_lower = ma - std_mult * std

    return ma, bb_upper, bb_lower

# === Condizione Reclaim Lower Band ===
def ind_bb_lower_reclaim_piercing_reclaim(
    df: pd.DataFrame,
    bb_lower: pd.Series,
    shift: int = 5
) -> pd.Series:
    """
    True se:
    - Open corrente > BB_Lower
    - Open di shift barre fa < BB_Lower
    """
    return (df['Open'] > bb_lower) & (df['Open'].shift(shift) < bb_lower)


# === Piercing Line Candlestick ===
def ind_bb_lower_reclaim_piercing_piercing(df: pd.DataFrame) -> pd.Series:
    """
    Pattern Piercing Line:
    - barra precedente bearish
    - barra corrente bullish
    - Open corrente sotto Close precedente
    - Close corrente sopra metà del body precedente
    """
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)

    return (
        (prev_close < prev_open) &
        (df['Close'] > df['Open']) &
        (df['Open'] < prev_close) &
        (df['Close'] > (prev_open + prev_close) / 2)
    )
    

# --- Griglia parametri per WF Optimization ---
strategy_bb_lower_reclaim_piercing_param_ranges = {
    'bb_period_range': range(15, 31, 5),     # periodo BB
    # 'bb_std_range'   : range(15, 26, 5),     # std * 0.1  → 1.5 – 2.5
    'bb_std_range'   : [1.5, 2.0, 2.5],   # valori REALI
    'reclaim_shift'  : range(3, 11, 2)       # barre lookback reclaim
}


# --- Funzione di strategia ---
def strategy_bb_lower_reclaim_piercing(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Strategia Bollinger Lower Band Reclaim + Piercing Line.

    Entry:
      - Open > Lower Bollinger Band
      - Open.shift(shift) < Lower Bollinger Band

    Exit:
      - Piercing Line candlestick pattern

    Regole K_Strategy:
    1) indicatori su df completo
    2) slicing per anno
    3) definizione entries/exits
    4) shift(1) + bool
    """

    # --- Parametri ---
    bb_period = params.get('bb_period_range')
    bb_std    = params.get('bb_std_range') #/ 10.0
    shift     = params.get('reclaim_shift')

    df = data.copy()

    # --- Indicatori ---
    df['BB_MA'], df['BB_Upper'], df['BB_Lower'] = ind_bb_lower_reclaim_piercing_bb(
        df,
        period=bb_period,
        std_mult=bb_std
    )

    df['BB_Reclaim'] = ind_bb_lower_reclaim_piercing_reclaim(
        df,
        bb_lower=df['BB_Lower'],
        shift=shift
    )

    df['Piercing'] = ind_bb_lower_reclaim_piercing_piercing(df)

    # --- Filtro anno ---
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- Segnali ---
    entries = df['BB_Reclaim']
    exits   = df['Piercing']

    # --- Shift anti look-ahead ---
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

############################
# Strategy gann_cci_kcb_momentum
############################

# =========================================================
# === CCI + Momentum Change (CCI_Change_Up)
# =========================================================
def ind_gann_cci_kcb_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())),
        raw=False
    )
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci


def ind_gann_cci_kcb_cci_change_up(
    df: pd.DataFrame,
    period: int = 14,
    shift_1: int = 5,
    shift_2: int = 10
) -> pd.Series:
    """
    CCI_Change_Up = (CCI > CCI.shift(shift_1)) AND (CCI.shift(shift_1) > CCI.shift(shift_2))
    Replica 1:1 dell'articolo.
    """
    cci = ind_gann_cci_kcb_cci(df, period=period)
    return (cci > cci.shift(shift_1)) & (cci.shift(shift_1) > cci.shift(shift_2))


# =========================================================
# === Bull Power (BullP_Rising)
# =========================================================
def ind_gann_cci_kcb_bull_power(df: pd.DataFrame, ema_period: int = 13) -> pd.Series:
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    bull_power = df['High'] - ema
    return bull_power


# =========================================================
# === Keltner Channel (KC_Open_Below_Upper)
# =========================================================
def ind_gann_cci_kcb_keltner_upper(
    df: pd.DataFrame,
    kc_period: int = 20,
    kc_multiplier: float = 2.0
) -> pd.Series:
    """
    KC_Mid = EMA(TP, kc_period)
    ATR = SMA(TR, kc_period)  [come nell'articolo]
    KC_Upper = KC_Mid + kc_multiplier * ATR
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    kc_mid = tp.ewm(span=kc_period, adjust=False).mean()

    tr = np.maximum(np.maximum(df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs()), (df['Low'] - df['Close'].shift(1)).abs())

    atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + kc_multiplier * atr
    return kc_upper


# =========================================================
# === Gann Hi-Lo (Gann_UpTrend)
# =========================================================
def ind_gann_cci_kcb_gann_uptrend(df: pd.DataFrame) -> pd.Series:
    gann_low = df['Low'].shift(1)
    return df['Close'] > gann_low


# =========================================================
# --- Griglia parametri per WF Optimization ---
# Replica parametri articolo + esplorazione ragionevole
#
# Numero totale combinazioni:
# 5 (cci_period)
# × 3 (cci_shift_1)
# × 3 (cci_shift_2)
# × 5 (ema_period)
# × 5 (kc_period)
# × 4 (kc_mult_tenths)
# = 4.500 combinazioni totali
# =========================================================
strategy_gann_cci_kcb_momentum_param_ranges = {
    'cci_period'     : range(10, 28, 6),     # 10,13,16,19,22
    'cci_shift_1'    : range(3, 9, 2),       # 3,5,7
    'cci_shift_2'    : range(8, 17, 3),      # 8,11,14
    'ema_period'     : range(8, 26, 6),      # 8,11,14,17,20
    'kc_period'      : range(14, 38, 8),     # 14,18,22,26,30
    'kc_mult_tenths' : range(15, 35, 5)      # 1.5,2.0,2.5,3.0 (÷10)
}


# =========================================================
# --- Funzione di strategia ---
# =========================================================
def strategy_gann_cci_kcb_momentum(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Replica fedele dell'articolo:
      Entry:
        - BullP_Rising
        - KC_Open_Below_Upper
      Exit:
        - Gann_UpTrend
        - CCI_Change_Up

    NOTE K_Strategy: solo segnali booleani. Niente SL/TP/MM.
    """

    # --- Parametri (coerenti con la griglia WFO)
    cci_p   = params.get('cci_period')
    s1      = params.get('cci_shift_1')
    s2      = params.get('cci_shift_2')
    ema_p   = params.get('ema_period')
    kc_p    = params.get('kc_period')
    kc_mt   = params.get('kc_mult_tenths')
    kc_mult = kc_mt / 10.0

    df = data.copy()

    # --- (1) Calcolo indicatori SU TUTTO IL DF
    df['CCI_Change_Up'] = ind_gann_cci_kcb_cci_change_up(
        df, period=cci_p, shift_1=s1, shift_2=s2
    )

    df['Bull_Power'] = ind_gann_cci_kcb_bull_power(df, ema_period=ema_p)
    df['BullP_Rising'] = df['Bull_Power'] > df['Bull_Power'].shift(1)

    df['KC_Upper'] = ind_gann_cci_kcb_keltner_upper(
        df, kc_period=kc_p, kc_multiplier=kc_mult
    )
    df['KC_Open_Below_Upper'] = df['Open'] < df['KC_Upper']

    df['Gann_UpTrend'] = ind_gann_cci_kcb_gann_uptrend(df)

    # --- (2) Slicing per anno (OBBLIGATORIO)
    if year is not None:
        df = df[df.index.year == int(year)]

    # --- (3) Entry / Exit sul df filtrato
    entries = df['BullP_Rising'] & df['KC_Open_Below_Upper']
    exits   = df['Gann_UpTrend'] & df['CCI_Change_Up']

    # --- (4) Shift anti look-ahead + normalizzazione boolean
    entries = entries.shift(1).astype(bool).fillna(False)
    exits   = exits.shift(1).astype(bool).fillna(False)

    return entries, exits

############################
# Strategy kvo_zero_line
############################

# === Klinger Volume Oscillator ===
def ind_kvo_zero_line_kvo(
    df: pd.DataFrame,
    short: int = 34,
    long: int = 55,
    signal: int = 13
):
    """
    Klinger Volume Oscillator (KVO) + Signal line.
    Implementazione coerente con l'articolo.
    Ritorna: (kvo, kvo_signal)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    dm = high - low
    direction = np.where(close > close.shift(1), 1, -1)

    vf = volume * np.abs(2 * (close - low) / (high - low).replace(0, np.nan)) * direction
    vf = pd.Series(vf, index=df.index)

    kvo = vf.ewm(span=short, adjust=False, min_periods=1).mean() - \
          vf.ewm(span=long, adjust=False, min_periods=1).mean()

    kvo_signal = kvo.ewm(span=signal, adjust=False, min_periods=1).mean()

    return kvo, kvo_signal


# --- Griglia parametri per WF Optimization (contenuta) ---
strategy_kvo_zero_line_param_ranges = {
    # 'short_range'  : range(30, 41, 2),   # short EMA KVO
    'short_range'  : range(30, 51, 2),   # short EMA KVO
    'long_range'   : range(50, 71, 4),   # long EMA KVO
    'signal_range' : range(9, 21, 2)     # signal line
}


# --- Funzione di strategia ---
def strategy_kvo_zero_line(
    data: pd.DataFrame,
    params: dict,
    year: int | None = None
):
    """
    Strategia KVO Zero-Line Risk Control.

    Entry:
      - KVO_Signal.diff() > 0

    Exit:
      - KVO cross sotto lo zero

    Ordine (vincolante per K_Strategy):
      1) Indicatori su df completo
      2) Slicing per anno (se richiesto)
      3) entries/exits sul df filtrato
      4) shift di 1 barra + normalizzazione boolean
    """

    short  = params.get('short_range')
    long   = params.get('long_range')
    signal = params.get('signal_range')

    df = data.copy()

    # 1) Indicatori
    df['KVO'], df['KVO_Signal'] = ind_kvo_zero_line_kvo(df, short=short, long=long, signal=signal)

    # 2) Filtro anno (slicing)
    if year is not None:
        df = df[df.index.year == int(year)]

    # 3) entries/exits
    entries = df['KVO_Signal'].diff() > 0
    exits = (df['KVO'] < 0) & (df['KVO'].shift(1) >= 0)

    # 4) Shift + normalizzazione
    shifted_entries = entries.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

# ============================================================
# Strategy k_gold_macd_reflex
# ============================================================

# === Helpers ===
def ind_k_gold_macd_reflex_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder-like RMA: EMA con alpha=1/period."""
    period = int(period)
    return series.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()


# === REFLEX (delta -> RMA -> Fast/Slow) ===
def ind_k_gold_macd_reflex_reflex(
    df: pd.DataFrame,
    rsi_period: int = 14,
    fast_period: int = 5,
    slow_period: int = 13
) -> tuple[pd.Series, pd.Series]:
    """
    Variante "Reflex" come nel codice sorgente:
      - delta = Close.diff()
      - delta_rma = RMA(delta, rsi_period)
      - Fast_Reflex = RMA(delta_rma, fast_period)
      - Slow_Reflex = RMA(delta_rma, slow_period)  (calcolata, anche se la regola usa il fast)
    Ritorna: (fast_reflex, slow_reflex)
    """
    close = df["Close"]
    delta = close.diff()
    delta_rma = ind_k_gold_macd_reflex_rma(delta, rsi_period)

    fast_reflex = ind_k_gold_macd_reflex_rma(delta_rma, fast_period)
    slow_reflex = ind_k_gold_macd_reflex_rma(delta_rma, slow_period)
    return fast_reflex, slow_reflex


def ind_k_gold_macd_reflex_reflex_rising(
    df: pd.DataFrame,
    rsi_period: int = 14,
    fast_period: int = 5,
    slow_period: int = 13,
    shift: int = 5
) -> pd.Series:
    """
    Condizione entry "reflex rising" = Fast_Reflex > Fast_Reflex.shift(shift)
    """
    fast_reflex, _ = ind_k_gold_macd_reflex_reflex(df, rsi_period, fast_period, slow_period)
    return fast_reflex > fast_reflex.shift(int(shift))


# === MACD ===
def ind_k_gold_macd_reflex_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """
    MACD classico:
      MACD = EMA_fast - EMA_slow
      Signal = EMA(MACD, signal)
    Ritorna: (macd, macd_signal)
    """
    close = df["Close"]
    ema_fast = close.ewm(span=int(fast), adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=int(signal), adjust=False, min_periods=1).mean()
    return macd, macd_sig



# --- Griglia parametri per WF Optimization ---
strategy_k_gold_macd_reflex_param_ranges = {
    # MACD
    "macd_fast_range"    : range(10, 18, 4),   # [10,12,14]
    "macd_slow_range"    : range(22, 38, 8),   # [22,26,30]
    "macd_signal_range"  : range(7, 13, 2),    # [7,9,11]

    # REFLEX (nel codice: REFLEX_PERIOD = "rsi_period")
    "reflex_period_range": range(12, 18, 2),   # [12,14,16]
    "reflex_fast_range"  : range(4, 7),     # [4,5,6]
    "reflex_slow_range"  : range(11, 17, 2),   # [11,13,15]
    "reflex_shift_range" : range(4, 7),     # [4,5,6]
}


# --- Funzione di strategia ---
def strategy_k_gold_macd_reflex(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Strategia: Entry su "Reflex Rising", Exit su "MACD cross below 0".

    Regole (sul df completo; filtro per anno applicato SOLO come maschera a valle):
      Entry se:
        Fast_Reflex > Fast_Reflex.shift(reflex_shift)
      Exit se:
        MACD passa da >=0 a <0 (cross down dello zero)

    I segnali sono shiftati di 1 barra per evitare look-ahead.
    Il filtro year, se passato, MASCHERA i segnali mantenendo l’allineamento con l’indice originale.
    """

    # --- Leggi parametri (coerenti con le chiavi della griglia WFO)
    macd_f   = int(params.get("macd_fast_range"))
    macd_s   = int(params.get("macd_slow_range"))
    macd_sig = int(params.get("macd_signal_range"))

    rsi_p    = int(params.get("reflex_period_range"))
    r_fast   = int(params.get("reflex_fast_range"))
    r_slow   = int(params.get("reflex_slow_range"))
    r_shift  = int(params.get("reflex_shift_range"))

    df = data.copy()

    # --- Indicatori
    df["Reflex_Fast"], df["Reflex_Slow"] = ind_k_gold_macd_reflex_reflex(
        df, rsi_period=rsi_p, fast_period=r_fast, slow_period=r_slow
    )
    df["MACD"], df["MACD_Signal"] = ind_k_gold_macd_reflex_macd(
        df, fast=macd_f, slow=macd_s, signal=macd_sig
    )

    # --- Condizioni raw (su df completo)
    entries_raw = df["Reflex_Fast"] > df["Reflex_Fast"].shift(r_shift)
    exits_raw   = (df["MACD"].shift(1) >= 0) & (df["MACD"] < 0)

    # --- Maschera anno (dopo calcolo indicatori)
    if year is not None:
        mask = (df.index.year == int(year))
        entries_raw = entries_raw & mask
        exits_raw   = exits_raw & mask

    # --- Shift di 1 barra per backtest
    shifted_entries = entries_raw.shift(1).astype(bool).fillna(False)
    shifted_exits   = exits_raw.shift(1).astype(bool).fillna(False)

    return shifted_entries, shifted_exits

######################################
### Kripytera Claude
######################################

############################
# Strategy c_chande_elder_pressure_reversion
############################

# =============================================================
# Fonte: "C-Chande–Elder Pressure Reversion Strategy" — Kryptera
# https://medium.com/@Kryptera/c-chande-elder-pressure-reversion-strategy-b92d7fb77f38
#
# Logica:
#   ENTRY : CMO(14) < livello_oversold  (default -50)
#           → estremo oversold, segnale contrarian di acquisto
#   EXIT  : Bears Power (Low - EMA(13)) incrocia da sotto a sopra lo zero
#           → la pressione ribassista si esaurisce
# =============================================================


def ind_c_chande_elder_cmo(df, period: int = 14):
    """
    Chande Momentum Oscillator.
    CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down)
    dove sum_up  = somma rolling(period) delle variazioni positive
         sum_down = somma rolling(period) delle variazioni negative (in valore assoluto)
    Oscilla tra -100 e +100.
    """
    diff     = df['Close'].diff()
    sum_up   = diff.clip(lower=0).rolling(period, min_periods=period).sum()
    sum_down = (-diff.clip(upper=0)).rolling(period, min_periods=period).sum()
    denom    = (sum_up + sum_down).replace(0, float('nan'))
    cmo      = 100.0 * (sum_up - sum_down) / denom
    return cmo


def ind_c_chande_elder_bears_power(df, ema_period: int = 13):
    """
    Bears Power (Alexander Elder).
    Bears Power = Low - EMA(Close, ema_period)
    Valori negativi → bears dominano; crossover sopra 0 → bears si esauriscono.
    """
    ema         = df['Close'].ewm(span=ema_period, adjust=False).mean()
    bears_power = df['Low'] - ema
    return bears_power


# =============================================================
# Griglia parametri WFO
# Totale combinazioni: 6 * 5 * 5 * 4 = 600  (< 50k)
# =============================================================
strategy_c_chande_elder_pressure_reversion_param_ranges = {
    # periodo lookback CMO
    'cmo_period_range'      : range(8, 22, 2),       # 8,10,12,14,16,18,20  → 7 valori
    # livello oversold CMO (in decimi, da dividere per 1 — già interi)
    'cmo_oversold_range'    : range(-65, -40, 5),    # -65,-60,-55,-50,-45  → 5 valori
    # periodo EMA per Bears Power
    'bear_ema_period_range' : range(8, 22, 3),       # 8,11,14,17,20        → 5 valori
    # livello di crossover Bears Power (in decimi, /10.0)
    'bear_level_tenths'     : range(-5, 6, 5),       # -0.5, 0.0, 0.5, 1.0 → 4 valori? no: -5,0,5 → 3
    # (tot: 7*5*5*3 = 525)
}


def strategy_c_chande_elder_pressure_reversion(
    data,
    params: dict,
    year: int | None = None
):
    """
    Entry: CMO(cmo_period) < cmo_oversold
    Exit : Bears Power(bear_ema_period) incrocia sopra bear_level (default 0)
           ovvero: BP.shift(1) <= level  AND  BP > level
    """
    import pandas as pd
    import numpy as np

    cmo_period      = params.get('cmo_period_range',      14)
    cmo_oversold    = params.get('cmo_oversold_range',    -50)
    bear_ema_period = params.get('bear_ema_period_range', 13)
    bear_level      = params.get('bear_level_tenths',      0) / 10.0

    df = data.copy()

    # ── Calcolo indicatori sull'intero DataFrame ─────────────────────────
    df['CMO']         = ind_c_chande_elder_cmo(df, period=cmo_period)
    df['Bears_Power'] = ind_c_chande_elder_bears_power(df, ema_period=bear_ema_period)

    # ── Filtro anno (DOPO il calcolo degli indicatori) ────────────────────
    if year is not None:
        df = df[df.index.year == int(year)]

    # ── Segnali ───────────────────────────────────────────────────────────
    # Entry: CMO in zona oversold estrema
    entries = df['CMO'] < cmo_oversold

    # Exit: Bears Power incrocia sopra il livello (default 0)
    bear_cross_above = (
        (df['Bears_Power'].shift(1) <= bear_level) &
        (df['Bears_Power']          >  bear_level)
    )
    exits = bear_cross_above

    # ── Shift anti look-ahead ─────────────────────────────────────────────
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return shifted_entries, shifted_exits

############################
# Strategy lscc_ema_stc
############################

def ind_lscc_ema_stc_ema(df: pd.DataFrame, period: int = 16) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False).mean()


def ind_lscc_ema_stc_stc(df: pd.DataFrame,
                         fast_length: int = 20,
                         slow_length: int = 40,
                         cycle_period: int = 10) -> pd.Series:
    close = df['Close']
    stoch_fast = (close - close.rolling(window=fast_length, min_periods=1).min()) / \
                 (close.rolling(window=fast_length, min_periods=1).max() - close.rolling(window=fast_length, min_periods=1).min())
    stoch_slow = stoch_fast.ewm(span=cycle_period, adjust=False).mean()
    stc = 2 * (stoch_slow - 0.5) / (stoch_slow.max() - stoch_slow.min()) * 100
    return stc


strategy_lscc_ema_stc_param_ranges = {
    'ema_period_range': range(10, 31, 5),
    'stc_fast_length_range': range(10, 41, 5),
    'stc_slow_length_range': range(20, 61, 10),
    'stc_cycle_period_range': range(5, 21, 5)
}


def strategy_lscc_ema_stc(data: pd.DataFrame, params: dict, year: int | None = None):
    ema_p = params.get('ema_period_range')
    stc_fast_p = params.get('stc_fast_length_range')
    stc_slow_p = params.get('stc_slow_length_range')
    stc_cycle_p = params.get('stc_cycle_period_range')

    df = data.copy()

    ema = ind_lscc_ema_stc_ema(df, period=ema_p)
    stc = ind_lscc_ema_stc_stc(df, fast_length=stc_fast_p, slow_length=stc_slow_p, cycle_period=stc_cycle_p)

    df['EMA'] = ema
    df['STC'] = stc

    if year is not None:
        df = df[df.index.year == int(year)]

    entries_long = (df['Open'] < df['EMA']) & (df['STC'].shift(1) <= 0) & (df['STC'] > 0)
    entries_short = (df['Open'] > df['EMA']) & (df['STC'].shift(1) >= 0) & (df['STC'] < 0)

    exits_long = df['STC'].shift(1) > 0
    exits_short = df['STC'].shift(1) < 0

    shifted_entries = entries_long | entries_short
    shifted_exits = exits_long | exits_short

    return shifted_entries, shifted_exits

    
######################################
### Strategie per Oro
######################################

############################
## Strategy gold_donchian_ema_atr_v1
############################

# === Donchian Channels ===
def ind_gold_donchian_ema_atr_v1_donchian(df: pd.DataFrame, lookback: int = 55):
    high = df["High"].rolling(window=lookback, min_periods=lookback).max()
    low  = df["Low"].rolling(window=lookback,  min_periods=lookback).min()
    mid  = (high + low) / 2.0
    return high, low, mid

# === ATR (Wilder) ===
def ind_gold_donchian_ema_atr_v1_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(np.maximum((h - l), (h - pc).abs()), (l - pc).abs())
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

# === EMA (trend filter) ===
def ind_gold_donchian_ema_atr_v1_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["Close"].ewm(span=period, adjust=False).mean()

# --- Griglia parametri RIDOTTA ---
strategy_gold_donchian_ema_atr_v1_param_ranges = {
    'breakout_len' : [40, 55, 70],
    'exit_len'     : [15, 20],
    'ema_slow'     : [150, 200],
    'atr_period'   : [14],
    'atr_k'        : [1.5, 2.0],
}

# --- Funzione strategia ---
def strategy_gold_donchian_ema_atr_v1(data: pd.DataFrame, params: dict, year: int | None = None):
    """
    Long-only daily su oro (XAUUSD/GLD):
      - Filtro trend: Close > EMA_slow e EMA_slow in salita (slope su 5 barre)
      - Entry: Close odierno > DonchianUpper di IERI e ieri il Close era <= DonchianUpper di IERI
      - Exit : Close < DonchianLower di IERI  OR Close < (EMA_slow - atr_k*ATR) OR Close < EMA_slow
      - Segnali normalizzati con _safe_shift_fill_bool(…, 1) per l’esecuzione alla barra successiva.
    """
    df = data.copy()

    # Parametri
    n_up   = int(params.get('breakout_len'))
    n_dn   = int(params.get('exit_len'))
    ema_s  = int(params.get('ema_slow'))
    atr_p  = int(params.get('atr_period'))
    atr_k  = float(params.get('atr_k'))

    # Indicatori sull'intero dataset
    up, _, _ = ind_gold_donchian_ema_atr_v1_donchian(df, lookback=n_up)
    _, dn, _ = ind_gold_donchian_ema_atr_v1_donchian(df, lookback=n_dn)
    ema_slow = ind_gold_donchian_ema_atr_v1_ema(df, period=ema_s)
    atr      = ind_gold_donchian_ema_atr_v1_atr(df, period=atr_p)

    # Maschera anno (a valle)
    if year is not None:
        df       = df[df.index.year == int(year)]
        up       = up.loc[df.index]
        dn       = dn.loc[df.index]
        ema_slow = ema_slow.loc[df.index]
        atr      = atr.loc[df.index]

    # Livelli "no look-ahead": Donchian di ieri
    up_prev = up.shift(1)
    dn_prev = dn.shift(1)

    # Filtro trend
    slope_lb = 5
    trend_ok = (df["Close"] > ema_slow) & (ema_slow > ema_slow.shift(slope_lb))

    # Entry: breakout sul livello di ieri + trend ok
    raw_entries = (df["Close"] > up_prev) & (df["Close"].shift(1) <= up_prev) & trend_ok

    # Exit: lower di ieri OR ATR-stop su EMA_slow OR perdita regime
    stop_line = ema_slow - atr_k * atr
    raw_exits = (df["Close"] < dn_prev) | (df["Close"] < stop_line) | (df["Close"] < ema_slow)

    return _safe_shift_fill_bool(raw_entries, 1), _safe_shift_fill_bool(raw_exits, 1)




