# k_strategies_agent.py
# Strategie K generate automaticamente dall'agente K-strategy.
# NON modificare manualmente — file gestito dall'agente.
# Per strategie manuali: k_strategies.py
#
# Strategie presenti (22):
#   cfg_volume_contraction_bb, ko_trend_strength_bollinger_reversion,
#   cms_keltner_srpr, cdns_bbmacd, cme_qqe_bears, pstg_roc_vidya,
#   squeeze_momentum, aee_qke, adbe_efficiency_momentum, smci_momentum_vidya,
#   connors_bollinger_reversion, tsi_stc_momentum, hma_bb_crossover,
#   bkr_er_rsi, cdns_bollinger_macd, trmb_sma_wma, msi_bb_qi,
#   cop_demarker_supertrend, are_dem_kc, cag_hma_upturn_bollinger,
#   mpwr_bb_std, bb_expansion

import pandas as pd
import numpy as np


############################
# Strategy cfg_volume_contraction_bb
############################

def ind_cfg_volume_contraction_bb_bollinger(df: pd.DataFrame,
                                            bb_period: int = 20,
                                            bb_std: float = 2.0) -> tuple:
    close = df['Close']
    bb_ma = close.rolling(window=bb_period, min_periods=1).mean()
    bb_std_dev = close.rolling(window=bb_period, min_periods=1).std(ddof=0)
    bb_upper = bb_ma + bb_std * bb_std_dev
    bb_lower = bb_ma - bb_std * bb_std_dev
    return bb_ma, bb_upper, bb_lower


def ind_cfg_volume_contraction_bb_volume_falling(df: pd.DataFrame,
                                                  volume_shift: int = 1) -> pd.Series:
    volume_falling = df['Volume'] < df['Volume'].shift(volume_shift)
    return volume_falling


def ind_cfg_volume_contraction_bb_bb_close_below_lower(df: pd.DataFrame,
                                                        bb_lower: pd.Series) -> pd.Series:
    bb_close_below_lower = df['Close'] < bb_lower
    return bb_close_below_lower


strategy_cfg_volume_contraction_bb_param_ranges = {
    'bb_period_range': range(15, 31, 5),
    'bb_std_range': range(15, 25, 2),
    'volume_shift_range': range(1, 4, 1),
}


def strategy_cfg_volume_contraction_bb(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period = params.get('bb_period_range', 20)
    bb_std = params.get('bb_std_range', 20) / 10.0
    volume_shift = params.get('volume_shift_range', 1)

    df = data.copy()

    bb_ma, bb_upper, bb_lower = ind_cfg_volume_contraction_bb_bollinger(
        df, bb_period=bb_period, bb_std=bb_std
    )

    volume_falling = ind_cfg_volume_contraction_bb_volume_falling(
        df, volume_shift=volume_shift
    )

    bb_close_below_lower = ind_cfg_volume_contraction_bb_bb_close_below_lower(
        df, bb_lower=bb_lower
    )

    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['Volume_Falling'] = volume_falling
    df['BB_Close_Below_Lower'] = bb_close_below_lower

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['Volume_Falling'] & ~df['BB_Close_Below_Lower']
    exits = df['BB_Close_Below_Lower']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return shifted_entries, shifted_exits


############################
# Strategy ko_trend_strength_bollinger_reversion
############################



def ind_ko_trend_strength_bollinger_reversion_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    h_l = high - low
    h_pc = np.abs(high - close.shift(1))
    l_pc = np.abs(low - close.shift(1))
    tr = np.maximum(np.maximum(h_l, h_pc), l_pc)
    
    plus_dm = np.where(
        (high - high.shift(1)) > (low.shift(1) - low),
        np.maximum(high - high.shift(1), 0),
        0
    )
    minus_dm = np.where(
        (low.shift(1) - low) > (high - high.shift(1)),
        np.maximum(low.shift(1) - low, 0),
        0
    )
    
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=1).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=1).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=1).sum().values
    
    safe_tr = np.where(tr_sum != 0, tr_sum, 1.0)
    plus_di = np.where(tr_sum != 0, 100.0 * plus_dm_sum / safe_tr, 0.0)
    minus_di = np.where(tr_sum != 0, 100.0 * minus_dm_sum / safe_tr, 0.0)
    
    di_sum = plus_di + minus_di
    safe_di_sum = np.where(di_sum != 0, di_sum, 1.0)
    dx = np.where(di_sum != 0, 100.0 * np.abs(plus_di - minus_di) / safe_di_sum, 0.0)
    
    adx = pd.Series(dx).rolling(window=period, min_periods=1).mean().values
    
    return pd.Series(adx, index=df.index)


def ind_ko_trend_strength_bollinger_reversion_bb(df: pd.DataFrame, period: int = 20, std_multiplier: float = 2.0):
    close = df['Close']
    
    bb_ma = close.rolling(window=period, min_periods=1).mean()
    bb_std = close.rolling(window=period, min_periods=1).std(ddof=1)
    bb_std = bb_std.fillna(0)
    
    bb_upper = bb_ma + std_multiplier * bb_std
    bb_lower = bb_ma - std_multiplier * bb_std
    
    return bb_ma, bb_upper, bb_lower


strategy_ko_trend_strength_bollinger_reversion_param_ranges = {
    'adx_period_range': range(10, 21, 2),
    'adx_level_range': range(15, 31, 5),
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
    'bb_shift_range': range(3, 8, 1),
}


def strategy_ko_trend_strength_bollinger_reversion(data: pd.DataFrame, params: dict, year: int | None = None):
    adx_period = params.get('adx_period_range', 14)
    adx_level = params.get('adx_level_range', 20)
    bb_period = params.get('bb_period_range', 20)
    bb_std = params.get('bb_std_range', 2.0) / 10.0
    bb_shift = params.get('bb_shift_range', 5)
    
    df = data.copy()
    
    adx = ind_ko_trend_strength_bollinger_reversion_adx(df, period=adx_period)
    bb_ma, bb_upper, bb_lower = ind_ko_trend_strength_bollinger_reversion_bb(df, period=bb_period, std_multiplier=bb_std)
    
    df['ADX'] = adx
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    adx_higher_than_level = df['ADX'] > adx_level
    bb_upper_rising = df['BB_Upper'] > df['BB_Upper'].shift(bb_shift)
    
    entries = adx_higher_than_level
    exits = bb_upper_rising
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    
    return shifted_entries, shifted_exits


############################
# Strategy cms_keltner_srpr
############################

def ind_cms_keltner_srpr_keltner_channel(df: pd.DataFrame,
                                         period: int = 20,
                                         multiplier: float = 2.0) -> tuple:
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    tp = (high + low + close) / 3
    kc_mid = tp.ewm(span=period, adjust=False, min_periods=1).mean()
    
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(hl, hpc), lpc)
    atr = tr.rolling(period, min_periods=1).mean()
    
    kc_upper = kc_mid + multiplier * atr
    kc_lower = kc_mid - multiplier * atr
    
    return kc_mid, kc_upper, kc_lower


def ind_cms_keltner_srpr_srpr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    srpr = close.rolling(period, min_periods=1).apply(
        lambda x: np.sum(x <= x[-1]) / len(x) * 100,
        raw=True
    )
    return srpr


strategy_cms_keltner_srpr_param_ranges = {
    'kc_period_range': range(15, 31, 5),
    'kc_multiplier_range': range(15, 26, 5),
    'srpr_period_range': range(10, 21, 5),
    'srpr_level_range': range(60, 80, 5),
    'srpr_xbars_range': range(3, 8, 1),
}


def strategy_cms_keltner_srpr(data: pd.DataFrame, params: dict, year: int | None = None):
    kc_period = params.get('kc_period_range', 20)
    kc_multiplier = params.get('kc_multiplier_range', 20) / 10.0
    srpr_period = params.get('srpr_period_range', 14)
    srpr_level = params.get('srpr_level_range', 70)
    srpr_xbars = params.get('srpr_xbars_range', 5)
    
    df = data.copy()
    
    kc_mid, kc_upper, kc_lower = ind_cms_keltner_srpr_keltner_channel(
        df, period=kc_period, multiplier=kc_multiplier
    )
    srpr = ind_cms_keltner_srpr_srpr(df, period=srpr_period)
    
    df['KC_Mid'] = kc_mid
    df['KC_Upper'] = kc_upper
    df['KC_Lower'] = kc_lower
    df['SRPR'] = srpr
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    # Entry: Close above KC_Lower
    entries = df['Close'] > df['KC_Lower']
    
    # Exit: SRPR below level for X consecutive bars
    srpr_below = df['SRPR'] < srpr_level
    exit_condition = srpr_below.rolling(srpr_xbars, min_periods=1).sum() == srpr_xbars
    exits = exit_condition
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    
    return shifted_entries, shifted_exits


############################
# Strategy cdns_bbmacd
############################

def ind_cdns_bbmacd_bollinger_bands(df: pd.DataFrame,
                                    period: int = 20,
                                    std_multiplier: float = 2.0) -> tuple:
    close = df['Close']
    bb_ma = close.rolling(window=period, min_periods=1).mean()
    bb_std = close.rolling(window=period, min_periods=1).std(ddof=0)
    bb_upper = bb_ma + std_multiplier * bb_std
    bb_lower = bb_ma - std_multiplier * bb_std
    return bb_ma, bb_upper, bb_lower


def ind_cdns_bbmacd_macd(df: pd.DataFrame,
                         fast: int = 12,
                         slow: int = 26,
                         signal: int = 9) -> tuple:
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    macd_prev = macd.shift(1)
    signal_prev = macd_signal.shift(1)
    return macd, macd_signal, macd_prev, signal_prev


strategy_cdns_bbmacd_param_ranges = {
    'bb_period_range': range(16, 25, 4),
    'bb_shift_range': range(4, 7, 1),
    'bb_std_range': range(15, 26, 5),
    'macd_fast_range': range(10, 15, 2),
    'macd_slow_range': range(22, 31, 4),
    'macd_signal_range': range(7, 12, 2),
}


def strategy_cdns_bbmacd(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period = params.get('bb_period_range', 20)
    bb_shift = params.get('bb_shift_range', 5)
    bb_std = params.get('bb_std_range', 20) / 10.0
    macd_fast = params.get('macd_fast_range', 12)
    macd_slow = params.get('macd_slow_range', 26)
    macd_signal = params.get('macd_signal_range', 9)

    df = data.copy()

    bb_ma, bb_upper, bb_lower = ind_cdns_bbmacd_bollinger_bands(
        df, period=bb_period, std_multiplier=bb_std
    )
    macd, macd_signal_line, macd_prev, signal_prev = ind_cdns_bbmacd_macd(
        df, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )

    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['MACD'] = macd
    df['Signal'] = macd_signal_line
    df['MACD_prev'] = macd_prev
    df['Signal_prev'] = signal_prev

    if year is not None:
        df = df[df.index.year == int(year)]

    bb_upper_falling = df['BB_Upper'] < df['BB_Upper'].shift(bb_shift)
    macd_pullback_reversal_up = (
        (df['MACD_prev'] < df['Signal_prev']) &
        (df['MACD'] > df['Signal']) &
        (df['MACD'] > 0)
    )

    entries = bb_upper_falling & macd_pullback_reversal_up
    exits = ~macd_pullback_reversal_up

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)

    return shifted_entries, shifted_exits


############################
# Strategy cme_qqe_bears
############################

def ind_cme_qqe_bears_qqe(df: pd.DataFrame,
                          rsi_period: int = 14,
                          smooth: int = 5,
                          factor: float = 4.236) -> tuple:
    close = df['Close'].values
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    roll_up = pd.Series(up).ewm(alpha=1/rsi_period, adjust=False).mean().values
    roll_down = pd.Series(down).ewm(alpha=1/rsi_period, adjust=False).mean().values
    
    roll_up = np.where(roll_up == 0, 1e-10, roll_up)
    roll_down = np.where(roll_down == 0, 1e-10, roll_down)
    
    rsi = 100 - (100 / (1 + roll_up / roll_down))
    rsi_series = pd.Series(rsi, index=df.index)
    
    rsi_ma = rsi_series.rolling(window=smooth, min_periods=1).mean()
    rsi_delta = rsi_ma.diff().abs().fillna(0)
    atr_rsi = pd.Series(rsi_delta).ewm(alpha=1/smooth, adjust=False).mean()
    
    value1 = rsi_ma.copy()
    v1_arr = rsi_ma.values
    atr_arr = atr_rsi.values
    v2_arr = np.empty(len(v1_arr))
    v2_arr[0] = 0.0
    for i in range(1, len(v2_arr)):
        direction = 1 if rsi[i] > v1_arr[i] else -1
        v2_arr[i] = v2_arr[i - 1] + direction * factor * atr_arr[i]
    value2 = pd.Series(v2_arr, index=df.index)
    return value1, value2


def ind_cme_qqe_bears_bears_power(df: pd.DataFrame,
                                   ema_period: int = 13) -> pd.Series:
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    bears_power = df['Low'] - ema
    return bears_power


strategy_cme_qqe_bears_param_ranges = {
    'rsi_period_range': range(10, 21, 5),
    'smooth_range': range(3, 8, 2),
    'factor_range': range(3, 6, 1),
    'qqe_level_range': range(40, 61, 10),
    'bear_power_level_range': range(-5, 6, 5),
}


def strategy_cme_qqe_bears(data: pd.DataFrame, params: dict, year: int | None = None):
    rsi_period = params.get('rsi_period_range', 14)
    smooth = params.get('smooth_range', 5)
    factor = params.get('factor_range', 4) * 1.236
    qqe_level = params.get('qqe_level_range', 50)
    bear_power_level = params.get('bear_power_level_range', 0)
    
    df = data.copy()
    
    qqe_value1, qqe_value2 = ind_cme_qqe_bears_qqe(
        df,
        rsi_period=rsi_period,
        smooth=smooth,
        factor=factor
    )
    
    bears_power = ind_cme_qqe_bears_bears_power(df, ema_period=13)
    
    df['QQE_Value1'] = qqe_value1
    df['QQE_Value2'] = qqe_value2
    df['Bears_Power'] = bears_power
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    qqe_above_level = df['QQE_Value1'] > qqe_level
    qqe_momentum = df['QQE_Value1'] > df['QQE_Value2']
    
    bears_power_above = df['Bears_Power'] > bear_power_level
    
    entries = qqe_above_level & qqe_momentum
    
    exits = bears_power_above
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    
    return shifted_entries, shifted_exits


############################
# Strategy pstg_roc_vidya
############################

def ind_pstg_roc_vidya_vidya(df: pd.DataFrame, period: int = 20, vol_period: int = 50) -> pd.Series:
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Volatility Index (CMO-based approximation)
    price_change = close.diff()
    up = price_change.where(price_change > 0, 0)
    down = -price_change.where(price_change < 0, 0)
    up_sum = up.rolling(period, min_periods=1).sum()
    down_sum = down.rolling(period, min_periods=1).sum()
    vi = (up_sum - down_sum).abs() / (up_sum + down_sum).replace(0, 1)
    
    # VIDYA calculation
    alpha = 2.0 / (period + 1)
    close_arr = close.values
    vi_arr = np.where(np.isnan(vi.values), 0.0, vi.values)
    vidya_arr = np.empty(len(close_arr))
    vidya_arr[0] = close_arr[0]
    for i in range(1, len(close_arr)):
        k = alpha * vi_arr[i]
        vidya_arr[i] = k * close_arr[i] + (1 - k) * vidya_arr[i - 1]
    
    return pd.Series(vidya_arr, index=close.index)

def ind_pstg_roc_vidya_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    close = df['Close']
    roc = close.pct_change(period) * 100
    return roc

strategy_pstg_roc_vidya_param_ranges = {
    'vidya_period_range': range(15, 31, 5),
    'vidya_vol_period_range': range(40, 61, 10),
    'roc_period_range': range(8, 13, 2),
    'roc_diff_threshold_range': range(-3, 2, 1),
}

def strategy_pstg_roc_vidya(data: pd.DataFrame, params: dict, year: int | None = None):
    vidya_period = params.get('vidya_period_range')
    vidya_vol_period = params.get('vidya_vol_period_range')
    roc_period = params.get('roc_period_range')
    roc_diff_threshold = params.get('roc_diff_threshold_range')
    
    df = data.copy()
    
    # Calculate indicators
    vidya = ind_pstg_roc_vidya_vidya(df, period=vidya_period, vol_period=vidya_vol_period)
    roc = ind_pstg_roc_vidya_roc(df, period=roc_period)
    roc_diff = roc.diff()
    
    df['VIDYA'] = vidya
    df['ROC'] = roc
    df['ROC_diff'] = roc_diff
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    # Entry: opening price drops below VIDYA (mean reversion pullback)
    entries = df['Open'] < df['VIDYA']
    
    # Exit: ROC momentum flattens (rate of change in ROC below threshold)
    exits = df['ROC_diff'] <= roc_diff_threshold
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy squeeze_momentum
############################

def ind_squeeze_momentum_smi(df: pd.DataFrame, 
                            length_bb: int = 20, 
                            mult_bb: float = 2.0, 
                            mult_kc: float = 1.5):
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    basis = close.rolling(length_bb, min_periods=1).mean()
    dev = close.rolling(length_bb, min_periods=1).std(ddof=0)
    upper_bb = basis + mult_bb * dev
    lower_bb = basis - mult_bb * dev
    
    rangee = (high - low).ewm(span=length_bb, adjust=False, min_periods=1).mean()
    upper_kc = basis + mult_kc * rangee
    lower_kc = basis - mult_kc * rangee
    
    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    
    delta = close - close.shift(length_bb)
    
    delta_arr = delta.values
    mom_arr = np.full(len(delta_arr), np.nan)
    _x = np.arange(length_bb)
    _xc = _x - np.mean(_x)
    _xvar = np.dot(_xc, _xc)
    _xmean = (length_bb - 1) / 2.0
    for i in range(length_bb - 1, len(delta_arr)):
        y = delta_arr[i - length_bb + 1:i + 1]
        if not np.any(np.isnan(y)):
            slope = np.dot(_xc, y) / _xvar
            mom_arr[i] = np.mean(y) + slope * _xmean
    momentum = pd.Series(mom_arr, index=df.index)
    return squeeze_on, squeeze_off, momentum

strategy_squeeze_momentum_param_ranges = {
    'length_bb_range': range(10, 31, 5),
    'mult_bb_range': range(15, 26, 5),
    'mult_kc_range': range(10, 21, 5),
}

def strategy_squeeze_momentum(data: pd.DataFrame, params: dict, year: int | None = None):
    length_bb = params.get('length_bb_range')
    mult_bb = params.get('mult_bb_range') / 10.0
    mult_kc = params.get('mult_kc_range') / 10.0
    
    df = data.copy()
    
    squeeze_on, squeeze_off, momentum = ind_squeeze_momentum_smi(
        df, length_bb=length_bb, mult_bb=mult_bb, mult_kc=mult_kc
    )
    
    df['squeeze_on'] = squeeze_on
    df['squeeze_off'] = squeeze_off
    df['momentum'] = momentum
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    entries = df['squeeze_off']
    exits = df['squeeze_on']
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy aee_qke
############################

def ind_aee_qke_qqe(df: pd.DataFrame, rsi_period: int = 14, smooth: int = 5, factor: float = 4.236):
    close = df['Close']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/rsi_period, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1/rsi_period, adjust=False, min_periods=1).mean()
    rsi = 100 - (100 / (1 + roll_up / roll_down.replace(0, np.nan)))
    rsi_ma = rsi.rolling(window=smooth, min_periods=1).mean()
    rsi_delta = rsi_ma.diff().abs().fillna(0)
    atr_rsi = rsi_delta.ewm(alpha=1/smooth, adjust=False, min_periods=1).mean()
    value1 = rsi_ma.copy()
    v1_arr = value1.values
    atr_arr = atr_rsi.values
    v2_arr = np.empty(len(v1_arr))
    v2_arr[0] = v1_arr[0] if len(v1_arr) > 0 else 0.0
    for i in range(1, len(v2_arr)):
        direction = 1 if v1_arr[i - 1] > v2_arr[i - 1] else -1
        v2_arr[i] = v2_arr[i - 1] + direction * factor * atr_arr[i]
    value2 = pd.Series(v2_arr, index=df.index)
    return value1, value2

def ind_aee_qke_kvo(df: pd.DataFrame, short: int = 34, long: int = 55, signal: int = 13):
    high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
    dm = high - low
    dm = dm.replace(0, np.nan)
    cm = np.cumsum(np.where(close > close.shift(1), dm, -dm))
    hl_diff = high - low
    hl_diff = hl_diff.replace(0, np.nan)
    vf = volume * np.abs(2 * (close - low) / hl_diff - 1)
    vf = vf.fillna(0)
    kvo = vf.ewm(span=short, adjust=False, min_periods=1).mean() - vf.ewm(span=long, adjust=False, min_periods=1).mean()
    kvo_signal = kvo.ewm(span=signal, adjust=False, min_periods=1).mean()
    return kvo, kvo_signal

strategy_aee_qke_param_ranges = {
    'qqe_period_range': range(10, 21, 5),
    'qqe_smooth_range': range(3, 8, 2),
    'qqe_factor_range': range(35, 56, 10),
    'qqe_shift_range': range(3, 8, 2),
    'kvo_short_range': range(25, 46, 20),
    'kvo_long_range': range(45, 66, 20),
    'kvo_signal_range': range(10, 18, 4),
}

def strategy_aee_qke(data: pd.DataFrame, params: dict, year: int | None = None):
    qqe_period = params.get('qqe_period_range')
    qqe_smooth = params.get('qqe_smooth_range')
    qqe_factor = params.get('qqe_factor_range') / 10.0
    qqe_shift = params.get('qqe_shift_range')
    kvo_short = params.get('kvo_short_range')
    kvo_long = params.get('kvo_long_range')
    kvo_signal = params.get('kvo_signal_range')

    df = data.copy()

    qqe_value1, qqe_value2 = ind_aee_qke_qqe(df, rsi_period=qqe_period, smooth=qqe_smooth, factor=qqe_factor)
    kvo, kvo_signal_line = ind_aee_qke_kvo(df, short=kvo_short, long=kvo_long, signal=kvo_signal)

    df['QQE_Value1'] = qqe_value1
    df['QQE_Value2'] = qqe_value2
    df['KVO'] = kvo
    df['KVO_Signal'] = kvo_signal_line

    if year is not None:
        df = df[df.index.year == int(year)]

    qqe_rising = df['QQE_Value1'] > df['QQE_Value1'].shift(qqe_shift)
    kvo_cross_above = (df['KVO_Signal'] > 0) & (df['KVO_Signal'].shift(1) <= 0)

    entries = qqe_rising
    exits = kvo_cross_above

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy adbe_efficiency_momentum
############################

def ind_adbe_efficiency_momentum_kaufman_er(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    return er

def ind_adbe_efficiency_momentum_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    return macd, signal_line

strategy_adbe_efficiency_momentum_param_ranges = {
    'er_period_range': range(10, 21, 5),
    'er_level_range': range(25, 40, 5),
    'macd_fast_range': range(8, 16, 4),
    'macd_slow_range': range(20, 32, 6),
    'macd_signal_range': range(6, 12, 2),
    'macd_level_range': range(5, 20, 5),
}

def strategy_adbe_efficiency_momentum(data: pd.DataFrame, params: dict, year: int | None = None):
    er_period = params.get('er_period_range')
    er_level = params.get('er_level_range') / 100.0
    macd_fast = params.get('macd_fast_range')
    macd_slow = params.get('macd_slow_range')
    macd_signal = params.get('macd_signal_range')
    macd_level = params.get('macd_level_range') / 10.0

    df = data.copy()

    er = ind_adbe_efficiency_momentum_kaufman_er(df, period=er_period)
    macd, signal_line = ind_adbe_efficiency_momentum_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    df['ER'] = er
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line

    if year is not None:
        df = df[df.index.year == int(year)]

    er_low = df['ER'] < er_level
    macd_high = df['MACD'] > macd_level

    entries = er_low
    exits = macd_high

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy smci_momentum_vidya
############################

def ind_smci_momentum_vidya_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    close = df['Close']
    momentum = close / close.shift(period) * 100
    return momentum

def ind_smci_momentum_vidya_momentum_trend_down(df: pd.DataFrame, length: int = 3, period: int = 10) -> pd.Series:
    momentum = ind_smci_momentum_vidya_momentum(df, period)
    momentum_diff = momentum.diff()
    trend_down = momentum_diff.rolling(length, min_periods=1).apply(lambda x: (x < 0).all() if len(x) == length else False, raw=True)
    return trend_down.fillna(False)

def ind_smci_momentum_vidya_vidya(df: pd.DataFrame, period: int = 14, alpha: float = 0.2) -> pd.Series:
    close = df['Close'].values
    vidya = np.full(len(close), np.nan)
    vidya[0] = close[0]
    
    base_vol = np.nanmean(np.abs(np.diff(close)))
    
    for i in range(1, len(close)):
        window_start = max(0, i - period)
        window = close[window_start:i + 1]
        vol = np.nanmean(np.abs(np.diff(window)))
        vol_ratio = np.clip(vol / base_vol if base_vol != 0 else 1, 0.2, 2)
        a = alpha * vol_ratio
        vidya[i] = a * close[i] + (1 - a) * vidya[i - 1]
    
    return pd.Series(vidya, index=df.index)

def ind_smci_momentum_vidya_vidya_rising(df: pd.DataFrame, shift: int = 5, period: int = 14, alpha: float = 0.2) -> pd.Series:
    vidya = ind_smci_momentum_vidya_vidya(df, period, alpha)
    vidya_rising = vidya > vidya.shift(shift)
    return vidya_rising

strategy_smci_momentum_vidya_param_ranges = {
    'momentum_length_range': range(2, 5, 1),
    'momentum_period_range': range(8, 13, 1),
    'vidya_alpha_range': range(15, 26, 5),
    'vidya_period_range': range(12, 17, 1),
    'vidya_shift_range': range(4, 7, 1)
}

def strategy_smci_momentum_vidya(data: pd.DataFrame, params: dict, year: int | None = None):
    momentum_length = params.get('momentum_length_range')
    momentum_period = params.get('momentum_period_range')
    vidya_alpha = params.get('vidya_alpha_range') / 100.0
    vidya_period = params.get('vidya_period_range')
    vidya_shift = params.get('vidya_shift_range')
    
    df = data.copy()
    
    momentum_trend_down = ind_smci_momentum_vidya_momentum_trend_down(df, momentum_length, momentum_period)
    vidya_rising = ind_smci_momentum_vidya_vidya_rising(df, vidya_shift, vidya_period, vidya_alpha)
    
    if year is not None:
        df = df[df.index.year == int(year)]
        momentum_trend_down = momentum_trend_down[df.index]
        vidya_rising = vidya_rising[df.index]
    
    entries = momentum_trend_down
    exits = vidya_rising
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    
    return shifted_entries, shifted_exits


############################
# Strategy connors_bollinger_reversion
############################

def ind_connors_bollinger_reversion_bollinger_bands(df, period=20, std_multiplier=2.0):
    close = df['Close']
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    bb_upper = ma + std_multiplier * std
    bb_lower = ma - std_multiplier * std
    return ma, bb_upper, bb_lower

def ind_connors_bollinger_reversion_connors_rsi(df, rsi_period=3, streak_period=2, rank_period=100):
    close = df['Close']
    delta = close.diff()
    
    # Classic RSI
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    safe_loss = np.where(avg_loss != 0, avg_loss, 1e-9)
    rs = avg_gain / safe_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Streak RSI
    sign_diff = np.sign(delta)
    streak_groups = (sign_diff != sign_diff.shift()).cumsum()
    streak = sign_diff.groupby(streak_groups).cumsum()
    
    # Calculate streak RSI using numpy operations
    streak_vals = streak.values
    streak_rsi = np.empty(len(streak_vals))
    streak_rsi[:] = np.nan
    
    for i in range(streak_period - 1, len(streak_vals)):
        window = streak_vals[max(0, i - streak_period + 1):i + 1]
        pos_sum = np.sum(window[window > 0])
        neg_sum = -np.sum(window[window < 0])
        safe_neg = neg_sum if neg_sum != 0 else 1e-9
        streak_rs = pos_sum / safe_neg
        streak_rsi[i] = 100 - (100 / (1 + streak_rs))
    
    streak_rsi_series = pd.Series(streak_rsi, index=close.index)
    
    # Percent Rank
    roc = close.pct_change()
    rank = roc.rolling(rank_period, min_periods=1).apply(
        lambda x: np.sum(x <= x[-1]) / len(x) * 100, 
        raw=True
    )
    
    # Combine all three
    connors_rsi = (rsi + streak_rsi_series + rank) / 3
    return connors_rsi

strategy_connors_bollinger_reversion_param_ranges = {
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
    'rsi_period_range': range(2, 6, 1),
    'crsi_oversold_range': range(25, 36, 5),
}

def strategy_connors_bollinger_reversion(data, params, year=None):
    bb_period = params.get('bb_period_range')
    bb_std = params.get('bb_std_range') / 10.0
    rsi_period = params.get('rsi_period_range')
    crsi_oversold = params.get('crsi_oversold_range')
    
    df = data.copy()
    
    bb_ma, bb_upper, bb_lower = ind_connors_bollinger_reversion_bollinger_bands(
        df, period=bb_period, std_multiplier=bb_std
    )
    
    connors_rsi = ind_connors_bollinger_reversion_connors_rsi(
        df, rsi_period=rsi_period, streak_period=2, rank_period=100
    )
    
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['ConnorsRSI'] = connors_rsi
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    # Entry: Open below BB lower AND ConnorsRSI oversold
    open_below_lower = df['Open'] < df['BB_Lower']
    crsi_oversold_signal = df['ConnorsRSI'] < crsi_oversold
    entries = open_below_lower & crsi_oversold_signal
    
    # Exit: Open below BB lower (next occurrence)
    exits = open_below_lower
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy tsi_stc_momentum
############################

def ind_tsi_stc_momentum_tsi(df: pd.DataFrame, long_period: int = 25, short_period: int = 13) -> pd.Series:
    close = df['Close']
    momentum = close.diff()
    abs_momentum = momentum.abs()
    
    # Double smoothed momentum
    first_smooth_mom = momentum.ewm(span=long_period, adjust=False, min_periods=1).mean()
    double_smooth_mom = first_smooth_mom.ewm(span=short_period, adjust=False, min_periods=1).mean()
    
    # Double smoothed absolute momentum
    first_smooth_abs = abs_momentum.ewm(span=long_period, adjust=False, min_periods=1).mean()
    double_smooth_abs = first_smooth_abs.ewm(span=short_period, adjust=False, min_periods=1).mean()
    
    safe_den = np.where(double_smooth_abs.values != 0, double_smooth_abs.values, 1.0)
    tsi = np.where(double_smooth_abs.values != 0, 100 * double_smooth_mom.values / safe_den, 0.0)
    
    return pd.Series(tsi, index=df.index)

def ind_tsi_stc_momentum_stc(df: pd.DataFrame, period: int = 10, fast_d: int = 3, slow_d: int = 3) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate STC (Schaff Trend Cycle)
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    
    safe_range = np.where((highest - lowest).values != 0, (highest - lowest).values, 1.0)
    pct_k = np.where((highest - lowest).values != 0, 100 * (close - lowest).values / safe_range, 50.0)
    
    pct_k_series = pd.Series(pct_k, index=df.index)
    pct_d = pct_k_series.ewm(span=fast_d, adjust=False, min_periods=1).mean()
    
    highest_d = pct_d.rolling(period, min_periods=1).max()
    lowest_d = pct_d.rolling(period, min_periods=1).min()
    
    safe_range_d = np.where((highest_d - lowest_d).values != 0, (highest_d - lowest_d).values, 1.0)
    pct_kd = np.where((highest_d - lowest_d).values != 0, 100 * (pct_d - lowest_d).values / safe_range_d, 50.0)
    
    pct_kd_series = pd.Series(pct_kd, index=df.index)
    stc = pct_kd_series.ewm(span=slow_d, adjust=False, min_periods=1).mean()
    
    return stc

strategy_tsi_stc_momentum_param_ranges = {
    'tsi_long_range': range(20, 31, 5),
    'tsi_short_range': range(10, 16, 2),
    'stc_period_range': range(8, 13, 2),
    'stc_threshold_range': range(20, 31, 5)
}

def strategy_tsi_stc_momentum(data: pd.DataFrame, params: dict, year: int | None = None):
    tsi_long_p = params.get('tsi_long_range')
    tsi_short_p = params.get('tsi_short_range')
    stc_period_p = params.get('stc_period_range')
    stc_threshold_p = params.get('stc_threshold_range')
    
    df = data.copy()
    
    tsi = ind_tsi_stc_momentum_tsi(df, long_period=tsi_long_p, short_period=tsi_short_p)
    stc = ind_tsi_stc_momentum_stc(df, period=stc_period_p, fast_d=3, slow_d=3)
    
    df['TSI'] = tsi
    df['STC'] = stc
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    entries = df['STC'] > stc_threshold_p
    exits = df['TSI'] < 0
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy hma_bb_crossover
############################

def ind_hma_bb_crossover_hma(df: pd.DataFrame, length: int) -> pd.Series:
    close = df['Close']
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = close.rolling(window=half_length, min_periods=1).mean()
    wma_full = close.rolling(window=length, min_periods=1).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_length, min_periods=1).mean()
    return hma

def ind_hma_bb_crossover_bb(df: pd.DataFrame, period: int, std_multiplier: float):
    close = df['Close']
    ma = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std(ddof=0)
    upper = ma + std_multiplier * std
    lower = ma - std_multiplier * std
    return ma, upper, lower

strategy_hma_bb_crossover_param_ranges = {
    'hma_fast_range': range(10, 31, 5),
    'hma_slow_range': range(40, 61, 10),
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
}

def strategy_hma_bb_crossover(data: pd.DataFrame, params: dict, year: int | None = None):
    hma_fast_p = params.get('hma_fast_range')
    hma_slow_p = params.get('hma_slow_range')
    bb_period_p = params.get('bb_period_range')
    bb_std_p = params.get('bb_std_range') / 10.0
    
    df = data.copy()
    
    hma_fast = ind_hma_bb_crossover_hma(df, hma_fast_p)
    hma_slow = ind_hma_bb_crossover_hma(df, hma_slow_p)
    bb_ma, bb_upper, bb_lower = ind_hma_bb_crossover_bb(df, bb_period_p, bb_std_p)
    
    df['HMA_Fast'] = hma_fast
    df['HMA_Slow'] = hma_slow
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    hma_fast_above_slow = df['HMA_Fast'] > df['HMA_Slow']
    hma_fast_below_slow = df['HMA_Fast'] < df['HMA_Slow']
    open_above_lower = df['Open'] > df['BB_Lower']
    open_above_upper = df['Open'] > df['BB_Upper']
    
    entries = hma_fast_above_slow & open_above_lower
    exits = hma_fast_below_slow & open_above_upper
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy bkr_er_rsi
############################

def ind_bkr_er_rsi_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    safe_avg_loss = np.where(avg_loss != 0, avg_loss, 1e-10)
    rs = avg_gain / safe_avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ind_bkr_er_rsi_efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
    close = df['Close']
    direction = np.abs(close - close.shift(period))
    volatility = np.abs(close.diff()).rolling(period, min_periods=1).sum()
    safe_volatility = np.where(volatility != 0, volatility, 1e-10)
    er = direction / safe_volatility
    return er

strategy_bkr_er_rsi_param_ranges = {
    'rsi_period_range': range(10, 21, 5),
    'rsi_oversold_range': range(25, 36, 5),
    'er_period_range': range(8, 16, 4),
    'er_exit_range': range(3, 9, 2)
}

def strategy_bkr_er_rsi(data: pd.DataFrame, params: dict, year: int | None = None):
    rsi_period = params.get('rsi_period_range')
    rsi_oversold = params.get('rsi_oversold_range')
    er_period = params.get('er_period_range')
    er_exit_threshold = params.get('er_exit_range') / 10.0
    
    df = data.copy()
    
    rsi = ind_bkr_er_rsi_rsi(df, period=rsi_period)
    er = ind_bkr_er_rsi_efficiency_ratio(df, period=er_period)
    
    df['RSI'] = rsi
    df['ER'] = er
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    rsi_oversold_condition = df['RSI'] < rsi_oversold
    rsi_confirmed_breakdown = (df['RSI'] < rsi_oversold) & (df['RSI'].shift(1) >= rsi_oversold)
    
    entries = rsi_confirmed_breakdown
    
    er_downtrend = df['ER'] > er_exit_threshold
    price_declining = df['Close'] < df['Close'].shift(1)
    exits = er_downtrend & price_declining
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy cdns_bollinger_macd
############################

def ind_cdns_bollinger_macd_bollinger_bands(df, period=20, std_multiplier=2.0):
    close = df['Close']
    bb_ma = close.rolling(period, min_periods=1).mean()
    bb_std = close.rolling(period, min_periods=1).std(ddof=0)
    bb_upper = bb_ma + std_multiplier * bb_std
    bb_lower = bb_ma - std_multiplier * bb_std
    return bb_ma, bb_upper, bb_lower

def ind_cdns_bollinger_macd_macd(df, fast=12, slow=26, signal=9):
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    return macd, macd_signal

strategy_cdns_bollinger_macd_param_ranges = {
    'bb_period_range': range(16, 25, 4),
    'bb_shift_range': range(4, 7, 1),
    'bb_std_range': range(15, 26, 5),
    'macd_fast_range': range(10, 15, 2),
    'macd_slow_range': range(22, 31, 4),
    'macd_signal_range': range(7, 12, 2),
}

def strategy_cdns_bollinger_macd(data, params, year=None):
    bb_period = params.get('bb_period_range')
    bb_shift = params.get('bb_shift_range')
    bb_std = params.get('bb_std_range') / 10.0
    macd_fast = params.get('macd_fast_range')
    macd_slow = params.get('macd_slow_range')
    macd_signal = params.get('macd_signal_range')
    
    df = data.copy()
    
    bb_ma, bb_upper, bb_lower = ind_cdns_bollinger_macd_bollinger_bands(df, bb_period, bb_std)
    macd, macd_sig = ind_cdns_bollinger_macd_macd(df, macd_fast, macd_slow, macd_signal)
    
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['MACD'] = macd
    df['MACD_Signal'] = macd_sig
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    bb_upper_falling = df['BB_Upper'] < df['BB_Upper'].shift(bb_shift)
    
    macd_prev = df['MACD'].shift(1)
    signal_prev = df['MACD_Signal'].shift(1)
    macd_pullback_up = (
        (macd_prev < signal_prev) &
        (df['MACD'] > df['MACD_Signal']) &
        (df['MACD'] > 0)
    )
    
    entries = bb_upper_falling
    exits = macd_pullback_up
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy trmb_sma_wma
############################

def ind_trmb_sma_wma_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df['Close'].rolling(period, min_periods=1).mean()

def ind_trmb_sma_wma_wma(df: pd.DataFrame, period: int) -> pd.Series:
    close = df['Close']
    weights = np.arange(1, period + 1)
    def wma_func(x):
        if len(x) < period:
            w = np.arange(1, len(x) + 1)
            return np.sum(x * w) / np.sum(w)
        return np.sum(x * weights) / np.sum(weights)
    return close.rolling(period, min_periods=1).apply(wma_func, raw=True)

strategy_trmb_sma_wma_param_ranges = {
    'sma_slow_range': range(16, 25, 4),
    'sma_fast_range': range(8, 13, 2),
    'wma_period_range': range(16, 25, 4),
    'wma_shift_range': range(4, 7, 1),
}

def strategy_trmb_sma_wma(data: pd.DataFrame, params: dict, year: int | None = None):
    sma_slow_p = params.get('sma_slow_range')
    sma_fast_p = params.get('sma_fast_range')
    wma_period_p = params.get('wma_period_range')
    wma_shift_p = params.get('wma_shift_range')
    
    df = data.copy()
    
    sma_slow = ind_trmb_sma_wma_sma(df, sma_slow_p)
    sma_fast = ind_trmb_sma_wma_sma(df, sma_fast_p)
    wma = ind_trmb_sma_wma_wma(df, wma_period_p)
    
    df['SMA_slow'] = sma_slow
    df['SMA_fast'] = sma_fast
    df['WMA'] = wma
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    wma_falling = df['WMA'] < df['WMA'].shift(wma_shift_p)
    sma_cross_down = (df['SMA_fast'] < df['SMA_slow']) & (df['SMA_fast'].shift(1) >= df['SMA_slow'].shift(1))
    
    entries = wma_falling
    exits = sma_cross_down
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy msi_bb_qi
############################

def ind_msi_bb_qi_bb_squeeze(df: pd.DataFrame, period: int = 20, lookback: int = 50) -> pd.Series:
    close = df['Close']
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper = ma + 2 * std
    lower = ma - 2 * std
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    min_bandwidth = bandwidth.rolling(lookback, min_periods=1).min()
    extreme_squeeze = (bandwidth <= min_bandwidth)
    return extreme_squeeze

def ind_msi_bb_qi_quantile_interrange(df: pd.DataFrame, period: int = 20, zscore_period: int = 20) -> pd.Series:
    close = df['Close']
    returns = close.pct_change()
    
    def calc_qi(x):
        if len(x) < 5:
            return 0.0
        sorted_x = np.sort(x[~np.isnan(x)])
        if len(sorted_x) < 5:
            return 0.0
        q95 = np.percentile(sorted_x, 95)
        q5 = np.percentile(sorted_x, 5)
        return q95 - q5
    
    qi = returns.rolling(period, min_periods=5).apply(lambda x: calc_qi(x), raw=True)
    qi_mean = qi.rolling(zscore_period, min_periods=1).mean()
    qi_std = qi.rolling(zscore_period, min_periods=1).std(ddof=0)
    safe_std = np.where(qi_std != 0, qi_std, 1.0)
    qi_zscore = np.where(qi_std != 0, (qi - qi_mean) / safe_std, 0.0)
    qi_zscore_series = pd.Series(qi_zscore, index=qi.index)
    
    return qi_zscore_series

strategy_msi_bb_qi_param_ranges = {
    'bb_period_range': range(15, 26, 5),
    'bb_lookback_range': range(40, 61, 10),
    'qi_period_range': range(15, 26, 5),
    'qi_zscore_period_range': range(15, 26, 5),
    'qi_exit_threshold_range': range(15, 26, 5),
}

def strategy_msi_bb_qi(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period = params.get('bb_period_range')
    bb_lookback = params.get('bb_lookback_range')
    qi_period = params.get('qi_period_range')
    qi_zscore_period = params.get('qi_zscore_period_range')
    qi_exit_threshold = params.get('qi_exit_threshold_range') / 10.0
    
    df = data.copy()
    
    bb_squeeze = ind_msi_bb_qi_bb_squeeze(df, period=bb_period, lookback=bb_lookback)
    qi_zscore = ind_msi_bb_qi_quantile_interrange(df, period=qi_period, zscore_period=qi_zscore_period)
    
    df['BB_Squeeze'] = bb_squeeze
    df['QI_ZScore'] = qi_zscore
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    entries = df['BB_Squeeze']
    exits = df['QI_ZScore'] >= qi_exit_threshold
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy cop_demarker_supertrend
############################

def ind_cop_demarker_supertrend_demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    
    de_max = np.where(high > high.shift(1), high - high.shift(1), 0)
    de_min = np.where(low < low.shift(1), low.shift(1) - low, 0)
    
    de_max_series = pd.Series(de_max, index=df.index)
    de_min_series = pd.Series(de_min, index=df.index)
    
    dem_max_sum = de_max_series.rolling(period, min_periods=1).sum()
    dem_min_sum = de_min_series.rolling(period, min_periods=1).sum()
    
    denominator = dem_max_sum + dem_min_sum
    safe_denom = np.where(denominator != 0, denominator, 1.0)
    demarker = np.where(denominator != 0, dem_max_sum / safe_denom, 0.5)
    
    return pd.Series(demarker, index=df.index)

def ind_cop_demarker_supertrend_supertrend(df: pd.DataFrame, period: int = 14, multiplier: float = 3.0):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    hl2 = (high + low) / 2.0
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = np.empty(len(tr))
    atr[0] = tr[0]
    for i in range(1, min(period, len(tr))):
        atr[i] = np.mean(tr[:i+1])
    for i in range(period, len(tr)):
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    
    upper_band = np.empty(len(upper_basic))
    lower_band = np.empty(len(lower_basic))
    supertrend = np.empty(len(close), dtype=bool)
    
    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    supertrend[0] = True
    
    for i in range(1, len(close)):
        if close[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_basic[i], upper_band[i-1])
        else:
            upper_band[i] = upper_basic[i]
            
        if close[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_basic[i], lower_band[i-1])
        else:
            lower_band[i] = lower_basic[i]
            
        if close[i] > upper_band[i-1]:
            supertrend[i] = True
        elif close[i] < lower_band[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]
    
    return pd.Series(upper_band, index=df.index), pd.Series(lower_band, index=df.index), pd.Series(supertrend, index=df.index)

strategy_cop_demarker_supertrend_param_ranges = {
    'dem_period_range': range(10, 21, 5),
    'dem_shift1_range': range(3, 8, 2),
    'dem_shift2_range': range(8, 13, 2),
    'st_period_range': range(10, 21, 5),
    'st_multiplier_range': range(20, 41, 10),
}

def strategy_cop_demarker_supertrend(data: pd.DataFrame, params: dict, year: int | None = None):
    dem_period = params.get('dem_period_range')
    dem_shift1 = params.get('dem_shift1_range')
    dem_shift2 = params.get('dem_shift2_range')
    st_period = params.get('st_period_range')
    st_multiplier = params.get('st_multiplier_range') / 10.0
    
    df = data.copy()
    
    demarker = ind_cop_demarker_supertrend_demarker(df, period=dem_period)
    upper_band, lower_band, supertrend = ind_cop_demarker_supertrend_supertrend(df, period=st_period, multiplier=st_multiplier)
    
    df['DeMarker'] = demarker
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    df['SuperTrend'] = supertrend
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    demarker_rising = df['DeMarker'].shift(dem_shift1) > df['DeMarker'].shift(dem_shift2)
    entries = demarker_rising
    
    exits = df['Close'] < df['Lower_Band']
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy are_dem_kc
############################

def ind_are_dem_kc_demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    
    demax = np.where(high > high.shift(1), high - high.shift(1), 0.0)
    demin = np.where(low < low.shift(1), low.shift(1) - low, 0.0)
    
    demax_series = pd.Series(demax, index=df.index)
    demin_series = pd.Series(demin, index=df.index)
    
    dem_max_sum = demax_series.rolling(period, min_periods=1).sum()
    dem_min_sum = demin_series.rolling(period, min_periods=1).sum()
    
    total_sum = dem_max_sum + dem_min_sum
    safe_total = np.where(total_sum != 0, total_sum, 1.0)
    demarker = np.where(total_sum != 0, dem_max_sum / safe_total, 0.5)
    
    return pd.Series(demarker, index=df.index)


def ind_are_dem_kc_keltner(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tp = (high + low + close) / 3.0
    kc_mid = tp.ewm(span=period, adjust=False, min_periods=1).mean()
    
    hl = high - low
    hpc = np.abs(high - close.shift(1))
    lpc = np.abs(low - close.shift(1))
    
    tr = np.maximum(np.maximum(hl.values, hpc.values), lpc.values)
    tr_series = pd.Series(tr, index=df.index)
    atr = tr_series.rolling(period, min_periods=1).mean()
    
    kc_upper = kc_mid + multiplier * atr
    kc_lower = kc_mid - multiplier * atr
    
    return kc_mid, kc_upper, kc_lower


strategy_are_dem_kc_param_ranges = {
    'dem_period_range': range(10, 21, 5),
    'dem_shift1_range': range(3, 8, 2),
    'dem_shift2_range': range(8, 13, 2),
    'kc_period_range': range(15, 26, 5),
    'kc_mult_range': range(15, 26, 5)
}


def strategy_are_dem_kc(data: pd.DataFrame, params: dict, year: int | None = None):
    dem_period = params.get('dem_period_range')
    dem_shift1 = params.get('dem_shift1_range')
    dem_shift2 = params.get('dem_shift2_range')
    kc_period = params.get('kc_period_range')
    kc_mult = params.get('kc_mult_range') / 10.0
    
    df = data.copy()
    
    demarker = ind_are_dem_kc_demarker(df, period=dem_period)
    kc_mid, kc_upper, kc_lower = ind_are_dem_kc_keltner(df, period=kc_period, multiplier=kc_mult)
    
    df['DeMarker'] = demarker
    df['KC_Mid'] = kc_mid
    df['KC_Upper'] = kc_upper
    df['KC_Lower'] = kc_lower
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    demarker_rising = df['DeMarker'].shift(dem_shift1) > df['DeMarker'].shift(dem_shift2)
    close_below_kc_lower = df['Close'] < df['KC_Lower']
    
    entries = demarker_rising
    exits = close_below_kc_lower
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy cag_hma_upturn_bollinger
############################

def ind_cag_hma_upturn_bollinger_hma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    close = df['Close']
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(np.sqrt(length)))
    
    wma_half = close.rolling(window=half_length, min_periods=1).mean()
    wma_full = close.rolling(window=length, min_periods=1).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_length, min_periods=1).mean()
    return hma

def ind_cag_hma_upturn_bollinger_bb(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0):
    close = df['Close']
    bb_ma = close.rolling(window=period, min_periods=1).mean()
    bb_std = close.rolling(window=period, min_periods=1).std(ddof=0)
    bb_upper = bb_ma + std_mult * bb_std
    bb_lower = bb_ma - std_mult * bb_std
    return bb_ma, bb_upper, bb_lower

strategy_cag_hma_upturn_bollinger_param_ranges = {
    'hma_fast_range': range(15, 26, 5),
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
    'bb_shift_range': range(3, 8, 2)
}

def strategy_cag_hma_upturn_bollinger(data: pd.DataFrame, params: dict, year: int | None = None):
    hma_fast_len = params.get('hma_fast_range')
    bb_period = params.get('bb_period_range')
    bb_std_mult = params.get('bb_std_range') / 10.0
    bb_shift = params.get('bb_shift_range')
    
    df = data.copy()
    
    hma_fast = ind_cag_hma_upturn_bollinger_hma(df, length=hma_fast_len)
    bb_ma, bb_upper, bb_lower = ind_cag_hma_upturn_bollinger_bb(df, period=bb_period, std_mult=bb_std_mult)
    
    df['HMA_Fast'] = hma_fast
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    hma_upturn = df['HMA_Fast'] > df['HMA_Fast'].shift(1)
    
    open_below_lower_shifted = df['Open'].shift(bb_shift) < df['BB_Lower'].shift(bb_shift)
    open_above_lower_now = df['Open'] > df['BB_Lower']
    bb_reclaim = open_above_lower_now & open_below_lower_shifted
    
    entries = hma_upturn
    exits = bb_reclaim
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy mpwr_bb_std
############################

def ind_mpwr_bb_std_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
    close = df['Close']
    sma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return sma, upper_band, lower_band

def ind_mpwr_bb_std_rolling_std(df: pd.DataFrame, period: int = 20):
    close = df['Close']
    rolling_std = close.rolling(period, min_periods=1).std(ddof=0)
    return rolling_std

strategy_mpwr_bb_std_param_ranges = {
    'bb_period_range': range(15, 26, 5),
    'bb_std_dev_range': range(15, 26, 5),
    'vol_period_range': range(10, 21, 5),
}

def strategy_mpwr_bb_std(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period = params.get('bb_period_range')
    bb_std_dev = params.get('bb_std_dev_range') / 10.0
    vol_period = params.get('vol_period_range')
    
    df = data.copy()
    
    sma, upper_band, lower_band = ind_mpwr_bb_std_bollinger(df, bb_period, bb_std_dev)
    rolling_std = ind_mpwr_bb_std_rolling_std(df, vol_period)
    
    df['SMA'] = sma
    df['BB_Upper'] = upper_band
    df['BB_Lower'] = lower_band
    df['Rolling_STD'] = rolling_std
    df['STD_Change'] = df['Rolling_STD'].diff()
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    entries = df['Open'] < df['BB_Upper']
    exits = df['STD_Change'] > 0
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


############################
# Strategy bb_expansion
############################

def ind_bb_expansion_bollinger(df: pd.DataFrame, period: int = 20, std_multiplier: float = 2.0):
    close = df['Close']
    bb_ma = close.rolling(period, min_periods=1).mean()
    bb_std = close.rolling(period, min_periods=1).std(ddof=0)
    bb_upper = bb_ma + std_multiplier * bb_std
    bb_lower = bb_ma - std_multiplier * bb_std
    return bb_ma, bb_upper, bb_lower

def ind_bb_expansion_bandwidth(df: pd.DataFrame, bb_upper: pd.Series, bb_lower: pd.Series, shift: int = 5):
    bandwidth = bb_upper - bb_lower
    expansion = bandwidth > bandwidth.shift(shift)
    return bandwidth, expansion

strategy_bb_expansion_param_ranges = {
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
    'shift_range': range(3, 8, 2),
}

def strategy_bb_expansion(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period = params.get('bb_period_range')
    bb_std = params.get('bb_std_range') / 10.0
    shift_days = params.get('shift_range')
    
    df = data.copy()
    
    bb_ma, bb_upper, bb_lower = ind_bb_expansion_bollinger(df, period=bb_period, std_multiplier=bb_std)
    bandwidth, expansion = ind_bb_expansion_bandwidth(df, bb_upper, bb_lower, shift=shift_days)
    
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['Bandwidth'] = bandwidth
    df['Expansion'] = expansion
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    open_below_upper = df['Open'] < df['BB_Upper']
    expansion_signal = df['Expansion']
    
    entries = open_below_upper
    exits = expansion_signal
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits
