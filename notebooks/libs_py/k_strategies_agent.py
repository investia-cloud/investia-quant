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


# ─────────────────────────────────────
# Fonte: I Let a Strategy Generator Build My AMD Trade — Here’s the Truth It Revealed
# URL:   https://medium.com/@Kryptera/i-let-a-strategy-generator-build-my-amd-trade-heres-the-truth-it-revealed-180aaf479441
# Data:  2026-06-12 23:00
# ─────────────────────────────────────

############################
# Strategy amd_momentum_rsi
############################

def ind_amd_momentum_rsi_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    close = df['Close']
    momentum = (close / close.shift(period)) * 100
    return momentum

def ind_amd_momentum_rsi_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(span=period, adjust=False, min_periods=1).mean()
    safe_roll_down = np.where(roll_down != 0, roll_down, 1.0)
    rs = np.where(roll_down != 0, roll_up / safe_roll_down, 0.0)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)

strategy_amd_momentum_rsi_param_ranges = {
    'fast_period_range': range(3, 8, 2),
    'slow_period_range': range(8, 16, 3),
    'rsi_period_range': range(10, 21, 5),
    'rsi_level_range': range(25, 36, 5),
}

def strategy_amd_momentum_rsi(data: pd.DataFrame, params: dict, year: int | None = None):
    fast_p = params.get('fast_period_range')
    slow_p = params.get('slow_period_range')
    rsi_p = params.get('rsi_period_range')
    rsi_level = params.get('rsi_level_range')
    
    df = data.copy()
    
    fast_momentum = ind_amd_momentum_rsi_momentum(df, period=fast_p)
    slow_momentum = ind_amd_momentum_rsi_momentum(df, period=slow_p)
    rsi = ind_amd_momentum_rsi_rsi(df, period=rsi_p)
    
    df['Fast_Momentum'] = fast_momentum
    df['Slow_Momentum'] = slow_momentum
    df['RSI'] = rsi
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    momentum_cross_down = (df['Fast_Momentum'] < df['Slow_Momentum']) & (df['Fast_Momentum'].shift(1) >= df['Slow_Momentum'].shift(1))
    rsi_exit_oversold = (df['RSI'] > rsi_level) & (df['RSI'].shift(1) <= rsi_level)
    
    entries = momentum_cross_down
    exits = rsi_exit_oversold
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: Multiple Indicator Trading Strategy in Python — A Full Guide. _ by Sofien Kaabar, CFA _ Investor’s Handbook _ Medium
# URL:   file:///home/luca/Downloads/Multiple Indicator Trading Strategy in Python — A Full Guide. _ by Sofien Kaabar, CFA _ Investor’s Handbook _ Medium.pdf
# Data:  2026-06-15 11:55
# ─────────────────────────────────────

############################
# Strategy multiple_indicator
############################

def ind_multiple_indicator_stochastic_smoothing(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    ema_high = np.empty(len(df))
    ema_low = np.empty(len(df))
    ema_close = np.empty(len(df))
    
    alpha = 2.0 / (2 + 1.0)
    
    ema_high[0] = high[0]
    ema_low[0] = low[0]
    ema_close[0] = close[0]
    
    for i in range(1, len(df)):
        ema_high[i] = alpha * high[i] + (1 - alpha) * ema_high[i-1]
        ema_low[i] = alpha * low[i] + (1 - alpha) * ema_low[i-1]
        ema_close[i] = alpha * close[i] + (1 - alpha) * ema_close[i-1]
    
    sso = np.empty(len(df))
    sso[:] = np.nan
    
    for i in range(lookback-1, len(df)):
        min_low = np.min(ema_low[max(0, i-lookback+1):i+1])
        max_high = np.max(ema_high[max(0, i-lookback+1):i+1])
        
        if max_high != min_low:
            sso[i] = 100 * (ema_close[i] - min_low) / (max_high - min_low)
        else:
            sso[i] = 50
    
    return pd.Series(sso, index=df.index)

def ind_multiple_indicator_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0):
    close = df['Close']
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    
    return ma, upper, lower

def ind_multiple_indicator_fib_timing(df: pd.DataFrame, fib_period: int = 21) -> pd.Series:
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    fib_signal = pd.Series(0, index=df.index)
    
    for i in range(fib_period, len(df)):
        period_high = high.iloc[i-fib_period:i].max()
        period_low = low.iloc[i-fib_period:i].min()
        
        fib_618 = period_low + 0.618 * (period_high - period_low)
        fib_382 = period_low + 0.382 * (period_high - period_low)
        
        current_price = close.iloc[i]
        
        if current_price >= fib_618:
            fib_signal.iloc[i] = 1
        elif current_price <= fib_382:
            fib_signal.iloc[i] = -1
    
    return fib_signal

strategy_multiple_indicator_param_ranges = {
    'sso_lookback_range': range(10, 21, 5),
    'bb_period_range': range(15, 26, 5),
    'bb_std_range': range(15, 26, 5),
    'fib_period_range': range(15, 26, 5)
}

def strategy_multiple_indicator(data: pd.DataFrame, params: dict, year: int | None = None):
    sso_lookback = params.get('sso_lookback_range')
    bb_period = params.get('bb_period_range')
    bb_std = params.get('bb_std_range') / 10.0
    fib_period = params.get('fib_period_range')
    
    df = data.copy()
    
    sso = ind_multiple_indicator_stochastic_smoothing(df, lookback=sso_lookback)
    bb_ma, bb_upper, bb_lower = ind_multiple_indicator_bollinger(df, period=bb_period, std_mult=bb_std)
    fib_signal = ind_multiple_indicator_fib_timing(df, fib_period=fib_period)
    
    df['SSO'] = sso
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['FIB_Signal'] = fib_signal
    
    if year is not None:
        df = df[df.index.year == int(year)]
    
    sso_oversold = df['SSO'] < 20
    sso_overbought = df['SSO'] > 80
    
    price_near_lower = df['Close'] <= df['BB_Lower'] * 1.02
    price_near_upper = df['Close'] >= df['BB_Upper'] * 0.98
    
    fib_bullish = df['FIB_Signal'] == 1
    fib_bearish = df['FIB_Signal'] == -1
    
    entries_long = sso_oversold & price_near_lower & fib_bullish
    entries_short = sso_overbought & price_near_upper & fib_bearish
    entries = entries_long | entries_short
    
    exits_long = (df['SSO'] > 70) | (df['Close'] >= df['BB_Upper'])
    exits_short = (df['SSO'] < 30) | (df['Close'] <= df['BB_Lower'])
    exits = exits_long | exits_short
    
    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Let an Algorithm Backtest Energy Transfer for 16 Years. Here’s What Happened.
# URL:   https://medium.com/@Kryptera/i-let-an-algorithm-backtest-energy-transfer-for-16-years-heres-what-happened-e0320effaedd
# Data:  2026-06-16 10:30
# ─────────────────────────────────────

############################
# Strategy tema_tsi
############################

import pandas as pd
import numpy as np


def ind_tema_tsi_tema(df: pd.DataFrame, tema_period: int = 20) -> pd.Series:
    close = df['Close']
    ema1 = close.ewm(span=tema_period, adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=tema_period, adjust=False, min_periods=1).mean()
    ema3 = ema2.ewm(span=tema_period, adjust=False, min_periods=1).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def ind_tema_tsi_tsi(df: pd.DataFrame,
                     tsi_long: int = 25,
                     tsi_short: int = 13,
                     tsi_signal: int = 7) -> tuple:
    close = df['Close']
    pc = close.diff(1)

    # Double smoothed price change
    ema1_pc = pc.ewm(span=tsi_long, adjust=False, min_periods=1).mean()
    ema2_pc = ema1_pc.ewm(span=tsi_short, adjust=False, min_periods=1).mean()

    # Double smoothed absolute price change
    apc = pc.abs()
    ema1_apc = apc.ewm(span=tsi_long, adjust=False, min_periods=1).mean()
    ema2_apc = ema1_apc.ewm(span=tsi_short, adjust=False, min_periods=1).mean()

    den_arr = ema2_apc.values
    num_arr = ema2_pc.values
    safe_den = np.where(den_arr != 0, den_arr, 1.0)
    tsi_arr = np.where(den_arr != 0, 100.0 * num_arr / safe_den, 0.0)
    tsi = pd.Series(tsi_arr, index=close.index)

    signal = tsi.ewm(span=tsi_signal, adjust=False, min_periods=1).mean()
    return tsi, signal


strategy_tema_tsi_param_ranges = {
    'tema_period_range': range(16, 25, 4),   # 16, 20, 24
    'tsi_long_range'   : range(20, 31, 5),   # 20, 25, 30
    'tsi_short_range'  : range(11, 16, 2),   # 11, 13, 15
    'tsi_signal_range' : range(5, 10, 2),    # 5, 7, 9
}


def strategy_tema_tsi(data: pd.DataFrame, params: dict, year: int | None = None):
    tema_period = params.get('tema_period_range')
    tsi_long    = params.get('tsi_long_range')
    tsi_short   = params.get('tsi_short_range')
    tsi_signal  = params.get('tsi_signal_range')

    df = data.copy()

    tema = ind_tema_tsi_tema(df, tema_period=tema_period)
    tsi, signal = ind_tema_tsi_tsi(df, tsi_long=tsi_long, tsi_short=tsi_short, tsi_signal=tsi_signal)

    df['TEMA']       = tema
    df['TSI']        = tsi
    df['TSI_Signal'] = signal

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: price closes above TEMA
    entries = df['Close'] > df['TEMA']

    # Exit: TSI crosses below signal AND TSI < 0
    tsi_cross_below = (df['TSI'] < df['TSI_Signal']) & (df['TSI'].shift(1) >= df['TSI_Signal'].shift(1))
    tsi_below_zero  = df['TSI'] < 0
    exits = tsi_cross_below & tsi_below_zero

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Stress-Tested Those 2 META Trading Strategies. Here’s the Honest Truth.
# URL:   https://medium.com/@Kryptera/i-stress-tested-those-2-meta-trading-strategies-heres-the-honest-truth-bf61a26cfe28
# Data:  2026-06-16 10:31
# ─────────────────────────────────────

############################
# Strategy lrsi_tsi
############################

import pandas as pd
import numpy as np


def ind_lrsi_tsi_laguerre_rsi(df: pd.DataFrame, gamma: float = 0.5) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    lrsi = np.zeros(n)

    for i in range(1, n):
        L0[i] = (1 - gamma) * close[i] + gamma * L0[i - 1]
        L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
        L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
        L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]

        cu = 0.0
        cd = 0.0
        if L0[i] >= L1[i]:
            cu += L0[i] - L1[i]
        else:
            cd += L1[i] - L0[i]
        if L1[i] >= L2[i]:
            cu += L1[i] - L2[i]
        else:
            cd += L2[i] - L1[i]
        if L2[i] >= L3[i]:
            cu += L2[i] - L3[i]
        else:
            cd += L3[i] - L2[i]

        denom = cu + cd
        if denom != 0:
            lrsi[i] = cu / denom
        else:
            lrsi[i] = 0.5

    return pd.Series(lrsi, index=df.index)


def ind_lrsi_tsi_tsi(df: pd.DataFrame, r: int = 25, s: int = 13) -> pd.Series:
    close = df['Close']
    momentum = close.diff(1)
    abs_momentum = momentum.abs()

    ema1_m = momentum.ewm(span=r, adjust=False, min_periods=1).mean()
    ema2_m = ema1_m.ewm(span=s, adjust=False, min_periods=1).mean()

    ema1_a = abs_momentum.ewm(span=r, adjust=False, min_periods=1).mean()
    ema2_a = ema1_a.ewm(span=s, adjust=False, min_periods=1).mean()

    num = 100.0 * ema2_m
    den = ema2_a
    den_arr = den.values
    num_arr = num.values
    safe_den = np.where(den_arr != 0, den_arr, 1.0)
    result = np.where(den_arr != 0, num_arr / safe_den, 0.0)

    return pd.Series(result, index=df.index)


strategy_lrsi_tsi_param_ranges = {
    'gamma_range': range(3, 8, 2),
    'tsi_r_range': range(15, 36, 10),
    'tsi_s_range': range(8, 19, 5),
}


def strategy_lrsi_tsi(data: pd.DataFrame, params: dict, year: int | None = None):
    gamma = params.get('gamma_range') / 10.0
    tsi_r = params.get('tsi_r_range')
    tsi_s = params.get('tsi_s_range')

    df = data.copy()

    lrsi = ind_lrsi_tsi_laguerre_rsi(df, gamma=gamma)
    tsi = ind_lrsi_tsi_tsi(df, r=tsi_r, s=tsi_s)

    df['LRSI'] = lrsi
    df['TSI'] = tsi

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['LRSI'] > df['LRSI'].shift(1)
    exits = (df['TSI'] > 0) & (df['TSI'].shift(1) <= 0)

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Found a Tension Flow Trend TradingView Indicator — So I Backtested It on 29 Years of Data
# URL:   https://medium.com/@Kryptera/i-found-a-tension-flow-trend-tradingview-indicator-so-i-backtested-it-on-29-years-of-data-a73874952129
# Data:  2026-06-16 10:31
# ─────────────────────────────────────

############################
# Strategy tension_flow_trend
############################

import pandas as pd
import numpy as np


def ind_tension_flow_trend_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    arr = series.values
    n = len(arr)
    out = np.empty(n)
    out[:] = np.nan
    for i in range(period - 1, n):
        window = arr[i - period + 1: i + 1]
        out[i] = np.dot(window, weights) / weights.sum()
    return pd.Series(out, index=series.index)


def ind_tension_flow_trend_hma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    close = df['Close']
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(round(period ** 0.5)))
    wma_half = ind_tension_flow_trend_wma(close, half_period)
    wma_full = ind_tension_flow_trend_wma(close, period)
    diff = 2.0 * wma_half - wma_full
    hma = ind_tension_flow_trend_wma(diff, sqrt_period)
    return hma


strategy_tension_flow_trend_param_ranges = {
    'hma_period_range': range(30, 71, 20),
    'signal_gap_range': range(15, 46, 15),
}


def strategy_tension_flow_trend(data: pd.DataFrame, params: dict, year: int | None = None):
    hma_period = params.get('hma_period_range')
    signal_gap = params.get('signal_gap_range')

    df = data.copy()

    hma = ind_tension_flow_trend_hma(df, period=hma_period)
    df['HMA'] = hma

    slope_up = hma > hma.shift(1)
    slope_down = hma < hma.shift(1)

    close = df['Close']
    cross_above = (close > hma) & (close.shift(1) <= hma.shift(1))
    cross_below = (close < hma) & (close.shift(1) >= hma.shift(1))

    raw_long_signal = cross_above & slope_up
    raw_short_signal = cross_below & slope_down

    raw_long_arr = raw_long_signal.values
    raw_short_arr = raw_short_signal.values
    n = len(raw_long_arr)

    entry_arr = np.zeros(n, dtype=bool)
    exit_arr = np.zeros(n, dtype=bool)

    last_signal_bar = -signal_gap - 1
    in_trade = False
    trade_direction = 0  # 1 long, -1 short

    for i in range(n):
        if not in_trade:
            if raw_long_arr[i] and (i - last_signal_bar) >= signal_gap:
                entry_arr[i] = True
                last_signal_bar = i
                in_trade = True
                trade_direction = 1
            elif raw_short_arr[i] and (i - last_signal_bar) >= signal_gap:
                entry_arr[i] = True
                last_signal_bar = i
                in_trade = True
                trade_direction = -1
        else:
            if trade_direction == 1:
                if raw_short_arr[i] and (i - last_signal_bar) >= signal_gap:
                    exit_arr[i] = True
                    entry_arr[i] = True
                    last_signal_bar = i
                    trade_direction = -1
                elif slope_down.values[i]:
                    exit_arr[i] = True
                    in_trade = False
                    trade_direction = 0
            elif trade_direction == -1:
                if raw_long_arr[i] and (i - last_signal_bar) >= signal_gap:
                    exit_arr[i] = True
                    entry_arr[i] = True
                    last_signal_bar = i
                    trade_direction = 1
                elif slope_up.values[i]:
                    exit_arr[i] = True
                    in_trade = False
                    trade_direction = 0

    entries = pd.Series(entry_arr, index=df.index)
    exits = pd.Series(exit_arr, index=df.index)

    if year is not None:
        mask = df.index.year == int(year)
        entries = entries[mask]
        exits = exits[mask]

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: Liquidity Sweeps as a Signal — The Hidden Edge in AMPE Indicator
# URL:   https://medium.com/@Kryptera/liquidity-sweeps-as-a-signal-the-hidden-edge-in-ampe-indicator-a0ccb0326a53
# Data:  2026-06-16 10:33
# ─────────────────────────────────────

############################
# Strategy ampe_liquidity
############################

import pandas as pd
import numpy as np


def ind_ampe_liquidity_sweeps(df: pd.DataFrame, sweep_len: int = 20):
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    prior_low  = low.shift(1).rolling(sweep_len, min_periods=1).min()
    prior_high = high.shift(1).rolling(sweep_len, min_periods=1).max()

    bull_sweep = (low < prior_low) & (close > prior_low)
    bear_sweep = (high > prior_high) & (close < prior_high)

    return bull_sweep, bear_sweep


def ind_ampe_liquidity_pressure(bull_sweep: pd.Series,
                                bear_sweep: pd.Series,
                                sweep_len: int = 20) -> pd.Series:
    sweep_pressure_len = max(2, sweep_len // 2)
    bull_liq = bull_sweep.rolling(sweep_pressure_len, min_periods=1).sum()
    bear_liq = bear_sweep.rolling(sweep_pressure_len, min_periods=1).sum()
    net = bull_liq - bear_liq
    liquidity_pressure = ((net - (-5)) / (5 - (-5))).clip(0, 1)
    return liquidity_pressure


def ind_ampe_liquidity_trend_pressure(df: pd.DataFrame,
                                      ema_fast_p: int = 12,
                                      ema_slow_p: int = 26,
                                      atr_p: int = 14,
                                      regime_lookback: int = 50):
    close = df['Close']
    high  = df['High']
    low   = df['Low']

    ema_fast = close.ewm(span=ema_fast_p, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=ema_slow_p, adjust=False, min_periods=1).mean()

    prev_close = close.shift(1)
    hl   = high.values - low.values
    hpc  = np.abs(high.values - prev_close.values)
    lpc  = np.abs(low.values  - prev_close.values)
    tr_vals = np.maximum(np.maximum(hl, hpc), lpc)
    tr = pd.Series(tr_vals, index=close.index)

    tr_arr  = tr.values
    atr_arr = np.empty(len(tr_arr))
    atr_arr[0] = tr_arr[0]
    alpha = 1.0 / atr_p
    for i in range(1, len(tr_arr)):
        atr_arr[i] = atr_arr[i-1] * (1 - alpha) + tr_arr[i] * alpha
    atr_val = pd.Series(atr_arr, index=close.index)

    safe_atr = np.where(atr_val.values != 0, atr_val.values, 1.0)
    trend_strength_vals = np.abs(ema_fast.values - ema_slow.values) / safe_atr
    trend_strength = pd.Series(trend_strength_vals, index=close.index)

    ts_median = trend_strength.rolling(regime_lookback, min_periods=1).median()
    trending_regime = trend_strength > ts_median

    trend_direction = (ema_fast - ema_slow) / pd.Series(safe_atr, index=close.index)
    td_std = trend_direction.rolling(regime_lookback, min_periods=1).std(ddof=0).replace(0, np.nan)
    trend_pressure = (trend_direction / td_std).clip(-3, 3)
    trend_pressure_norm = ((trend_pressure - (-3)) / (3 - (-3))).clip(0, 1)

    return trend_pressure_norm, trending_regime, ema_fast, ema_slow


def ind_ampe_liquidity_momentum_pressure(df: pd.DataFrame, mom_p: int = 14, smooth: int = 5) -> pd.Series:
    close = df['Close']
    roc = close.pct_change(mom_p)
    roc_std = roc.rolling(mom_p, min_periods=1).std(ddof=0).replace(0, np.nan)
    z = roc / roc_std
    z_smooth = z.ewm(span=smooth, adjust=False, min_periods=1).mean()
    mom_norm = ((z_smooth - (-3)) / (3 - (-3))).clip(0, 1)
    return mom_norm


def ind_ampe_liquidity_composite(trend_pressure: pd.Series,
                                 mom_pressure: pd.Series,
                                 liq_pressure: pd.Series,
                                 trending_regime: pd.Series) -> pd.Series:
    t_w = np.where(trending_regime.values, 0.45, 0.20)
    m_w = np.where(trending_regime.values, 0.30, 0.20)
    l_w = np.where(trending_regime.values, 0.15, 0.35)
    c_w = np.where(trending_regime.values, 0.10, 0.25)

    vol_comp = 1.0 - liq_pressure.rolling(10, min_periods=1).std(ddof=0).clip(0, 0.5) / 0.5

    ampe = (t_w * trend_pressure.values +
            m_w * mom_pressure.values +
            l_w * liq_pressure.values +
            c_w * vol_comp.values)
    return pd.Series(ampe, index=trend_pressure.index)


strategy_ampe_liquidity_param_ranges = {
    'sweep_len_range'   : range(10, 31, 10),
    'ema_fast_range'    : range(8, 17, 4),
    'ema_slow_range'    : range(20, 41, 10),
    'mom_p_range'       : range(10, 21, 5),
    'regime_lb_range'   : range(30, 61, 15),
    'threshold_range'   : range(6, 9, 1),
}


def strategy_ampe_liquidity(data: pd.DataFrame, params: dict, year: int | None = None):
    sweep_len   = params.get('sweep_len_range')
    ema_fast_p  = params.get('ema_fast_range')
    ema_slow_p  = params.get('ema_slow_range')
    mom_p       = params.get('mom_p_range')
    regime_lb   = params.get('regime_lb_range')
    threshold   = params.get('threshold_range') / 10.0

    df = data.copy()

    bull_sweep, bear_sweep = ind_ampe_liquidity_sweeps(df, sweep_len=sweep_len)
    liq_pressure = ind_ampe_liquidity_pressure(bull_sweep, bear_sweep, sweep_len=sweep_len)
    trend_pressure, trending_regime, ema_fast, ema_slow = ind_ampe_liquidity_trend_pressure(
        df,
        ema_fast_p=ema_fast_p,
        ema_slow_p=ema_slow_p,
        atr_p=14,
        regime_lookback=regime_lb
    )
    mom_pressure = ind_ampe_liquidity_momentum_pressure(df, mom_p=mom_p, smooth=5)
    ampe = ind_ampe_liquidity_composite(trend_pressure, mom_pressure, liq_pressure, trending_regime)

    df['bull_sweep']      = bull_sweep
    df['bear_sweep']      = bear_sweep
    df['liq_pressure']    = liq_pressure
    df['trend_pressure']  = trend_pressure
    df['mom_pressure']    = mom_pressure
    df['trending_regime'] = trending_regime
    df['ema_fast']        = ema_fast
    df['ema_slow']        = ema_slow
    df['ampe']            = ampe

    if year is not None:
        df = df[df.index.year == int(year)]

    ampe_s     = df['ampe']
    liq_s      = df['liq_pressure']
    bull_sw    = df['bull_sweep']
    bear_sw    = df['bear_sweep']
    ema_f      = df['ema_fast']
    ema_sl     = df['ema_slow']
    prev_ampe  = ampe_s.shift(1)

    entries_long  = (ampe_s >= threshold) & (prev_ampe < threshold) & bull_sw & (ema_f > ema_sl)
    entries_short = (ampe_s <= (1.0 - threshold)) & (prev_ampe > (1.0 - threshold)) & bear_sw & (ema_f < ema_sl)
    entries = entries_long | entries_short

    exits_long  = (ampe_s < 0.5) & (prev_ampe >= 0.5)
    exits_long  = exits_long | (bear_sw & (liq_s < 0.4) & (ema_f > ema_sl))

    exits_short = (ampe_s > 0.5) & (prev_ampe <= 0.5)
    exits_short = exits_short | (bull_sw & (liq_s > 0.6) & (ema_f < ema_sl))

    exits = exits_long | exits_short

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Let a Strategy Generator Scan 42+ Years of AAPL Data. Here’s What Survived.
# URL:   https://medium.com/@Kryptera/i-let-a-strategy-generator-scan-60-years-of-aapl-data-heres-what-survived-62dc8a2c903e
# Data:  2026-06-19 02:00
# ─────────────────────────────────────

############################
# Strategy laguerre_cts
############################

import pandas as pd
import numpy as np


def ind_laguerre_cts_lrsi(df: pd.DataFrame, gamma: float = 0.5) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    lrsi = np.zeros(n)

    for i in range(1, n):
        L0[i] = (1 - gamma) * close[i] + gamma * L0[i - 1]
        L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
        L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
        L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]

        cu = 0.0
        cd = 0.0
        if L0[i] >= L1[i]:
            cu += L0[i] - L1[i]
        else:
            cd += L1[i] - L0[i]
        if L1[i] >= L2[i]:
            cu += L1[i] - L2[i]
        else:
            cd += L2[i] - L1[i]
        if L2[i] >= L3[i]:
            cu += L2[i] - L3[i]
        else:
            cd += L3[i] - L2[i]

        denom = cu + cd
        if denom != 0.0:
            lrsi[i] = cu / denom
        else:
            lrsi[i] = 0.0

    return pd.Series(lrsi, index=df.index)


def ind_laguerre_cts_cts(df: pd.DataFrame, period: int = 20) -> pd.Series:
    close = df['Close']
    n = len(close)
    cts_arr = np.zeros(n)
    close_arr = close.values

    for i in range(period - 1, n):
        window = close_arr[i - period + 1: i + 1]
        ref = window[-1]
        up = np.sum(window[1:] > window[:-1])
        dn = np.sum(window[1:] < window[:-1])
        total = up + dn
        if total != 0:
            cts_arr[i] = (up - dn) / total
        else:
            cts_arr[i] = 0.0

    return pd.Series(cts_arr, index=df.index)


strategy_laguerre_cts_param_ranges = {
    'gamma_range'     : range(3, 8, 2),   # 3 values: [3,5,7] -> divide by 10
    'lrsi_ob_range'   : range(15, 30, 7), # 3 values: [15,22,29] -> divide by 100
    'cts_period_range': range(10, 31, 10),# 3 values: [10,20,30]
    'cts_low_range'   : range(-5, 1, 2),  # 3 values: [-5,-3,-1] -> divide by 10
}


def strategy_laguerre_cts(data: pd.DataFrame, params: dict, year: int | None = None):
    gamma       = params.get('gamma_range') / 10.0
    lrsi_ob     = params.get('lrsi_ob_range') / 100.0
    cts_period  = params.get('cts_period_range')
    cts_low     = params.get('cts_low_range') / 10.0

    df = data.copy()

    lrsi = ind_laguerre_cts_lrsi(df, gamma=gamma)
    cts  = ind_laguerre_cts_cts(df, period=cts_period)

    df['LRSI'] = lrsi
    df['CTS']  = cts

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: Laguerre RSI oversold (below threshold)
    lrsi_oversold = df['LRSI'] < lrsi_ob

    # Exit: CTS crosses above lower band
    cts_prev = df['CTS'].shift(1)
    cts_cross_above_lower = (df['CTS'] > cts_low) & (cts_prev <= cts_low)

    entries = lrsi_oversold
    exits   = cts_cross_above_lower

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: AMD-Laguerre Keltner Breakdown Strategy
# URL:   https://medium.com/@Kryptera/amd-laguerre-keltner-breakdown-strategy-407f9d6b6730
# Data:  2026-06-19 02:01
# ─────────────────────────────────────

############################
# Strategy leks
############################

import pandas as pd
import numpy as np


def ind_leks_laguerre_rsi(df: pd.DataFrame, gamma: float = 0.5) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    lrsi = np.empty(n)
    L0 = L1 = L2 = L3 = 0.0
    for i in range(n):
        price = close[i]
        L0_new = (1.0 - gamma) * price + gamma * L0
        L1_new = -gamma * L0_new + L0_new + gamma * L1
        L2_new = -gamma * L1_new + L1_new + gamma * L2
        L3_new = -gamma * L2_new + L2_new + gamma * L3
        L0, L1, L2, L3 = L0_new, L1_new, L2_new, L3_new
        CU = max(L0 - L1, 0.0) + max(L1 - L2, 0.0) + max(L2 - L3, 0.0)
        CD = max(L1 - L0, 0.0) + max(L2 - L1, 0.0) + max(L3 - L2, 0.0)
        denom = CU + CD
        lrsi[i] = CU / denom if denom != 0.0 else 0.0
    return pd.Series(lrsi, index=df.index)


def ind_leks_keltner(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_ = df['Open'].values

    tp = (high + low + close) / 3.0
    tp_s = pd.Series(tp, index=df.index)
    kc_mid = tp_s.ewm(span=period, adjust=False, min_periods=1).mean()

    prev_close = np.empty(len(close))
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    hl = high - low
    hpc = np.abs(high - prev_close)
    lpc = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(hl, hpc), lpc)
    atr = pd.Series(tr, index=df.index).rolling(period, min_periods=1).mean()

    kc_lower = kc_mid - multiplier * atr

    open_s = pd.Series(open_, index=df.index)
    return kc_mid, kc_lower, open_s


strategy_leks_param_ranges = {
    'gamma_range'      : range(3, 8, 2),   # 3 values: [3,5,7] / 10 => [0.3,0.5,0.7]
    'lrsi_level_range' : range(4, 7, 1),   # 3 values: [4,5,6] / 10 => [0.4,0.5,0.6]
    'kc_period_range'  : range(10, 31, 10),# 3 values: [10,20,30]
    'kc_mult_range'    : range(15, 31, 5), # 4 values: [15,20,25,30] / 10
}


def strategy_leks(data: pd.DataFrame, params: dict, year: int | None = None):
    gamma       = params.get('gamma_range') / 10.0
    lrsi_level  = params.get('lrsi_level_range') / 10.0
    kc_period   = params.get('kc_period_range')
    kc_mult     = params.get('kc_mult_range') / 10.0

    df = data.copy()

    lrsi = ind_leks_laguerre_rsi(df, gamma=gamma)
    kc_mid, kc_lower, open_s = ind_leks_keltner(df, period=kc_period, multiplier=kc_mult)

    df['LRsi']     = lrsi
    df['KC_Lower'] = kc_lower
    df['Open_s']   = open_s

    if year is not None:
        df = df[df.index.year == int(year)]

    lrsi_cross_below = (df['LRsi'] < lrsi_level) & (df['LRsi'].shift(1) >= lrsi_level)
    open_below_lower = df['Open_s'] < df['KC_Lower']

    entries = lrsi_cross_below
    exits   = open_below_lower

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Generated 2 Trading Strategies on the Same Stock. Both Beat the Market On a Simple Backtest.
# URL:   https://medium.com/@Kryptera/i-generated-2-trading-strategies-on-the-same-stock-both-beat-the-market-on-a-simple-backtest-13361472f4a9
# Data:  2026-06-19 02:02
# ─────────────────────────────────────

############################
# Strategy lrsi_tsi_qqe_stc
############################

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Strategy 1 helpers: Laguerre RSI + TSI
# ─────────────────────────────────────────────

def ind_lrsi_tsi_qqe_stc_laguerre_rsi(close: pd.Series, gamma: float = 0.5) -> pd.Series:
    arr = close.values.astype(float)
    n = len(arr)
    lrsi = np.empty(n)
    L0 = L1 = L2 = L3 = 0.0
    for i in range(n):
        p = arr[i]
        L0_new = (1.0 - gamma) * p + gamma * L0
        L1_new = -gamma * L0_new + L0_new + gamma * L1
        L2_new = -gamma * L1_new + L1_new + gamma * L2
        L3_new = -gamma * L2_new + L2_new + gamma * L3
        L0, L1, L2, L3 = L0_new, L1_new, L2_new, L3_new
        CU = max(L0 - L1, 0.0) + max(L1 - L2, 0.0) + max(L2 - L3, 0.0)
        CD = max(L1 - L0, 0.0) + max(L2 - L1, 0.0) + max(L3 - L2, 0.0)
        denom = CU + CD
        lrsi[i] = CU / denom if denom != 0.0 else 0.0
    return pd.Series(lrsi, index=close.index)


def ind_lrsi_tsi_qqe_stc_tsi(close: pd.Series,
                              long: int = 25,
                              short: int = 13,
                              signal: int = 7) -> pd.Series:
    momentum = close.diff()
    ema1 = momentum.ewm(span=long, adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=short, adjust=False, min_periods=1).mean()
    abs_ema1 = momentum.abs().ewm(span=long, adjust=False, min_periods=1).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False, min_periods=1).mean()
    num = ema2.values
    den = abs_ema2.values
    safe_den = np.where(den != 0.0, den, 1.0)
    tsi_arr = np.where(den != 0.0, 100.0 * num / safe_den, 0.0)
    return pd.Series(tsi_arr, index=close.index)


# ─────────────────────────────────────────────
# Strategy 2 helpers: QQE + STC
# ─────────────────────────────────────────────

def ind_lrsi_tsi_qqe_stc_qqe(close: pd.Series,
                              rsi_period: int = 14,
                              smooth: int = 5,
                              factor: float = 4.236) -> pd.Series:
    delta = close.diff()
    up   = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    alpha = 1.0 / rsi_period
    roll_up   = up.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    rd = roll_down.values
    ru = roll_up.values
    safe_rd = np.where(rd != 0.0, rd, 1.0)
    rsi_arr = np.where(rd != 0.0, 100.0 - 100.0 / (1.0 + ru / safe_rd), 100.0)
    rsi = pd.Series(rsi_arr, index=close.index)

    rsi_ma = rsi.rolling(window=smooth, min_periods=1).mean()
    rsi_delta = rsi_ma.diff().abs().fillna(0.0)
    atr_rsi = rsi_delta.ewm(alpha=1.0 / smooth, adjust=False, min_periods=1).mean()

    v1_arr  = rsi_ma.values
    atr_arr = atr_rsi.values
    n = len(v1_arr)
    v2_arr = np.empty(n)
    v2_arr[0] = v1_arr[0]
    for i in range(1, n):
        prev_trail  = v2_arr[i - 1]
        prev_value1 = v1_arr[i - 1]
        direction = 1.0 if prev_value1 > prev_trail else -1.0
        v2_arr[i] = prev_trail + direction * factor * atr_arr[i]
    return pd.Series(v2_arr, index=close.index)


def ind_lrsi_tsi_qqe_stc_stc(close: pd.Series,
                              fast: int = 23,
                              slow: int = 50,
                              cycle: int = 10,
                              smooth: int = 3) -> pd.Series:
    fast_ema  = close.ewm(span=fast,  adjust=False, min_periods=1).mean()
    slow_ema  = close.ewm(span=slow,  adjust=False, min_periods=1).mean()
    macd_line = fast_ema - slow_ema

    low_macd  = macd_line.rolling(cycle, min_periods=1).min()
    high_macd = macd_line.rolling(cycle, min_periods=1).max()
    rng       = (high_macd - low_macd).values
    num       = (macd_line - low_macd).values
    safe_rng  = np.where(rng != 0.0, rng, 1.0)
    stoch_arr = np.where(rng != 0.0, 100.0 * num / safe_rng, 0.0)
    stoch_macd = pd.Series(stoch_arr, index=close.index)

    stc = stoch_macd.ewm(span=smooth, adjust=False, min_periods=1).mean() \
                    .ewm(span=smooth, adjust=False, min_periods=1).mean()
    return stc


# ─────────────────────────────────────────────
# Param ranges  (≤ 1000 combos)
# Strategy selection: 1 = LRSI+TSI, 2 = QQE+STC
# We expose both strategies via the same function
# with a "strategy_id" toggle (1 or 2).
# ─────────────────────────────────────────────

strategy_lrsi_tsi_qqe_stc_param_ranges = {
    'gamma_range'     : range(3, 8, 2),      # /10 → 0.3, 0.5, 0.7
    'tsi_long_range'  : range(20, 31, 5),    # 20, 25, 30
    'tsi_short_range' : range(10, 16, 5),    # 10, 15
    'qqe_factor_range': range(38, 49, 5),    # /10 → 3.8, 4.3, 4.8
    'stc_cycle_range' : range(8, 14, 3),     # 8, 11
    'stc_fast_range'  : range(20, 28, 4),    # 20, 24
}
# 3 * 3 * 2 * 3 * 2 * 2 = 216  ✓


def strategy_lrsi_tsi_qqe_stc(data: pd.DataFrame,
                               params: dict,
                               year: int | None = None):
    gamma      = params.get('gamma_range') / 10.0
    tsi_long   = params.get('tsi_long_range')
    tsi_short  = params.get('tsi_short_range')
    qqe_factor = params.get('qqe_factor_range') / 10.0
    stc_cycle  = params.get('stc_cycle_range')
    stc_fast   = params.get('stc_fast_range')

    df = data.copy()
    close = df['Close']

    # ── Strategy 1 signals ──
    lrsi    = ind_lrsi_tsi_qqe_stc_laguerre_rsi(close, gamma=gamma)
    lrsi_rising = lrsi > lrsi.shift(1)

    tsi = ind_lrsi_tsi_qqe_stc_tsi(close, long=tsi_long, short=tsi_short, signal=7)
    tsi_cross_above_zero = (tsi > 0.0) & (tsi.shift(1) <= 0.0)

    # ── Strategy 2 signals ──
    qqe_v2  = ind_lrsi_tsi_qqe_stc_qqe(close, rsi_period=14, smooth=5, factor=qqe_factor)
    qqe_falling = qqe_v2 < qqe_v2.shift(5)

    stc = ind_lrsi_tsi_qqe_stc_stc(close, fast=stc_fast, slow=50, cycle=stc_cycle, smooth=3)
    stc_cross_above_25 = (stc > 25.0) & (stc.shift(1) <= 25.0)

    df['lrsi_rising']        = lrsi_rising
    df['tsi_cross_zero']     = tsi_cross_above_zero
    df['qqe_falling']        = qqe_falling
    df['stc_cross_25']       = stc_cross_above_25

    if year is not None:
        df = df[df.index.year == int(year)]

    # Combine both strategies: entry if either fires, exit if either fires
    entries = df['lrsi_rising'] | df['qqe_falling']
    exits   = df['tsi_cross_zero'] | df['stc_cross_25']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: What Wick Behavior Actually Tells You About Order Flow
# URL:   https://medium.com/@Kryptera/what-wick-behavior-actually-tells-you-about-order-flow-9a4b6e64664e
# Data:  2026-06-19 02:03
# ─────────────────────────────────────

############################
# Strategy wick_absorption
############################

import numpy as np
import pandas as pd


def ind_wick_absorption_ratio(df: pd.DataFrame,
                               atr_period: int = 14,
                               smooth_period: int = 10,
                               vol_cap_mult: float = 2.5) -> pd.Series:
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']
    vol   = df['Volume']

    upper_wick = high - np.maximum(open_.values, close.values)
    lower_wick = np.minimum(open_.values, close.values) - low
    total_range = (high - low).values + 1e-8

    absorption_raw = (lower_wick - upper_wick) / total_range
    absorption_raw = pd.Series(absorption_raw, index=df.index)

    # ATR calculation (Wilder method)
    prev_close = close.shift(1)
    hl   = (high - low).values
    hpc  = np.abs(high.values - prev_close.values)
    lpc  = np.abs(low.values  - prev_close.values)
    tr   = np.maximum(np.maximum(hl, hpc), lpc)
    tr_s = pd.Series(tr, index=df.index)

    atr_arr  = np.empty(len(tr))
    atr_arr[:] = np.nan
    atr_arr[0] = tr[0]
    for i in range(1, len(tr)):
        atr_arr[i] = (atr_arr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    atr = pd.Series(atr_arr, index=df.index)

    safe_atr = np.where(atr.values != 0, atr.values, 1.0)
    atr_norm_abs = pd.Series(absorption_raw.values / safe_atr, index=df.index)

    # Volume weighting with cap
    vol_arr = vol.values.astype(float)
    vol_med = pd.Series(vol_arr, index=df.index).rolling(smooth_period, min_periods=1).median()
    vol_cap = vol_cap_mult * vol_med.values
    vol_w   = np.minimum(vol_arr, vol_cap)
    safe_vol_w = np.where(vol_w != 0, vol_w, 1.0)
    vw_abs  = atr_norm_abs.values * vol_w

    # Rolling volume-weighted average
    def _rolling_vwavg(x):
        n = len(x)
        half = n // 2
        sig = x[:half]
        wts = x[half:]
        denom = np.sum(wts)
        if denom == 0:
            return 0.0
        return np.sum(sig * wts) / denom

    combined = pd.DataFrame({'sig': vw_abs, 'wt': vol_w}, index=df.index)
    # Concatenate as double-length array trick via numpy
    sig_arr = np.array(vw_abs, dtype=float)
    wt_arr  = np.array(vol_w,  dtype=float)

    # Manual rolling VW smooth
    out = np.empty(len(sig_arr))
    out[:] = np.nan
    for i in range(len(sig_arr)):
        start = max(0, i - smooth_period + 1)
        s = sig_arr[start:i+1]
        w = wt_arr[start:i+1]
        denom = np.sum(w)
        if denom == 0:
            out[i] = 0.0
        else:
            out[i] = np.sum(s * w) / denom

    return pd.Series(out, index=df.index)


def ind_wick_absorption_bull_bear(df: pd.DataFrame) -> tuple:
    high  = df['High']
    low   = df['Low']
    open_ = df['Open']
    close = df['Close']

    total_range = (high - low).values + 1e-8
    upper_wick  = (high - np.maximum(open_.values, close.values))
    lower_wick  = (np.minimum(open_.values, close.values) - low)

    bull_abs = pd.Series(lower_wick / total_range, index=df.index)
    bear_abs = pd.Series(upper_wick / total_range, index=df.index)
    return bull_abs, bear_abs


strategy_wick_absorption_param_ranges = {
    'atr_period_range'    : range(10, 25, 5),
    'smooth_period_range' : range(5, 21, 5),
    'vol_cap_mult_range'  : range(20, 36, 5),
    'entry_thresh_range'  : range(1, 4, 1),
    'exit_thresh_range'   : range(0, 3, 1),
}


def strategy_wick_absorption(data: pd.DataFrame, params: dict, year: int | None = None):
    atr_period    = params.get('atr_period_range')
    smooth_period = params.get('smooth_period_range')
    vol_cap_mult  = params.get('vol_cap_mult_range') / 10.0
    entry_thresh  = params.get('entry_thresh_range') / 10.0
    exit_thresh   = params.get('exit_thresh_range')  / 10.0

    df = data.copy()

    absorption = ind_wick_absorption_ratio(
        df,
        atr_period=atr_period,
        smooth_period=smooth_period,
        vol_cap_mult=vol_cap_mult
    )
    bull_abs, bear_abs = ind_wick_absorption_bull_bear(df)

    df['absorption']  = absorption
    df['bull_abs']    = bull_abs
    df['bear_abs']    = bear_abs

    # Rolling stats for dynamic thresholding
    abs_mean = df['absorption'].rolling(smooth_period, min_periods=1).mean()
    abs_std  = df['absorption'].rolling(smooth_period, min_periods=1).std(ddof=0).replace(0, np.nan)

    df['abs_mean'] = abs_mean
    df['abs_std']  = abs_std

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: absorption score crosses above positive threshold (bullish absorption dominant)
    thresh_pos = df['abs_mean'] + entry_thresh * df['abs_std'].fillna(0)
    thresh_neg = df['abs_mean'] - entry_thresh * df['abs_std'].fillna(0)

    # Long: strong bullish absorption (lower wicks dominating)
    entries_long  = (df['absorption'] > thresh_pos) & (df['bull_abs'] > df['bear_abs'])
    # Short: strong bearish absorption (upper wicks dominating)
    entries_short = (df['absorption'] < thresh_neg) & (df['bear_abs'] > df['bull_abs'])
    entries = entries_long | entries_short

    # Exit when absorption crosses back near zero
    exits_long  = df['absorption'] < exit_thresh
    exits_short = df['absorption'] > -exit_thresh
    exits = (exits_long & entries_long.shift(1).fillna(False)) | \
            (exits_short & entries_short.shift(1).fillna(False))
    exits = exits_long | exits_short

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Found Sloped LinReg Volume Profile Indicator on TradingView. Then I Backtested It.
# URL:   https://medium.com/@Kryptera/i-found-sloped-linreg-volume-profile-indicator-on-tradingview-then-i-backtested-it-2d4a9168f00c
# Data:  2026-06-19 02:04
# ─────────────────────────────────────

############################
# Strategy sloped_linreg_vp
############################

import numpy as np
import pandas as pd


def ind_sloped_linreg_vp_levels(df: pd.DataFrame,
                                 linreg_len: int = 100,
                                 n_slots: int = 50,
                                 value_area_pct: float = 0.70,
                                 channel_std: float = 2.0):
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    vol   = df['Volume'].values
    n     = len(close)

    x       = np.arange(linreg_len, dtype=float)
    x_mean  = x.mean()
    x_var   = ((x - x_mean) ** 2).sum()

    poc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    for i in range(linreg_len - 1, n):
        w_close = close[i - linreg_len + 1: i + 1]
        w_high  = high [i - linreg_len + 1: i + 1]
        w_low   = low  [i - linreg_len + 1: i + 1]
        w_vol   = vol  [i - linreg_len + 1: i + 1]

        w_mean = w_close.mean()
        cov    = ((x - x_mean) * (w_close - w_mean)).sum()
        slope  = cov / x_var if x_var != 0.0 else 0.0
        intercept = w_mean - slope * x_mean

        midline = slope * x + intercept

        stdev_vals = w_close - midline
        stdev_val  = stdev_vals.std(ddof=0)
        if stdev_val == 0.0:
            stdev_val = 1e-10

        dH = w_high - midline
        dL = w_low  - midline

        dev_max = dH.max()
        dev_min = dL.min()
        if dev_max == dev_min:
            dev_max = dev_min + 1e-10

        slot_size = (dev_max - dev_min) / n_slots
        slot_vol  = np.zeros(n_slots, dtype=float)

        for j in range(linreg_len):
            bar_h = dH[j]
            bar_l = dL[j]
            bar_range = bar_h - bar_l
            if bar_range <= 0.0:
                bar_range = 1e-10
            for s in range(n_slots):
                s_lo = dev_min + s * slot_size
                s_hi = s_lo + slot_size
                overlap = min(bar_h, s_hi) - max(bar_l, s_lo)
                if overlap > 0.0:
                    slot_vol[s] += w_vol[j] * (overlap / bar_range)

        poc_slot = int(np.argmax(slot_vol))
        total_vol = slot_vol.sum()

        target_vol = value_area_pct * total_vol
        accum = slot_vol[poc_slot]
        lo_idx = poc_slot
        hi_idx = poc_slot

        while accum < target_vol:
            lo_next = lo_idx - 1
            hi_next = hi_idx + 1
            can_lo  = lo_next >= 0
            can_hi  = hi_next < n_slots
            if not can_lo and not can_hi:
                break
            vol_lo = slot_vol[lo_next] if can_lo else -1.0
            vol_hi = slot_vol[hi_next] if can_hi else -1.0
            if vol_lo >= vol_hi:
                lo_idx = lo_next
                accum += slot_vol[lo_idx]
            else:
                hi_idx = hi_next
                accum += slot_vol[hi_idx]

        current_midline = slope * (linreg_len - 1) + intercept

        poc_dev = dev_min + (poc_slot + 0.5) * slot_size
        vah_dev = dev_min + (hi_idx + 1.0)   * slot_size
        val_dev = dev_min + lo_idx            * slot_size

        poc_arr[i] = current_midline + poc_dev
        vah_arr[i] = current_midline + vah_dev
        val_arr[i] = current_midline + val_dev

    poc = pd.Series(poc_arr, index=df.index)
    vah = pd.Series(vah_arr, index=df.index)
    val = pd.Series(val_arr, index=df.index)
    return poc, vah, val


strategy_sloped_linreg_vp_param_ranges = {
    'linreg_len_range'   : range(60, 141, 40),
    'n_slots_range'      : range(30, 71, 20),
    'value_area_range'   : range(60, 81, 10),
    'channel_std_range'  : range(15, 31, 5),
}


def strategy_sloped_linreg_vp(data: pd.DataFrame, params: dict, year: int | None = None):
    linreg_len     = params.get('linreg_len_range')
    n_slots        = params.get('n_slots_range')
    value_area_pct = params.get('value_area_range') / 100.0
    channel_std    = params.get('channel_std_range') / 10.0

    df = data.copy()

    poc, vah, val = ind_sloped_linreg_vp_levels(
        df,
        linreg_len=linreg_len,
        n_slots=n_slots,
        value_area_pct=value_area_pct,
        channel_std=channel_std,
    )

    df['POC'] = poc
    df['VAH'] = vah
    df['VAL'] = val

    if year is not None:
        df = df[df.index.year == int(year)]

    close     = df['Close']
    poc_s     = df['POC']
    vah_s     = df['VAH']
    val_s     = df['VAL']

    cross_above_poc = (close > poc_s) & (close.shift(1) <= poc_s.shift(1))
    entries = cross_above_poc

    exit_above_vah = close > vah_s
    exit_below_val = close < val_s
    exits = exit_above_vah | exit_below_val

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: Can a Pressure-Based Oscillator
Beat Buy & Hold Across 40 Stocks?
# URL:   https://medium.com/@Kryptera/can-a-pressure-based-oscillator-beat-buy-hold-across-40-stocks-ccec005d57ba
# Data:  2026-06-19 02:04
# ─────────────────────────────────────

############################
# Strategy atp_pressure
############################

import numpy as np
import pandas as pd


def ind_atp_pressure_absorption_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_ = df['Open'].values

    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    candle_range = high - low
    safe_range = np.where(candle_range != 0, candle_range, 1.0)

    bull_absorption = lower_wick / safe_range
    bear_absorption = upper_wick / safe_range

    diff = bull_absorption - bear_absorption

    diff_series = pd.Series(diff, index=df.index)
    ar = diff_series.rolling(period, min_periods=1).mean()
    return ar


def ind_atp_pressure_band_pressure(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    high = df['High'].values
    low = df['Low'].values
    close_arr = close.values
    prev_close = np.empty(len(close_arr))
    prev_close[0] = close_arr[0]
    prev_close[1:] = close_arr[:-1]

    tr = np.maximum(
        np.maximum(high - low, np.abs(high - prev_close)),
        np.abs(low - prev_close)
    )

    # Wilder RMA for ATR
    atr_arr = np.empty(len(tr))
    atr_arr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr_arr[i] = atr_arr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    # Wilder RMA for price (basis)
    rma_arr = np.empty(len(close_arr))
    rma_arr[0] = close_arr[0]
    for i in range(1, len(close_arr)):
        rma_arr[i] = rma_arr[i - 1] * (1.0 - alpha) + close_arr[i] * alpha

    safe_atr = np.where(atr_arr != 0, atr_arr, 1.0)
    bp = (close_arr - rma_arr) / safe_atr

    return pd.Series(bp, index=df.index)


def ind_atp_pressure_body_momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close'].values
    open_ = df['Open'].values

    body = np.abs(close - open_)
    body_series = pd.Series(body, index=df.index)
    body_dir = np.where(close >= open_, body, -body)
    body_dir_series = pd.Series(body_dir, index=df.index)

    body_avg = body_series.rolling(period, min_periods=1).mean()
    safe_avg = body_avg.replace(0, np.nan).fillna(1.0)
    bm = body_dir_series / safe_avg
    return bm


def ind_atp_pressure_composite(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.Series:
    ar = ind_atp_pressure_absorption_ratio(df, period)
    bp = ind_atp_pressure_band_pressure(df, period)
    bm = ind_atp_pressure_body_momentum(df, period)

    # Normalize each component using rolling z-score
    def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
        roll_mean = s.rolling(win, min_periods=1).mean()
        roll_std = s.rolling(win, min_periods=1).std(ddof=0).replace(0, np.nan).fillna(1.0)
        return (s - roll_mean) / roll_std

    norm_win = max(period * 2, 20)
    ar_n = rolling_zscore(ar, norm_win)
    bp_n = rolling_zscore(bp, norm_win)
    bm_n = rolling_zscore(bm, norm_win)

    composite = 0.4 * ar_n + 0.4 * bp_n + 0.2 * bm_n

    # Smooth
    atp = composite.ewm(span=smooth, adjust=False, min_periods=1).mean()
    return atp


strategy_atp_pressure_param_ranges = {
    'period_range': range(10, 25, 7),        # [10, 17, 24]
    'smooth_range': range(3, 10, 3),         # [3, 6, 9]
    'long_thresh_range': range(3, 10, 3),    # [3, 6, 9]  -> divide by 10
    'exit_thresh_range': range(-6, 1, 3),    # [-6, -3, 0] -> divide by 10
}


def strategy_atp_pressure(data: pd.DataFrame, params: dict, year: int | None = None):
    period = params.get('period_range')
    smooth = params.get('smooth_range')
    long_thresh = params.get('long_thresh_range') / 10.0
    exit_thresh = params.get('exit_thresh_range') / 10.0

    df = data.copy()

    atp = ind_atp_pressure_composite(df, period=period, smooth=smooth)
    df['ATP'] = atp

    if year is not None:
        df = df[df.index.year == int(year)]

    atp_val = df['ATP']
    atp_prev = atp_val.shift(1)

    # Entry: ATP crosses above long_thresh
    entries = (atp_val >= long_thresh) & (atp_prev < long_thresh)

    # Exit: ATP drops below exit_thresh
    exits = (atp_val <= exit_thresh) & (atp_prev > exit_thresh)

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Found a Phase Exhaustion Reversal TradingView Indicator, Backtested It on 1 Random Stocks
# URL:   https://medium.com/@Kryptera/i-found-a-phase-exhaustion-reversal-tradingview-indicator-backtested-it-on-1-random-stocks-ec24b7771f48
# Data:  2026-06-19 02:06
# ─────────────────────────────────────

############################
# Strategy pxr
############################

import numpy as np
import pandas as pd


def ind_pxr_efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    """Kaufman Efficiency Ratio for a given period."""
    arr = close.values
    n = len(arr)
    er = np.zeros(n)
    for i in range(period, n):
        window = arr[i - period: i + 1]
        transport = abs(window[-1] - window[0])
        agitation = np.sum(np.abs(np.diff(window)))
        if agitation != 0.0:
            er[i] = transport / agitation
        else:
            er[i] = 0.0
    # sign by direction
    direction = np.sign(arr - np.roll(arr, period))
    direction[:period] = 0.0
    signed_er = er * direction
    return pd.Series(signed_er, index=close.index)


def ind_pxr_phase_gap(df: pd.DataFrame,
                      short_period: int = 7,
                      medium_period: int = 24,
                      peak_window: int = 10,
                      gap_threshold: float = 0.4) -> tuple:
    close = df['Close']

    er_short = ind_pxr_efficiency_ratio(close, short_period)
    er_med = ind_pxr_efficiency_ratio(close, medium_period)

    # Phase gap = short ER minus medium ER
    phase_gap = er_short - er_med

    # Normalized phase gap using rolling max abs
    abs_gap = phase_gap.abs()
    rolling_max = abs_gap.rolling(peak_window, min_periods=1).max()
    safe_max = rolling_max.replace(0.0, 1.0)
    norm_gap = phase_gap / safe_max

    # Regime classification based on medium ER
    er_med_abs = er_med.abs()
    regime = pd.Series('NEUTRAL', index=close.index)
    regime = regime.where(er_med_abs < 0.15, 'NEUTRAL')
    regime[er_med_abs >= 0.55] = 'EXTENDED'
    regime[(er_med_abs >= 0.25) & (er_med_abs < 0.55)] = 'COMPRESSED'
    regime[(norm_gap.abs() >= 0.65) & (er_med_abs >= 0.15) & (er_med_abs < 0.55)] = 'TRANSITIONING'
    # Threshold per regime
    threshold = pd.Series(gap_threshold, index=close.index)
    threshold[regime == 'EXTENDED'] = gap_threshold * 0.85
    threshold[regime == 'COMPRESSED'] = gap_threshold * 1.20
    threshold[regime == 'TRANSITIONING'] = gap_threshold * 1.00
    threshold[regime == 'NEUTRAL'] = 9999.0  # never fires

    return er_short, er_med, phase_gap, norm_gap, regime, threshold


def strategy_pxr(data: pd.DataFrame, params: dict, year: int | None = None):
    short_p = params.get('short_range')
    medium_p = params.get('medium_range')
    peak_w = params.get('peak_window_range')
    gap_thr = params.get('gap_threshold_range') / 100.0

    df = data.copy()

    er_short, er_med, phase_gap, norm_gap, regime, threshold = ind_pxr_phase_gap(
        df,
        short_period=short_p,
        medium_period=medium_p,
        peak_window=peak_w,
        gap_threshold=gap_thr
    )

    df['er_short'] = er_short
    df['er_med'] = er_med
    df['phase_gap'] = phase_gap
    df['norm_gap'] = norm_gap
    df['regime'] = regime
    df['threshold'] = threshold

    if year is not None:
        df = df[df.index.year == int(year)]

    # Peak-recede logic:
    # Phase gap was stretched beyond threshold (burst), now collapsing back (exhaustion)
    # Long signal: norm_gap was negative (short ER < med ER) and now recovering toward 0
    #   i.e., prev norm_gap < -threshold and current norm_gap > prev norm_gap (gap collapsing)
    #   and medium ER > 0 (structural uptrend)
    # Short signal: norm_gap was positive (short ER > med ER) and now collapsing
    #   and medium ER < 0 (structural downtrend)

    ng = df['norm_gap']
    er_m = df['er_med']
    thr = df['threshold']
    reg = df['regime']

    not_neutral = reg != 'NEUTRAL'

    # Burst exhaustion: gap was stretched, now receding
    gap_was_stretched_neg = ng.shift(1) < -thr.shift(1)
    gap_receding_from_neg = ng > ng.shift(1)  # gap getting less negative = collapsing
    long_signal = gap_was_stretched_neg & gap_receding_from_neg & (er_m > 0) & not_neutral

    gap_was_stretched_pos = ng.shift(1) > thr.shift(1)
    gap_receding_from_pos = ng < ng.shift(1)  # gap getting less positive = collapsing
    short_signal = gap_was_stretched_pos & gap_receding_from_pos & (er_m < 0) & not_neutral

    entries = long_signal | short_signal

    # Exit: phase gap crosses zero again (realignment complete) or regime goes NEUTRAL
    exit_gap_zero = (ng * ng.shift(1)) < 0  # sign change in norm_gap
    exit_neutral = reg == 'NEUTRAL'
    # Exit long when medium ER turns negative
    exit_long = (er_m < 0) & (ng.shift(1) < 0)
    # Exit short when medium ER turns positive
    exit_short = (er_m > 0) & (ng.shift(1) > 0)

    exits = exit_gap_zero | exit_neutral | exit_long | exit_short

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


strategy_pxr_param_ranges = {
    'short_range': range(5, 12, 3),        # [5, 8, 11]
    'medium_range': range(18, 33, 7),      # [18, 25, 32]
    'peak_window_range': range(8, 17, 4),  # [8, 12, 16]
    'gap_threshold_range': range(30, 55, 12),  # [30, 42, 54]
}


# ─────────────────────────────────────
# Fonte: I Built a Trading Strategy Generator — Then Realized Backtests Weren’t Enough
# URL:   https://medium.com/@Kryptera/i-built-a-trading-strategy-generator-then-realized-backtests-werent-enough-d3489940e874
# Data:  2026-06-19 02:06
# ─────────────────────────────────────

############################
# Strategy tema_vidya
############################

import pandas as pd
import numpy as np


def ind_tema_vidya_tema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    close = df['Close']
    ema1 = close.ewm(span=period, adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=1).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=1).mean()
    tema = 3 * (ema1 - ema2) + ema3
    return tema


def ind_tema_vidya_vidya(df: pd.DataFrame, period: int = 14, alpha: float = 0.2) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    vidya = np.empty(n)
    vidya[0] = close[0]

    diffs = np.abs(np.diff(close))
    base_vol = np.mean(diffs) if len(diffs) > 0 else 1.0
    safe_base = base_vol if base_vol != 0.0 else 1.0

    for i in range(1, n):
        start = max(0, i - period)
        window = close[start:i + 1]
        w_diffs = np.abs(np.diff(window))
        vol = np.mean(w_diffs) if len(w_diffs) > 0 else 0.0
        vol_ratio = np.clip(vol / safe_base, 0.2, 2.0)
        a = alpha * vol_ratio
        vidya[i] = a * close[i] + (1.0 - a) * vidya[i - 1]

    return pd.Series(vidya, index=df.index)


strategy_tema_vidya_param_ranges = {
    'tema_period_range': range(10, 31, 10),
    'tema_shift_range' : range(3, 9, 3),
    'vidya_period_range': range(7, 22, 7),
    'vidya_alpha_range' : range(1, 4, 1),
}


def strategy_tema_vidya(data: pd.DataFrame, params: dict, year: int | None = None):
    tema_period  = params.get('tema_period_range')
    tema_shift   = params.get('tema_shift_range')
    vidya_period = params.get('vidya_period_range')
    vidya_alpha  = params.get('vidya_alpha_range') / 10.0

    df = data.copy()

    tema  = ind_tema_vidya_tema(df, period=tema_period)
    vidya = ind_tema_vidya_vidya(df, period=vidya_period, alpha=vidya_alpha)

    df['TEMA']  = tema
    df['VIDYA'] = vidya

    if year is not None:
        df = df[df.index.year == int(year)]

    tema_rising = df['TEMA'] > df['TEMA'].shift(tema_shift)

    open_below_vidya_after_above = (
        (df['Open'] < df['VIDYA']) &
        (df['Open'].shift(1) > df['VIDYA'].shift(1))
    )

    entries = tema_rising
    exits   = open_below_vidya_after_above

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Ran 1,000 Simulated Futures on This FANG Trading Strategy. Here’s What I Found.
# URL:   https://medium.com/@Kryptera/i-ran-1-000-simulated-futures-on-this-fang-trading-strategy-heres-what-i-found-4394cafc2058
# Data:  2026-06-20 02:01
# ─────────────────────────────────────

############################
# Strategy fang_hma_qqe
############################

import pandas as pd
import numpy as np


def ind_fang_hma_qqe_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    result = series.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / w_sum, raw=True
    )
    return result


def ind_fang_hma_qqe_hma(df: pd.DataFrame, period: int) -> pd.Series:
    close = df['Close']
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    wma_full = ind_fang_hma_qqe_wma(close, period)
    wma_half = ind_fang_hma_qqe_wma(close, half_period)
    diff = 2.0 * wma_half - wma_full
    hma = ind_fang_hma_qqe_wma(diff, sqrt_period)
    return hma


def ind_fang_hma_qqe_qqe(df: pd.DataFrame,
                          qqe_period: int = 14,
                          qqe_smooth: int = 5,
                          qqe_factor: float = 1.618,
                          qqe_shift: int = 6) -> pd.Series:
    close = df['Close']
    n = len(close)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    gain_arr = gain.values
    loss_arr = loss.values
    avg_gain = np.empty(n)
    avg_loss = np.empty(n)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    # seed with simple average
    if n >= qqe_period:
        avg_gain[qqe_period - 1] = np.mean(gain_arr[1:qqe_period])
        avg_loss[qqe_period - 1] = np.mean(loss_arr[1:qqe_period])
        for i in range(qqe_period, n):
            avg_gain[i] = (avg_gain[i - 1] * (qqe_period - 1) + gain_arr[i]) / qqe_period
            avg_loss[i] = (avg_loss[i - 1] * (qqe_period - 1) + loss_arr[i]) / qqe_period

    safe_avg_loss = np.where(avg_loss != 0, avg_loss, 1.0)
    rs = np.where(avg_loss != 0, avg_gain / safe_avg_loss, 0.0)
    rsi_arr = 100.0 - 100.0 / (1.0 + rs)
    rsi = pd.Series(rsi_arr, index=close.index)

    # Smooth RSI with EMA (qqe_smooth times)
    smoothed = rsi.ewm(span=qqe_smooth, adjust=False, min_periods=1).mean()
    for _ in range(qqe_smooth - 1):
        smoothed = smoothed.ewm(span=qqe_smooth, adjust=False, min_periods=1).mean()

    # Apply shift
    value1 = smoothed.shift(qqe_shift)

    return value1


def ind_fang_hma_qqe_qqe_slope(value1: pd.Series) -> pd.Series:
    slope = value1.diff()
    return slope


strategy_fang_hma_qqe_param_ranges = {
    'hma_fast_range': range(16, 33, 8),
    'hma_slow_range': range(40, 81, 20),
    'qqe_period_range': range(10, 20, 4),
    'qqe_smooth_range': range(3, 8, 2),
    'qqe_factor_range': range(14, 22, 4),
    'qqe_shift_range': range(4, 9, 2),
}


def strategy_fang_hma_qqe(data: pd.DataFrame, params: dict, year: int | None = None):
    hma_fast_p = params.get('hma_fast_range')
    hma_slow_p = params.get('hma_slow_range')
    qqe_period_p = params.get('qqe_period_range')
    qqe_smooth_p = params.get('qqe_smooth_range')
    qqe_factor_p = params.get('qqe_factor_range') / 10.0
    qqe_shift_p = params.get('qqe_shift_range')

    df = data.copy()

    hma_fast = ind_fang_hma_qqe_hma(df, period=hma_fast_p)
    hma_slow = ind_fang_hma_qqe_hma(df, period=hma_slow_p)

    value1 = ind_fang_hma_qqe_qqe(
        df,
        qqe_period=qqe_period_p,
        qqe_smooth=qqe_smooth_p,
        qqe_factor=qqe_factor_p,
        qqe_shift=qqe_shift_p
    )
    qqe_slope = ind_fang_hma_qqe_qqe_slope(value1)

    df['HMA_Fast'] = hma_fast
    df['HMA_Slow'] = hma_slow
    df['QQE_Value1'] = value1
    df['QQE_Slope'] = qqe_slope

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: QQE Value1 slope negative
    entries = df['QQE_Slope'] < 0

    # Exit: fast HMA crosses below slow HMA
    hma_fast_s = df['HMA_Fast']
    hma_slow_s = df['HMA_Slow']
    cross_below = (hma_fast_s < hma_slow_s) & (hma_fast_s.shift(1) >= hma_slow_s.shift(1))
    exits = cross_below

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: The Strategy That Lied Beautifully
# URL:   https://medium.com/@Kryptera/the-strategy-that-lied-beautifully-12c51bb272d8
# Data:  2026-06-20 02:01
# ─────────────────────────────────────

############################
# Strategy stx_wma_trix
############################

import pandas as pd
import numpy as np


def ind_stx_wma_trix_wma(df: pd.DataFrame, wma_period: int = 20) -> pd.Series:
    close = df['Close']
    weights = np.arange(1, wma_period + 1, dtype=float)
    wsum = weights.sum()
    wma = close.rolling(wma_period).apply(
        lambda x: np.dot(x, weights) / wsum,
        raw=True
    )
    return wma


def ind_stx_wma_trix_trix(df: pd.DataFrame, trix_period: int = 15) -> pd.Series:
    close = df['Close']
    ema1 = close.ewm(span=trix_period, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_period, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_period, adjust=False).mean()
    ema3_prev = ema3.shift(1)
    safe_den = np.where(ema3_prev.values != 0, ema3_prev.values, 1.0)
    trix_vals = np.where(
        ema3_prev.values != 0,
        (ema3.values - ema3_prev.values) / safe_den * 100,
        np.nan
    )
    trix = pd.Series(trix_vals, index=close.index)
    return trix


strategy_stx_wma_trix_param_ranges = {
    'wma_period_range': range(16, 25, 4),   # [16, 20, 24]
    'trix_period_range': range(12, 19, 3),  # [12, 15, 18]
}


def strategy_stx_wma_trix(data: pd.DataFrame, params: dict, year: int | None = None):
    wma_period = params.get('wma_period_range')
    trix_period = params.get('trix_period_range')

    df = data.copy()

    wma = ind_stx_wma_trix_wma(df, wma_period=wma_period)
    trix = ind_stx_wma_trix_trix(df, trix_period=trix_period)

    df['WMA'] = wma
    df['TRIX'] = trix

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: Close crosses below WMA (short-side momentum entry)
    close_below_wma = df['Close'] < df['WMA']
    close_prev_above_wma = df['Close'].shift(1) >= df['WMA'].shift(1)
    entries = close_below_wma & close_prev_above_wma

    # Exit: TRIX crosses below zero
    trix_prev_above_zero = df['TRIX'].shift(1) >= 0
    trix_below_zero = df['TRIX'] < 0
    exits = trix_prev_above_zero & trix_below_zero

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Found a Whale Liquidity and Absorption ProfileTradingView Indicator That Led Me Down a Python…
# URL:   https://medium.com/@Kryptera/i-found-a-whale-liquidity-and-absorption-profiletradingview-indicator-that-led-me-down-a-python-8839de7d4b9c
# Data:  2026-06-20 02:02
# ─────────────────────────────────────

############################
# Strategy whale_absorption
############################

import pandas as pd
import numpy as np


def ind_whale_absorption_profile(
    df: pd.DataFrame,
    lookback: int = 200,
    n_bins: int = 35,
    strong_pct: float = 97.0,
    abs_threshold: float = 0.40,
    delta_threshold: float = 0.40,
    grace_window: int = 3,
):
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    open_ = df['Open'].values
    vol   = df['Volume'].values
    n     = len(close)

    bar_range    = high - low
    upper_wick   = np.where(close >= open_, high - close, high - open_)
    lower_wick   = np.where(close >= open_, open_ - low,  close - low)
    bar_dir      = np.where(close >= open_, 1, -1)

    safe_range       = np.where(bar_range != 0, bar_range, 1.0)
    upper_wick_frac  = np.clip(upper_wick / safe_range, 0.0, 1.0)
    lower_wick_frac  = np.clip(lower_wick / safe_range, 0.0, 1.0)
    absorption_vol   = np.where(
        bar_dir == 1,
        vol * upper_wick_frac,
        vol * lower_wick_frac,
    )

    bull_vol   = np.where(bar_dir == 1, vol, 0.0)
    bear_vol   = np.where(bar_dir == -1, vol, 0.0)

    delta_signal     = np.zeros(n, dtype=np.float64)
    absorption_signal = np.zeros(n, dtype=np.float64)

    for t in range(lookback, n):
        w_start = t - lookback
        w_end   = t  # exclusive → bars [w_start, t-1]

        w_close   = close[w_start:w_end]
        w_vol     = vol[w_start:w_end]
        w_bull    = bull_vol[w_start:w_end]
        w_bear    = bear_vol[w_start:w_end]
        w_abs     = absorption_vol[w_start:w_end]

        p_min = w_close.min()
        p_max = w_close.max()
        if p_max == p_min:
            continue

        bin_edges  = np.linspace(p_min, p_max, n_bins + 1)
        bin_idx    = np.searchsorted(bin_edges, w_close, side='right') - 1
        bin_idx    = np.clip(bin_idx, 0, n_bins - 1)

        strong_thresh = np.percentile(w_vol, strong_pct)

        bin_strong_bull = np.zeros(n_bins, dtype=np.float64)
        bin_weak_bull   = np.zeros(n_bins, dtype=np.float64)
        bin_strong_bear = np.zeros(n_bins, dtype=np.float64)
        bin_weak_bear   = np.zeros(n_bins, dtype=np.float64)
        bin_absorption  = np.zeros(n_bins, dtype=np.float64)

        for b in range(len(w_close)):
            bi = bin_idx[b]
            bv = w_vol[b]
            if w_bull[b] > 0:
                if bv >= strong_thresh:
                    bin_strong_bull[bi] += bv
                else:
                    bin_weak_bull[bi] += bv
            else:
                if bv >= strong_thresh:
                    bin_strong_bear[bi] += bv
                else:
                    bin_weak_bear[bi] += bv
            bin_absorption[bi] += w_abs[b]

        bin_delta = (bin_strong_bull + bin_weak_bull) - (bin_strong_bear + bin_weak_bear)

        cur_price = close[t]
        cur_bin   = np.searchsorted(bin_edges, cur_price, side='right') - 1
        cur_bin   = int(np.clip(cur_bin, 0, n_bins - 1))

        max_abs   = bin_absorption.max()
        max_delta = np.abs(bin_delta).max()

        if max_abs > 0:
            absorption_signal[t] = bin_absorption[cur_bin] / max_abs
        if max_delta > 0:
            delta_signal[t] = bin_delta[cur_bin] / max_delta

    absorption_series = pd.Series(absorption_signal, index=df.index)
    delta_series      = pd.Series(delta_signal,      index=df.index)
    return absorption_series, delta_series


strategy_whale_absorption_param_ranges = {
    'lookback_range'    : range(100, 251, 50),   # 3 values: 100, 150, 200, 250
    'n_bins_range'      : range(25, 46, 10),     # 3 values: 25, 35, 45
    'abs_thresh_range'  : range(30, 51, 10),     # 3 values: 30, 40, 50
    'delta_thresh_range': range(30, 51, 10),     # 3 values: 30, 40, 50
    'grace_range'       : range(1, 5, 1),        # 4 values: 1, 2, 3, 4
}
# Total: 4 * 3 * 3 * 3 * 4 = 432 combinations


def strategy_whale_absorption(data: pd.DataFrame, params: dict, year: int | None = None):
    lookback      = params.get('lookback_range', 200)
    n_bins        = params.get('n_bins_range', 35)
    abs_thresh    = params.get('abs_thresh_range', 40) / 100.0
    delta_thresh  = params.get('delta_thresh_range', 40) / 100.0
    grace_window  = params.get('grace_range', 3)

    df = data.copy()

    absorption_s, delta_s = ind_whale_absorption_profile(
        df,
        lookback=lookback,
        n_bins=n_bins,
        strong_pct=97.0,
        abs_threshold=abs_thresh,
        delta_threshold=delta_thresh,
        grace_window=grace_window,
    )

    df['absorption'] = absorption_s
    df['delta']      = delta_s

    if year is not None:
        df = df[df.index.year == int(year)]

    abs_gate   = df['absorption'] >= abs_thresh
    delta_gate = df['delta'].abs() >= delta_thresh
    both_gates = abs_gate & delta_gate

    bullish_delta = df['delta'] > 0
    bearish_delta = df['delta'] < 0

    raw_entries = both_gates & bullish_delta
    raw_exits   = both_gates & bearish_delta

    # Grace window: extend signal if any of the last grace_window bars triggered
    grace_entries = raw_entries.copy()
    grace_exits   = raw_exits.copy()
    for g in range(1, grace_window + 1):
        grace_entries = grace_entries | raw_entries.shift(g).fillna(False)
        grace_exits   = grace_exits   | raw_exits.shift(g).fillna(False)

    entries = grace_entries.astype(bool)
    exits   = grace_exits.astype(bool)

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Generated a Trading Strategy Using My Python Bundle — Then Tried to Break It
# URL:   https://medium.com/@Kryptera/i-generated-a-trading-strategy-using-my-python-bundle-then-tried-to-break-it-b8de9fc402c3
# Data:  2026-06-20 02:02
# ─────────────────────────────────────

############################
# Strategy bb_tsi
############################

def ind_bb_tsi_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0, shift: int = 5):
    close = df['Close']
    ma = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    bb_upper = ma + std_mult * std
    bb_upper_rising = bb_upper > bb_upper.shift(shift)
    return bb_upper, bb_upper_rising


def ind_bb_tsi_tsi(df: pd.DataFrame, long: int = 25, short: int = 13, signal: int = 7):
    close = df['Close']
    momentum = close.diff()
    ema1 = momentum.ewm(span=long, adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=short, adjust=False, min_periods=1).mean()
    abs_ema1 = momentum.abs().ewm(span=long, adjust=False, min_periods=1).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False, min_periods=1).mean()
    abs_ema2_arr = abs_ema2.values
    ema2_arr = ema2.values
    safe_den = np.where(abs_ema2_arr != 0, abs_ema2_arr, 1.0)
    tsi_arr = np.where(abs_ema2_arr != 0, 100.0 * ema2_arr / safe_den, 0.0)
    tsi = pd.Series(tsi_arr, index=close.index)
    tsi_signal = tsi.ewm(span=signal, adjust=False, min_periods=1).mean()
    return tsi, tsi_signal


strategy_bb_tsi_param_ranges = {
    'bb_period_range' : range(16, 25, 4),
    'bb_shift_range'  : range(4, 7, 1),
    'bb_std_range'    : range(1, 4, 1),
    'tsi_long_range'  : range(20, 31, 5),
    'tsi_short_range' : range(10, 17, 3),
    'tsi_signal_range': range(6, 9, 1),
}


def strategy_bb_tsi(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period  = params.get('bb_period_range')
    bb_shift   = params.get('bb_shift_range')
    bb_std     = float(params.get('bb_std_range'))
    tsi_long   = params.get('tsi_long_range')
    tsi_short  = params.get('tsi_short_range')
    tsi_signal = params.get('tsi_signal_range')

    df = data.copy()

    _, bb_upper_rising = ind_bb_tsi_bollinger(df, period=bb_period, std_mult=bb_std, shift=bb_shift)
    tsi, _ = ind_bb_tsi_tsi(df, long=tsi_long, short=tsi_short, signal=tsi_signal)

    df['BB_Upper_Rising'] = bb_upper_rising
    df['TSI'] = tsi

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['BB_Upper_Rising']

    tsi_cross_below_zero = (df['TSI'] < 0) & (df['TSI'].shift(1) >= 0)
    exits = tsi_cross_below_zero

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: Why “Just Pick a Good Strategy” Isn’t Enough
# URL:   https://medium.com/@Kryptera/why-just-pick-a-good-strategy-isnt-enough-e6c007aed4f8
# Data:  2026-06-21 02:00
# ─────────────────────────────────────

############################
# Strategy tema_vidya_momentum_rsi
############################

import pandas as pd
import numpy as np


def ind_tema_vidya_momentum_rsi_momentum(df: pd.DataFrame, period: int) -> pd.Series:
    close = df['Close']
    safe_shift = close.shift(period)
    safe_den = np.where(safe_shift != 0, safe_shift.values, 1.0)
    mom = np.where(safe_shift.values != 0, close.values / safe_den * 100, 100.0)
    return pd.Series(mom, index=df.index)


def ind_tema_vidya_momentum_rsi_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(span=period, adjust=False, min_periods=1).mean()
    safe_den = np.where(roll_down.values != 0, roll_down.values, 1.0)
    rs = np.where(roll_down.values != 0, roll_up.values / safe_den, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=df.index)


def ind_tema_vidya_momentum_rsi_tema(df: pd.DataFrame, period: int) -> pd.Series:
    close = df['Close']
    ema1 = close.ewm(span=period, adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=1).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=1).mean()
    tema = 3.0 * (ema1 - ema2) + ema3
    return tema


def ind_tema_vidya_momentum_rsi_vidya(df: pd.DataFrame, period: int, alpha: float) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    vidya = np.empty(n)
    vidya[0] = close[0]

    diffs = np.abs(np.diff(close))
    base_vol = np.mean(diffs) if len(diffs) > 0 else 1.0
    if base_vol == 0:
        base_vol = 1.0

    for i in range(1, n):
        start = max(0, i - period)
        window = close[start:i + 1]
        if len(window) > 1:
            vol = np.mean(np.abs(np.diff(window)))
        else:
            vol = 0.0
        vol_ratio = vol / base_vol
        vol_ratio = min(max(vol_ratio, 0.2), 2.0)
        a = alpha * vol_ratio
        vidya[i] = a * close[i] + (1.0 - a) * vidya[i - 1]

    return pd.Series(vidya, index=df.index)


strategy_tema_vidya_momentum_rsi_param_ranges = {
    'momentum_fast_range': range(5, 15, 5),
    'momentum_slow_range': range(10, 25, 5),
    'rsi_period_range':    range(10, 21, 5),
    'rsi_level_range':     range(25, 40, 5),
    'tema_period_range':   range(15, 30, 5),
    'vidya_period_range':  range(10, 21, 5),
}


def strategy_tema_vidya_momentum_rsi(data: pd.DataFrame, params: dict, year: int | None = None):
    mom_fast   = params.get('momentum_fast_range')
    mom_slow   = params.get('momentum_slow_range')
    rsi_period = params.get('rsi_period_range')
    rsi_level  = params.get('rsi_level_range')
    tema_p     = params.get('tema_period_range')
    vidya_p    = params.get('vidya_period_range')
    vidya_alpha = 0.2
    tema_shift  = 5

    df = data.copy()

    # AMD-like: Momentum-RSI signals
    fast_mom = ind_tema_vidya_momentum_rsi_momentum(df, mom_fast)
    slow_mom = ind_tema_vidya_momentum_rsi_momentum(df, mom_slow)
    # Entry: fast momentum crosses down below slow momentum
    mom_cross_down = (fast_mom < slow_mom) & (fast_mom.shift(1) >= slow_mom.shift(1))

    rsi = ind_tema_vidya_momentum_rsi_rsi(df, rsi_period)
    # Exit: RSI recovers from oversold
    rsi_recovery = (rsi > rsi_level) & (rsi.shift(1) <= rsi_level)

    df['FastMom'] = fast_mom
    df['SlowMom'] = slow_mom
    df['RSI'] = rsi
    df['MomCrossDown'] = mom_cross_down
    df['RSIRecovery'] = rsi_recovery

    # RCL-like: TEMA-VIDYA signals
    tema = ind_tema_vidya_momentum_rsi_tema(df, tema_p)
    vidya = ind_tema_vidya_momentum_rsi_vidya(df, vidya_p, vidya_alpha)
    # Entry: TEMA is rising (over tema_shift bars)
    tema_rising = tema > tema.shift(tema_shift)
    # Exit: open crosses below VIDYA from above
    open_below_vidya = (df['Open'] < vidya) & (df['Open'].shift(1) > vidya.shift(1))

    df['TEMA'] = tema
    df['VIDYA'] = vidya
    df['TEMARising'] = tema_rising
    df['OpenBelowVIDYA'] = open_below_vidya

    if year is not None:
        df = df[df.index.year == int(year)]

    # Combined entries: either momentum cross-down OR tema rising
    entries = df['MomCrossDown'] | df['TEMARising']

    # Combined exits: either RSI recovery OR open below VIDYA
    exits = df['RSIRecovery'] | df['OpenBelowVIDYA']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Let a Python Tool Scan Thousands of Strategies for AXP. Here’s the One That Survived.
# URL:   https://medium.com/@Kryptera/i-let-a-python-tool-scan-thousands-of-strategies-for-axp-heres-the-one-that-survived-40dee8c3bec2
# Data:  2026-06-22 02:00
# ─────────────────────────────────────

############################
# Strategy axp_std_momentum
############################

def ind_axp_std_momentum_std_falling(df: pd.DataFrame, std_period: int = 14, std_shift: int = 5) -> pd.Series:
    std = df['Close'].rolling(std_period, min_periods=1).std()
    return std < std.shift(std_shift)


def ind_axp_std_momentum_mom_cross_below(df: pd.DataFrame, mom_period: int = 10, level: float = 100.0) -> pd.Series:
    close = df['Close']
    shifted = close.shift(mom_period)
    safe_den = np.where(shifted != 0, shifted, 1.0)
    momentum = np.where(shifted != 0, close / safe_den * 100.0, 100.0)
    mom = pd.Series(momentum, index=df.index)
    return (mom < level) & (mom.shift(1) >= level)


strategy_axp_std_momentum_param_ranges = {
    'std_period_range'  : range(10, 25, 5),
    'std_shift_range'   : range(3, 9, 3),
    'mom_period_range'  : range(5, 20, 5),
}


def strategy_axp_std_momentum(data: pd.DataFrame, params: dict, year: int | None = None):
    std_period = params.get('std_period_range')
    std_shift  = params.get('std_shift_range')
    mom_period = params.get('mom_period_range')

    df = data.copy()

    std_falling = ind_axp_std_momentum_std_falling(df, std_period=std_period, std_shift=std_shift)
    mom_cross   = ind_axp_std_momentum_mom_cross_below(df, mom_period=mom_period, level=100.0)

    df['STD_Falling']              = std_falling
    df['Momentum_Cross_Below_Level'] = mom_cross

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['STD_Falling']
    exits   = df['Momentum_Cross_Below_Level']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: How to Build a Full Backtest Framework — And Why Good Strategy Might Still Be Useless
# URL:   https://medium.com/@Kryptera/how-to-build-a-full-backtest-framework-and-why-good-strategy-might-still-be-useless-e097291f6c9d
# Data:  2026-06-24 02:00
# ─────────────────────────────────────

############################
# Strategy bb_mcginley
############################

import pandas as pd
import numpy as np


def ind_bb_mcginley_bandwidth_above_avg(df: pd.DataFrame,
                                        bb_period: int = 20,
                                        bb_std: float = 2.0,
                                        bb_lookback: int = 20) -> pd.Series:
    close = df['Close']
    ma = close.rolling(bb_period, min_periods=1).mean()
    std = close.rolling(bb_period, min_periods=1).std(ddof=0)
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    safe_ma = np.where(ma != 0, ma, 1.0)
    bw = np.where(ma != 0, (upper - lower) / safe_ma, 0.0)
    bw_series = pd.Series(bw, index=close.index)
    bw_avg = bw_series.rolling(bb_lookback, min_periods=1).mean()
    return bw_series > bw_avg


def ind_bb_mcginley_dynamic(df: pd.DataFrame,
                            mcginley_period: int = 14) -> pd.Series:
    close = df['Close']
    arr = close.values
    n = len(arr)
    md = np.empty(n)
    md[0] = arr[0]
    k = mcginley_period
    for i in range(1, n):
        prev = md[i - 1]
        price = arr[i]
        denom = k * (price / prev) ** 4 if prev != 0 else k
        if denom == 0:
            denom = 1.0
        md[i] = prev + (price - prev) / denom
    return pd.Series(md, index=close.index)


def ind_bb_mcginley_exit_signal(df: pd.DataFrame,
                                mcginley: pd.Series) -> pd.Series:
    open_price = df['Open']
    # Open crosses below McGinley after being above it
    above_prev = open_price.shift(1) >= mcginley.shift(1)
    below_now = open_price < mcginley
    return above_prev & below_now


strategy_bb_mcginley_param_ranges = {
    'bb_period_range'    : range(16, 25, 4),
    'bb_std_range'       : range(1, 4, 1),
    'bb_lookback_range'  : range(16, 25, 4),
    'mcginley_range'     : range(11, 18, 3),
}


def strategy_bb_mcginley(data: pd.DataFrame, params: dict, year: int | None = None):
    bb_period    = params.get('bb_period_range')
    bb_std       = float(params.get('bb_std_range'))
    bb_lookback  = params.get('bb_lookback_range')
    mcginley_p   = params.get('mcginley_range')

    df = data.copy()

    bw_above_avg = ind_bb_mcginley_bandwidth_above_avg(
        df, bb_period=bb_period, bb_std=bb_std, bb_lookback=bb_lookback
    )
    mcginley = ind_bb_mcginley_dynamic(df, mcginley_period=mcginley_p)
    exit_sig = ind_bb_mcginley_exit_signal(df, mcginley=mcginley)

    df['BB_Width_Above_Avg'] = bw_above_avg
    df['McGinley'] = mcginley
    df['Exit_Signal'] = exit_sig

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['BB_Width_Above_Avg']
    exits   = df['Exit_Signal']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Left My Laptop Running for 3 Nights. It Found 51 Trading Strategies While I Slept.
# URL:   https://medium.com/@Kryptera/i-left-my-laptop-running-for-3-nights-it-found-51-trading-strategies-while-i-slept-c1959667dd95
# Data:  2026-06-26 02:00
# ─────────────────────────────────────

############################
# Strategy bull_cts_nvda
############################

import pandas as pd
import numpy as np


def ind_bull_cts_nvda_bull_power(df: pd.DataFrame, ema_period: int = 13) -> pd.Series:
    close = df['Close']
    ema_arr = np.empty(len(close))
    close_arr = close.values
    alpha = 2.0 / (ema_period + 1)
    ema_arr[0] = close_arr[0]
    for i in range(1, len(close_arr)):
        ema_arr[i] = alpha * close_arr[i] + (1 - alpha) * ema_arr[i - 1]
    ema = pd.Series(ema_arr, index=close.index)
    bull_power = df['High'] - ema
    return bull_power


def ind_bull_cts_nvda_cts(df: pd.DataFrame, period: int = 20) -> pd.Series:
    close = df['Close'].values
    n = len(close)
    cts = np.full(n, np.nan)
    for i in range(period, n):
        diff = np.diff(close[i - period: i + 1])
        up = diff[diff > 0].sum()
        down = -diff[diff < 0].sum()
        total = up + down
        if total != 0:
            cts[i] = 100.0 * (up - down) / total
        else:
            cts[i] = 0.0
    return pd.Series(cts, index=df.index)


strategy_bull_cts_nvda_param_ranges = {
    'ema_period_range'    : range(10, 21, 5),
    'bull_level_range'    : range(0, 1, 1),
    'cts_period_range'    : range(15, 26, 5),
    'cts_lower_range'     : range(-60, -39, 10),
    'bull_shift_range'    : range(3, 8, 2),
    'cts_shift_range'     : range(3, 8, 2),
}


def strategy_bull_cts_nvda(data: pd.DataFrame, params: dict, year: int | None = None):
    ema_period  = params.get('ema_period_range')
    bull_level  = float(params.get('bull_level_range'))
    cts_period  = params.get('cts_period_range')
    cts_lower   = float(params.get('cts_lower_range'))
    bull_shift  = params.get('bull_shift_range')
    cts_shift   = params.get('cts_shift_range')

    df = data.copy()

    bull_power = ind_bull_cts_nvda_bull_power(df, ema_period=ema_period)
    cts        = ind_bull_cts_nvda_cts(df, period=cts_period)

    df['Bull_Power'] = bull_power
    df['CTS']        = cts

    # Entry: Bull Power crosses below level (was >= level, now < level)
    bp_prev = df['Bull_Power'].shift(bull_shift)
    entries = (bp_prev >= bull_level) & (df['Bull_Power'] < bull_level)

    # Exit: CTS crosses above lower level (was <= lower, now > lower, after shift)
    cts_prev = df['CTS'].shift(cts_shift)
    exits = (cts_prev <= cts_lower) & (df['CTS'] > cts_lower)

    if year is not None:
        df       = df[df.index.year == int(year)]
        entries  = entries[entries.index.year == int(year)]
        exits    = exits[exits.index.year == int(year)]

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Found a RMA ATR Bands Indicator on TradingView — And It Outperformed Buy and Hold on a Single…
# URL:   https://medium.com/@Kryptera/i-found-a-rma-atr-bands-indicator-on-tradingview-and-it-outperformed-buy-and-hold-on-a-single-b735a434b3ec
# Data:  2026-06-27 02:00
# ─────────────────────────────────────

############################
# Strategy rma_atr_bands
############################

import pandas as pd
import numpy as np


def ind_rma_atr_bands_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA (Running Moving Average) smoothing."""
    alpha = 1.0 / period
    arr = series.values.astype(float)
    out = np.empty(len(arr))
    out[:] = np.nan
    # Find first valid index
    first_valid = 0
    while first_valid < len(arr) and np.isnan(arr[first_valid]):
        first_valid += 1
    if first_valid >= len(arr):
        return pd.Series(out, index=series.index)
    out[first_valid] = arr[first_valid]
    for i in range(first_valid + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def ind_rma_atr_bands_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR using RMA smoothing."""
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)
    hl = high - low
    hpc = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    lpc = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(np.maximum(hl, hpc), lpc)
    tr_series = pd.Series(tr, index=df.index)
    return ind_rma_atr_bands_rma(tr_series, period)


def ind_rma_atr_bands_bands(df: pd.DataFrame,
                             rma_period: int = 14,
                             atr_period: int = 14,
                             upper_mult: float = 0.4,
                             lower_mult: float = 1.6):
    """
    Compute RMA ATR Bands.
    - RMA applied to High price
    - Upper band: rma_high + upper_mult * ATR
    - Lower band: rma_high - lower_mult * ATR
    Returns: rma_high, upper_band, lower_band, atr
    """
    rma_high = ind_rma_atr_bands_rma(df['High'], rma_period)
    atr = ind_rma_atr_bands_atr(df, atr_period)
    upper_band = rma_high + upper_mult * atr
    lower_band = rma_high - lower_mult * atr
    return rma_high, upper_band, lower_band, atr


def ind_rma_atr_bands_signals(df: pd.DataFrame,
                               upper_band: pd.Series,
                               lower_band: pd.Series):
    """
    Stateful trend logic:
    - trend flips to +1 (long entry) when close crosses above upper_band
    - trend flips to -1 (exit) when close drops below lower_band
    Returns: long_signals, exit_signals as boolean Series
    """
    close = df['Close'].values.astype(float)
    upper = upper_band.values.astype(float)
    lower = lower_band.values.astype(float)
    n = len(close)
    trend = np.zeros(n, dtype=np.int8)
    long_sig = np.zeros(n, dtype=bool)
    exit_sig = np.zeros(n, dtype=bool)

    current_trend = 0
    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            trend[i] = current_trend
            continue
        if close[i] > upper[i]:
            if current_trend != 1:
                long_sig[i] = True
            current_trend = 1
        elif close[i] < lower[i]:
            if current_trend != -1:
                exit_sig[i] = True
            current_trend = -1
        trend[i] = current_trend

    return (pd.Series(long_sig, index=df.index),
            pd.Series(exit_sig, index=df.index))


strategy_rma_atr_bands_param_ranges = {
    'rma_period_range': range(10, 25, 7),   # [10, 17, 24] -> 3 values
    'atr_period_range': range(10, 25, 7),   # [10, 17, 24] -> 3 values
    'upper_mult_range': range(3, 7, 2),     # [3, 5] -> 2 values (divide by 10)
    'lower_mult_range': range(12, 22, 5),   # [12, 17] -> 2 values (divide by 10)
}
# Total: 3 * 3 * 2 * 2 = 36 combinations — well within limit


def strategy_rma_atr_bands(data: pd.DataFrame, params: dict, year: int | None = None):
    rma_period  = params.get('rma_period_range')
    atr_period  = params.get('atr_period_range')
    upper_mult  = params.get('upper_mult_range') / 10.0
    lower_mult  = params.get('lower_mult_range') / 10.0

    df = data.copy()

    rma_high, upper_band, lower_band, atr = ind_rma_atr_bands_bands(
        df,
        rma_period=rma_period,
        atr_period=atr_period,
        upper_mult=upper_mult,
        lower_mult=lower_mult
    )

    df['RMA_High']   = rma_high
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    df['ATR']        = atr

    long_sig, exit_sig = ind_rma_atr_bands_signals(df, df['Upper_Band'], df['Lower_Band'])
    df['LongSig'] = long_sig
    df['ExitSig'] = exit_sig

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['LongSig']
    exits   = df['ExitSig']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Left My Desktop App Running Overnight.
# URL:   https://medium.com/@Kryptera/i-left-my-desktop-app-running-overnight-313832b440b1
# Data:  2026-06-28 02:00
# ─────────────────────────────────────

############################
# Strategy ao_er_amt
############################

def ind_ao_er_amt_ao(df: pd.DataFrame, short_period: int = 5, long_period: int = 34) -> pd.Series:
    median_price = (df['High'] + df['Low']) / 2.0
    sma_short = median_price.rolling(window=short_period, min_periods=1).mean()
    sma_long  = median_price.rolling(window=long_period,  min_periods=1).mean()
    ao = sma_short - sma_long
    return ao


def ind_ao_er_amt_ao_falling_consecutive(ao: pd.Series, bars: int = 3) -> pd.Series:
    cond = (ao < ao.shift(1)).astype(float)
    result = cond.rolling(bars, min_periods=bars).sum() == bars
    return result.fillna(False)


def ind_ao_er_amt_er(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df['Close']
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    safe_vol = np.where(volatility.values != 0, volatility.values, 1.0)
    er_vals = np.where(volatility.values != 0, change.values / safe_vol, 0.0)
    er = pd.Series(er_vals, index=close.index)
    return er


strategy_ao_er_amt_param_ranges = {
    'ao_short_range'  : range(3, 8,  2),   # 3 values: [3, 5, 7]
    'ao_long_range'   : range(24, 40, 8),  # 2 values: [24, 32] ~approx 34
    'ao_bars_range'   : range(2, 5,  1),   # 3 values: [2, 3, 4]
    'er_period_range' : range(10, 20, 4),  # 3 values: [10, 14, 18]
    'er_level_range'  : range(6, 9,  1),   # 3 values: [6, 7, 8]  → divide by 10
}


def strategy_ao_er_amt(data: pd.DataFrame, params: dict, year: int | None = None):
    ao_short = params.get('ao_short_range')
    ao_long  = params.get('ao_long_range')
    ao_bars  = params.get('ao_bars_range')
    er_per   = params.get('er_period_range')
    er_level = params.get('er_level_range') / 10.0

    df = data.copy()

    ao = ind_ao_er_amt_ao(df, short_period=ao_short, long_period=ao_long)
    ao_falling = ind_ao_er_amt_ao_falling_consecutive(ao, bars=ao_bars)
    er = ind_ao_er_amt_er(df, period=er_per)

    df['AO']         = ao
    df['AO_Falling'] = ao_falling
    df['ER']         = er

    if year is not None:
        df = df[df.index.year == int(year)]

    entries = df['AO_Falling']

    er_trend_up = (df['ER'] > er_level) & (df['Close'] > df['Close'].shift(1))
    exits = er_trend_up

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: I Built a Hidden Markov Model to Detect Gold Market Regimes — Here’s What the Data Revealed
# URL:   https://medium.com/@Kryptera/i-built-a-hidden-markov-model-to-detect-gold-market-regimes-heres-what-the-data-revealed-46c84f2ca83d
# Data:  2026-06-29 02:00
# ─────────────────────────────────────

############################
# Strategy hmm_regime_gold
############################

import numpy as np
import pandas as pd


def ind_hmm_regime_gold_features(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
    close = df['Close']
    high = df['High']
    low = df['Low']

    log_ret = np.log(close / close.shift(1))
    returns = close.pct_change()
    vol = returns.rolling(vol_window, min_periods=1).std()
    safe_close = np.where(close.values != 0, close.values, 1.0)
    hl_range = (high.values - low.values) / safe_close

    out = df.copy()
    out['log_ret'] = log_ret
    out['returns'] = returns
    out['vol_rolling'] = vol
    out['hl_range'] = pd.Series(hl_range, index=df.index)
    return out


def ind_hmm_regime_gold_regime(df_feat: pd.DataFrame,
                                train_window: int = 60,
                                smooth_window: int = 5) -> pd.Series:
    """
    Approximate HMM regime detection using pandas/numpy only.
    We use a rolling k-means style clustering on 3 features:
      log_ret, vol_rolling, hl_range
    States are sorted by mean log_ret: 0=Bearish, 1=Ranging, 2=Bullish
    Then apply majority-vote smoothing.
    """
    log_ret = df_feat['log_ret'].values
    vol = df_feat['vol_rolling'].values
    hl = df_feat['hl_range'].values
    n = len(df_feat)

    raw_states = np.full(n, np.nan)

    for i in range(train_window, n):
        window_log = log_ret[i - train_window:i]
        window_vol = vol[i - train_window:i]
        window_hl = hl[i - train_window:i]

        # Normalize features
        def safe_norm(arr):
            mu = np.nanmean(arr)
            sd = np.nanstd(arr)
            if sd == 0:
                return arr - mu
            return (arr - mu) / sd

        f0 = safe_norm(window_log)
        f1 = safe_norm(window_vol)
        f2 = safe_norm(window_hl)

        # Simple k-means with 3 clusters initialized by percentiles
        # Initialize centroids by percentile splits on log_ret
        p33 = np.nanpercentile(f0, 33)
        p67 = np.nanpercentile(f0, 67)

        c0 = np.array([np.nanmean(f0[f0 <= p33]),
                       np.nanmean(f1[f0 <= p33]),
                       np.nanmean(f2[f0 <= p33])])
        mask_mid = (f0 > p33) & (f0 <= p67)
        c1 = np.array([np.nanmean(f0[mask_mid]) if mask_mid.sum() > 0 else 0.0,
                       np.nanmean(f1[mask_mid]) if mask_mid.sum() > 0 else 0.0,
                       np.nanmean(f2[mask_mid]) if mask_mid.sum() > 0 else 0.0])
        c2 = np.array([np.nanmean(f0[f0 > p67]),
                       np.nanmean(f1[f0 > p67]),
                       np.nanmean(f2[f0 > p67])])

        centroids = np.array([c0, c1, c2])

        # Replace NaN centroids
        for ci in range(3):
            for fi in range(3):
                if np.isnan(centroids[ci, fi]):
                    centroids[ci, fi] = 0.0

        # K-means iterations
        features_mat = np.column_stack([f0, f1, f2])
        valid_mask = ~np.any(np.isnan(features_mat), axis=1)
        features_valid = features_mat[valid_mask]

        for _ in range(10):
            # Assign
            dists = np.array([
                np.sum((features_valid - centroids[k]) ** 2, axis=1)
                for k in range(3)
            ])
            labels = np.argmin(dists, axis=0)
            # Update
            new_centroids = np.zeros((3, 3))
            for k in range(3):
                mk = labels == k
                if mk.sum() > 0:
                    new_centroids[k] = features_valid[mk].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Sort states by mean log_ret of centroid (0=bearish, 1=ranging, 2=bullish)
        centroid_ret = centroids[:, 0]
        sort_order = np.argsort(centroid_ret)  # ascending
        rank_map = np.empty(3, dtype=int)
        for rank, orig in enumerate(sort_order):
            rank_map[orig] = rank

        # Current point
        cur_log = log_ret[i]
        cur_vol = vol[i]
        cur_hl = hl[i]

        if np.isnan(cur_log) or np.isnan(cur_vol) or np.isnan(cur_hl):
            raw_states[i] = np.nan
            continue

        # Normalize current point using train window stats
        def norm_val(val, arr):
            mu = np.nanmean(arr)
            sd = np.nanstd(arr)
            if sd == 0:
                return 0.0
            return (val - mu) / sd

        cur_f0 = norm_val(cur_log, window_log)
        cur_f1 = norm_val(cur_vol, window_vol)
        cur_f2 = norm_val(cur_hl, window_hl)
        cur_feat = np.array([cur_f0, cur_f1, cur_f2])

        dists_cur = np.array([np.sum((cur_feat - centroids[k]) ** 2) for k in range(3)])
        assigned = np.argmin(dists_cur)
        raw_states[i] = rank_map[assigned]

    # Build series
    regime_raw = pd.Series(raw_states, index=df_feat.index)

    # Majority vote smoothing
    def majority_vote(x):
        counts = np.bincount(x[~np.isnan(x)].astype(int), minlength=3)
        return float(np.argmax(counts))

    regime_smooth = regime_raw.rolling(smooth_window, min_periods=1).apply(
        lambda x: majority_vote(x), raw=True
    )

    return regime_smooth


strategy_hmm_regime_gold_param_ranges = {
    'train_window_range': range(40, 81, 20),
    'vol_window_range': range(5, 16, 5),
    'smooth_window_range': range(3, 10, 3),
}


def strategy_hmm_regime_gold(data: pd.DataFrame, params: dict, year: int | None = None):
    train_window = params.get('train_window_range')
    vol_window = params.get('vol_window_range')
    smooth_window = params.get('smooth_window_range')

    df = data.copy()

    df_feat = ind_hmm_regime_gold_features(df, vol_window=vol_window)

    regime = ind_hmm_regime_gold_regime(
        df_feat,
        train_window=train_window,
        smooth_window=smooth_window
    )

    df['regime'] = regime
    df['log_ret'] = df_feat['log_ret']
    df['vol_rolling'] = df_feat['vol_rolling']

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: regime switches to Bullish (2)
    # Exit: regime switches to Bearish (0) or Ranging (1)
    regime_s = df['regime']
    prev_regime = regime_s.shift(1)

    entries = (regime_s == 2.0) & (prev_regime != 2.0)
    exits = (regime_s != 2.0) & (prev_regime == 2.0)

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: The Regime Report: How to Find the Strategy Inside Your Strategy
# URL:   https://medium.com/@Kryptera/the-regime-report-how-to-find-the-strategy-inside-your-strategy-1fa10a6ae4d1
# Data:  2026-07-08 02:00
# ─────────────────────────────────────

############################
# Strategy bears_power_cts_regime
############################

import pandas as pd
import numpy as np


def ind_bears_power_cts_regime_bears_power(df: pd.DataFrame, bp_period: int = 13) -> pd.Series:
    close = df['Close']
    ema = close.ewm(span=bp_period, adjust=False, min_periods=1).mean()
    bears_power = df['Low'] - ema
    return bears_power


def ind_bears_power_cts_regime_cts(df: pd.DataFrame, cts_period: int = 20, cts_mult: float = 1.5) -> tuple:
    close = df['Close']
    ma = close.rolling(cts_period, min_periods=1).mean()
    std = close.rolling(cts_period, min_periods=1).std(ddof=0)
    lower = ma - cts_mult * std
    return ma, lower


def ind_bears_power_cts_regime_regime(df: pd.DataFrame, vol_lookback: int = 20, trend_lookback: int = 200) -> pd.Series:
    close = df['Close']
    sma200 = close.rolling(trend_lookback, min_periods=1).mean()
    is_trending = close > sma200

    ret = close.pct_change()
    realized_vol = ret.rolling(vol_lookback, min_periods=1).std()
    vol_median = realized_vol.expanding(min_periods=60).median()

    safe_med = np.where(vol_median.notna() & (vol_median != 0), vol_median, np.nan)
    is_high_vol = realized_vol > pd.Series(safe_med, index=realized_vol.index)

    conditions = [
        (is_trending & is_high_vol),
        (is_trending & ~is_high_vol),
        (~is_trending & is_high_vol),
        (~is_trending & ~is_high_vol),
    ]
    regime = np.select(conditions,
                       ['trend_highvol', 'trend_lowvol', 'chop_highvol', 'chop_lowvol'],
                       default='unknown')
    return pd.Series(regime, index=df.index)


strategy_bears_power_cts_regime_param_ranges = {
    'bp_period_range'     : range(10, 21, 5),
    'cts_period_range'    : range(15, 31, 5),
    'cts_mult_range'      : range(10, 25, 5),
    'vol_lookback_range'  : range(15, 31, 5),
    'trend_lookback_range': range(150, 251, 50),
}


def strategy_bears_power_cts_regime(data: pd.DataFrame, params: dict, year: int | None = None):
    bp_period      = params.get('bp_period_range')
    cts_period     = params.get('cts_period_range')
    cts_mult       = params.get('cts_mult_range') / 10.0
    vol_lookback   = params.get('vol_lookback_range')
    trend_lookback = params.get('trend_lookback_range')

    df = data.copy()

    bears_power = ind_bears_power_cts_regime_bears_power(df, bp_period=bp_period)
    cts_ma, cts_lower = ind_bears_power_cts_regime_cts(df, cts_period=cts_period, cts_mult=cts_mult)
    regime = ind_bears_power_cts_regime_regime(df, vol_lookback=vol_lookback, trend_lookback=trend_lookback)

    df['BearsPower']  = bears_power
    df['CTS_MA']      = cts_ma
    df['CTS_Lower']   = cts_lower
    df['regime']      = regime

    # Bears Power is falling: current value < previous value
    bp_falling = df['BearsPower'] < df['BearsPower'].shift(1)

    # CTS cross above lower: close crosses above lower band
    cts_cross_above_lower = (df['Close'] > df['CTS_Lower']) & (df['Close'].shift(1) <= df['CTS_Lower'].shift(1))

    # Regime gate: allow entries only in trend_lowvol or chop_highvol
    ALLOWED_REGIMES = {'trend_lowvol', 'chop_highvol'}
    regime_gate = df['regime'].shift(1).isin(ALLOWED_REGIMES)

    entries = bp_falling & regime_gate
    exits   = cts_cross_above_lower

    if year is not None:
        df_year = df[df.index.year == int(year)]
        entries = entries[df_year.index]
        exits   = exits[df_year.index]

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: Does Volume Actually Time the Crowd? I Tested It on the One Stock Where the Crowd Is Loudest.
# URL:   https://medium.com/@Kryptera/does-volume-actually-time-the-crowd-i-tested-it-on-the-one-stock-where-the-crowd-is-loudest-91135a89aa0b
# Data:  2026-07-09 02:00
# ─────────────────────────────────────

############################
# Strategy crowd_timer
############################

import numpy as np
import pandas as pd


def ind_crowd_timer_signals(
    df: pd.DataFrame,
    run_window: int = 20,
    vol_baseline: int = 60,
    z_threshold_scaled: int = 20,
    quantile_window: int = 252,
    run_pctl_scaled: int = 90,
) -> tuple:
    close = df['Close']
    volume = df['Volume']

    z_threshold = z_threshold_scaled / 10.0
    run_pctl = run_pctl_scaled / 100.0

    # N-day cumulative price run
    price_run = close.pct_change(run_window)

    # Volume z-score
    vol_mean = volume.rolling(vol_baseline, min_periods=1).mean()
    vol_std = volume.rolling(vol_baseline, min_periods=1).std(ddof=1)
    vol_std_arr = vol_std.values
    vol_mean_arr = vol_mean.values
    volume_arr = volume.values
    safe_std = np.where(vol_std_arr != 0, vol_std_arr, 1.0)
    volz_arr = np.where(vol_std_arr != 0, (volume_arr - vol_mean_arr) / safe_std, 0.0)
    vol_z = pd.Series(volz_arr, index=df.index)

    # Rolling quantile thresholds (no look-ahead)
    up_run_th = price_run.rolling(quantile_window, min_periods=max(1, quantile_window // 4)).quantile(run_pctl)
    down_run_th = price_run.rolling(quantile_window, min_periods=max(1, quantile_window // 4)).quantile(1.0 - run_pctl)

    # SignalTop: extended up run + volume spike -> expect pullback (contrarian: go short / exit long)
    signal_top = (price_run > up_run_th) & (vol_z > z_threshold)

    # SignalBottom: extended down run + volume spike -> expect bounce (contrarian: go long)
    signal_bottom = (price_run < down_run_th) & (vol_z > z_threshold)

    return signal_top, signal_bottom


strategy_crowd_timer_param_ranges = {
    'run_window_range': range(10, 31, 10),
    'vol_baseline_range': range(40, 81, 20),
    'z_threshold_scaled_range': range(15, 26, 5),
    'run_pctl_scaled_range': range(80, 96, 5),
}


def strategy_crowd_timer(data: pd.DataFrame, params: dict, year: int | None = None):
    run_window = params.get('run_window_range')
    vol_baseline = params.get('vol_baseline_range')
    z_threshold_scaled = params.get('z_threshold_scaled_range')
    run_pctl_scaled = params.get('run_pctl_scaled_range')

    df = data.copy()

    signal_top, signal_bottom = ind_crowd_timer_signals(
        df,
        run_window=run_window,
        vol_baseline=vol_baseline,
        z_threshold_scaled=z_threshold_scaled,
        quantile_window=252,
        run_pctl_scaled=run_pctl_scaled,
    )

    df['SignalTop'] = signal_top
    df['SignalBottom'] = signal_bottom

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: SignalBottom (contrarian long on capitulation)
    entries = df['SignalBottom']

    # Exit: SignalTop fires (crowd-top / extended up run with volume spike)
    exits = df['SignalTop']

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits


# ─────────────────────────────────────
# Fonte: A Massive Backtest on AMAT — and Why the Monte Carlo Simulation Made Me Pump the Brakes
# URL:   https://medium.com/@Kryptera/a-massive-backtest-on-amat-and-why-the-monte-carlo-simulation-made-me-pump-the-brakes-1d110e268ca7
# Data:  2026-07-14 02:00
# ─────────────────────────────────────

############################
# Strategy cci_ema_amat
############################

import pandas as pd
import numpy as np


def ind_cci_ema_amat_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    sma_tp = tp.rolling(period, min_periods=1).mean()
    mean_dev = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    safe_den = np.where(mean_dev != 0, mean_dev.values, 1.0)
    cci_vals = np.where(
        mean_dev.values != 0,
        (tp.values - sma_tp.values) / (0.015 * safe_den),
        0.0
    )
    return pd.Series(cci_vals, index=df.index)


def ind_cci_ema_amat_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    ema = df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()
    return ema


strategy_cci_ema_amat_param_ranges = {
    'cci_period_range' : range(11, 25, 4),
    'cci_level_range'  : range(-10, 11, 10),
    'ema_period_range' : range(16, 25, 4),
}


def strategy_cci_ema_amat(data: pd.DataFrame, params: dict, year: int | None = None):
    cci_period = params.get('cci_period_range')
    cci_level  = params.get('cci_level_range')
    ema_period = params.get('ema_period_range')

    df = data.copy()

    cci = ind_cci_ema_amat_cci(df, period=cci_period)
    ema = ind_cci_ema_amat_ema(df, period=ema_period)

    df['CCI'] = cci
    df['EMA'] = ema

    if year is not None:
        df = df[df.index.year == int(year)]

    # Entry: CCI drops below the threshold level
    entries = df['CCI'] < cci_level

    # Exit: Open crosses below EMA after having been above it the prior bar
    open_below_ema       = df['Open'] < df['EMA']
    open_above_ema_prev  = df['Open'].shift(1) > df['EMA'].shift(1)
    exits = open_below_ema & open_above_ema_prev

    shifted_entries = entries.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    shifted_exits   = exits.shift(1).astype(bool).fillna(False).infer_objects(copy=False)
    return shifted_entries, shifted_exits
