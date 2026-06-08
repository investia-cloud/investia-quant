"""
k_portfolios.py — Refactored from notebooks/libs/k_portfolios.ipynb
"""

# # Importo le definizioni statiche dei ticker
# %run k_tickers.ipynb

#
# Definizione dei portafogli di TS
#

# Definizione del portafoglio (con parametri globali inclusi)

portfolio_us_trading_2025 = {
    "Title": "US Trading",  # Titolo del portafoglio
    "params_global": {
        "ratio": '4:1',  # Default ratio 4:1
        "end_date": datetime.now(),
        "operating_freq": "1D",  # Frequenza operativa
        "init_cash": 100_000,  # Capitale iniziale
        "fees": 0.001,  # Commissioni trading
        "slippage": 0, # 0.002: Attenzione: utilizzato da APR 10, 2025
        "sl_stop": None, # 0.003: Attenzione: non utilizzato
        # "sl_stop": 0.003, # 0.003: Attenzione: non utilizzato
        "price_col" : "Close"
    },
    "trading_systems": [
        {
            # TS1
            "symbol": "ANET",
            "strategy": "ko_bb",
            "param_ranges": {
                # 2024: {'fast_period_range': 22, 'slow_period_range': 56, 'period_range': 19, 'std_dev_range': 2.5, 'squeeze_window_range': 14}	
                # 2025:	{'fast_period_range': 20, 'slow_period_range': 40, 'period_range': 19, 'std_dev_range': 2, 'squeeze_window_range': 12}
                'fast_period_range' : [20,22],  # Range for fast period of KO
                'slow_period_range' : [40,56],  # Range for slow period of KO
                'period_range' : [19],  # Range for Bollinger Bands period
                'std_dev_range' : [2, 2.5],  # Range for Bollinger Bands standard deviation
                'squeeze_window_range' : [12,14]  # Range for BB Squeeze window size
            },
        },
        {
            # TS2
            "symbol": "APO",
            "strategy": "macd_mf",
            # 2024	{'short_window_range': 14, 'long_window_range': 27, 'signal_window_range': 14}
            # 2025	{'short_window_range': 12, 'long_window_range': 20, 'signal_window_range': 14}	
            "param_ranges": {
                'short_window_range': [12,14],
                'long_window_range': [20,27],
                'signal_window_range': [14]
            }
        },
        {
            # TS3
            "symbol": "AVGO",
            "strategy": "bollinger",
            # 2024	{'period_range': 2, 'std_dev_range': 1.5, 'squeeze_period_range': 22}
            # 2025	{'period_range': 2, 'std_dev_range': 1.5, 'squeeze_period_range': 2}
            "param_ranges": {
                'period_range' : [2],  # Range for Bollinger Bands period
                'std_dev_range' : [1.5],  # Range for standard deviation (multiplier)
                'squeeze_period_range' : [2,22]  # Range for rolling window in squeeze condition
            }
        },
        {
            # TS4
            "symbol": "AXON",
            "strategy": "ichimoku",
            # 2024	{'tenkan_period_range': 27, 'kijun_period_range': 18, 'senkou_span_b_period_range': 41}
            # 2025	{'tenkan_period_range': 30, 'kijun_period_range': 24, 'senkou_span_b_period_range': 40}	
            "param_ranges": {
                # Define dynamic ranges for Ichimoku parameters
                'tenkan_period_range' : [27,30],  # Tenkan-Sen period range
                'kijun_period_range' : [18,24],  # Kijun-Sen period range
                'senkou_span_b_period_range' : [40,41]  # Senkou Span B period range
            }
        },
        # TS5
        {
            "symbol": "CEG",
            "strategy": "holding"
        },
        {
            # TS6
            "symbol": "DECK",
            "strategy": "hma_atr",
            # 2024	{'hma_short_period_range': 5, 'hma_long_period_range': 21, 'atr_period_range': 26, 'atr_multiplier_range': 2.5, 'entry_shift_range': 5, 'exit_shift_range': 1}	
            # 2025	{'hma_short_period_range': 5, 'hma_long_period_range': 21, 'atr_period_range': 18, 'atr_multiplier_range': 2.5, 'entry_shift_range': 5, 'exit_shift_range': 1}
            "param_ranges": {
                # Define dynamic ranges for HMA and ATR parameters
                'hma_short_period_range' : [5],  # Range for short-term HMA periods
                'hma_long_period_range' : [21],  # Range for long-term HMA periods
                'atr_period_range' : [18,26],  # Range for ATR periods
                'atr_multiplier_range' : [2.5],  # ATR multiplier range
                'entry_shift_range' : [5],  # Range for entry shift
                'exit_shift_range' : [1]  # Range for exit shift
            }
        },
        {
            # TS7
            "symbol": "FICO",
            "strategy": "fractal_vortex",
            # 2024	{'vortex_period_range': 15, 'fractal_period_range': 10}
            # 2025	{'vortex_period_range': 7, 'fractal_period_range': 28}
            "param_ranges": {
                # Define dynamic ranges for HMA and ATR parameters
                'vortex_period_range' : [7,15],
                'fractal_period_range' : [10,28]
            }
        },
        {
            # TS8
            "symbol": "FOX",
            "strategy": "macd_zero_line_rejection",
            # 2024	{'short_window_range': 8, 'long_window_range': 20, 'signal_window_range': 11, 'shift_1_range': 7, 'shift_2_range': 4, 'shift_3_range': 1, 'volume_window_range': 4}
            # 2025	{'short_window_range': 8, 'long_window_range': 20, 'signal_window_range': 5, 'shift_1_range': 7, 'shift_2_range': 1, 'shift_3_range': 10, 'volume_window_range': 4}	
            "param_ranges": {
                # Define dynamic ranges for MACD periods, shift values, and volume rolling window
                'short_window_range' :[8],
                'long_window_range' : [20],
                'signal_window_range' :[5,11],
                'shift_1_range' : [7],
                'shift_2_range' : [1,4],
                'shift_3_range' : [1,10],
                'volume_window_range' : [4]  # Volume rolling window to optimize
            }
        },
        # TS9 FOXA non VERIFICATO!
        {
            # TS10
            "symbol": "GDDY",
            "strategy": "macd_daily_weekly",
            # 2024	{'short_window_range': 6, 'long_window_range': 20, 'signal_window_range': 7}
            # 2025	{'short_window_range': 5, 'long_window_range': 21, 'signal_window_range': 8}	
            "param_ranges": {
                # Define dynamic ranges for MACD parameters
                'short_window_range' : [5,6],  # Range for short window
                'long_window_range' : [20,21],  # Range for long window
                'signal_window_range' : [7,8]  # Range for signal window
            }
        },
        {
            # TS11
            "symbol": "HWM",
            "strategy": "ko_bb",
            # 2024	{'fast_period_range': 20, 'slow_period_range': 40, 'period_range': 15, 'std_dev_range': 2.5, 'squeeze_window_range': 24}	
            # 2025	{'fast_period_range': 20, 'slow_period_range': 40, 'period_range': 15, 'std_dev_range': 2.5, 'squeeze_window_range': 10}	
            "param_ranges": {
                'fast_period_range' : [20],  # Range for fast period of KO
                'slow_period_range' : [40],  # Range for slow period of KO
                'period_range' : [15],  # Range for Bollinger Bands period
                'std_dev_range' : [2.5],  # Range for Bollinger Bands standard deviation
                'squeeze_window_range' : [10,24]  # Range for BB Squeeze window size
            }
        },
        {
            # TS12
            "symbol": "KKR",
            "strategy": "tema_heikin_ashi",
            # 2024	{'entry_shift_1_range': 5, 'entry_shift_2_range': 1, 'exit_shift_1_range': 7, 'exit_shift_2_range': 7, 'tema_period_range': 18}	
            # 2025	{'entry_shift_1_range': 5, 'entry_shift_2_range': 1, 'exit_shift_1_range': 7, 'exit_shift_2_range': 7, 'tema_period_range': 18}	
            "param_ranges": {
                # Define dynamic ranges for MACD e TEMA parameters
                'entry_shift_1_range' : [5],
                'entry_shift_2_range' : [1],
                'exit_shift_1_range' : [7],
                'exit_shift_2_range' : [7],
                'tema_period_range' : [18]
            }
        },
        {
            # TS13
            "symbol": "META",
            "strategy": "ichimoku",
            # 2024	{'tenkan_period_range': 15, 'kijun_period_range': 24, 'senkou_span_b_period_range': 48}	
            # 2025	{'tenkan_period_range': 18, 'kijun_period_range': 24, 'senkou_span_b_period_range': 55}	
            "param_ranges": {
                'tenkan_period_range' : [15,18],  # Range for Tenkan-sen periods
                'kijun_period_range' : [24],  # Range for Kijun-sen periods
                'senkou_span_b_period_range' : [48,55]  # Range for Senkou Span B periods
            }
        },
        {
            # TS14
            "symbol": "NFLX",
            "strategy": "sma_mf",
            # 2024	{'daily_short_window_range': 1, 'daily_long_window_range': 146, 'weekly_short_window_range': 7, 'weekly_long_window_range': 30}
            # 2025	{'daily_short_window_range': 1, 'daily_long_window_range': 136, 'weekly_short_window_range': 7, 'weekly_long_window_range': 30}	
            "param_ranges": {
                'daily_short_window_range' : [1],  # Range for short window (daily)
                'daily_long_window_range' : [136,146],  # Range for long window (daily)
                'weekly_short_window_range' : [7],  # Range for short window (weekly)
                'weekly_long_window_range' : [30]  # Range for long window (weekly)    
            }
        },
        {
            # TS15
            "symbol": "NRG",
            "strategy": "ko_bb",
            # 2024	{'fast_period_range': 22, 'slow_period_range': 52, 'period_range': 19, 'std_dev_range': 2, 'squeeze_window_range': 20}
            # 2025	{'fast_period_range': 20, 'slow_period_range': 40, 'period_range': 25, 'std_dev_range': 2.5, 'squeeze_window_range': 30}	
            "param_ranges": {
                'fast_period_range' : [22,20],  # Range for fast period of KO
                'slow_period_range' : [52,40],  # Range for slow period of KO
                'period_range' : [19,25],  # Range for Bollinger Bands period
                'std_dev_range' : [2, 2.5],  # Range for Bollinger Bands standard deviation
                'squeeze_window_range' : [20,30]  # Range for BB Squeeze window size
            }
        },
        {
            # TS16
            "symbol": "NVDA",
            "strategy": "gmma",
            "ratio" : '2:1',
            # 2024	{'short_period_start_range': 5, 'short_period_end_range': 12, 'short_period_step_range': 1, 'long_period_start_range': 35, 'long_period_end_range': 68, 'long_period_step_range': 1}
            # 2025	{'short_period_start_range': 9, 'short_period_end_range': 16, 'short_period_step_range': 2, 'long_period_start_range': 39, 'long_period_end_range': 50, 'long_period_step_range': 1}	
            "param_ranges": {
                # Define dynamic ranges for GMMA periods
                'short_period_start_range' : [5,9],
                'short_period_end_range' : [12,16],
                'short_period_step_range' : [1, 2],
                'long_period_start_range' : [35,39],
                'long_period_end_range' : [68,50],
                'long_period_step_range' : [1]
            }
        },
        {
            # TS17
            "symbol": "PLTR",
            "strategy": "gmma",
            "ratio" : '1:1',
            # 2024	{'short_period_start_range': 1, 'short_period_end_range': 10, 'short_period_step_range': 3, 'long_period_start_range': 37, 'long_period_end_range': 58, 'long_period_step_range': 2}
            # 2025	{'short_period_start_range': 1, 'short_period_end_range': 20, 'short_period_step_range': 1, 'long_period_start_range': 31, 'long_period_end_range': 54, 'long_period_step_range': 1}	
            "param_ranges": {
                # Define dynamic ranges for GMMA periods
                'short_period_start_range' : [1],
                'short_period_end_range' : [10,20],
                'short_period_step_range' : [1,3],
                'long_period_start_range' : [31,37],
                'long_period_end_range' : [54,58],
                'long_period_step_range' : [1,2]
            }
        },
        {
            # TS18
            "symbol": "RCL",
            "strategy": "macd_aroon",
            # 2024	{'aroon_period_range': 15, 'fast_period_range': 15, 'slow_period_range': 26, 'macd_signal_range': 20}
            # 2025	{'aroon_period_range': 15, 'fast_period_range': 15, 'slow_period_range': 22, 'macd_signal_range': 15}	
            "param_ranges": {
                # Define dynamic ranges for parameters
                'aroon_period_range' : [15],
                'fast_period_range' : [15],
                'slow_period_range' : [22,26],
                'macd_signal_range' : [15,20]
            }
        },
        {
            # TS19
            "symbol": "SYF",
            "strategy": "ema_confluence",
            # 2024	{'MA_1_range': 5, 'MA_2_range': 21, 'MA_3_range': 91, 'MA_4_range': 166}
            # 2025	{'MA_1_range': 5, 'MA_2_range': 21, 'MA_3_range': 91, 'MA_4_range': 131}	
            "param_ranges": {
                # Define dynamic ranges for EMA periods
                'MA_1_range' : [5],
                'MA_2_range' : [21],
                'MA_3_range' : [91],
                'MA_4_range' : [131,166]
            }
        },
        {
            # TS20
            "symbol": "TPL",
            "strategy": "hma_smi",
            # 2024	{'smi_period_range': 26, 'hma_period_range': 5}	
            # 2025	{'smi_period_range': 20, 'hma_period_range': 5}	
            "param_ranges": {
                # Define dynamic ranges for parameters
                'smi_period_range' : [20,26],  # SMI period range
                'hma_period_range' :[5]  # HMA period range
            }
        },
        {
            # TS21
            "symbol": "TPR",
            "strategy": "hma_atr",
            # 2024	{'hma_short_period_range': 5, 'hma_long_period_range': 21, 'atr_period_range': 26, 'atr_multiplier_range': 2, 'entry_shift_range': 3, 'exit_shift_range': 1}	
            # 2025	{'hma_short_period_range': 17, 'hma_long_period_range': 21, 'atr_period_range': 18, 'atr_multiplier_range': 2.5, 'entry_shift_range': 3, 'exit_shift_range': 5}
            "param_ranges": {
                # Define dynamic ranges for HMA and ATR parameters
                'hma_short_period_range' : [5,17],  # Range for short-term HMA periods
                'hma_long_period_range' : [21],  # Range for long-term HMA periods
                'atr_period_range' : [18,26],  # Range for ATR periods
                'atr_multiplier_range' : [2,2.5],  # ATR multiplier range
                'entry_shift_range' : [3],  # Range for entry shift
                'exit_shift_range' : [1,5]  # Range for exit shift
            }
        },
        {
            # TS22
            "symbol": "TRGP",
            "strategy": "ma_slope_atr_rsi",
            # 2024	{'ma_slope_period_range': 12, 'atr_period_range': 10, 'rsi_period_range': 29}	
            # 2025	{'ma_slope_period_range': 18, 'atr_period_range': 5, 'rsi_period_range': 19}	
            "param_ranges": {
                # Define dynamic ranges for periods
                'ma_slope_period_range' : [12,18],  # Range for MA Slope period
                'atr_period_range' : [5,10],  # Range for ATR period
                'rsi_period_range' : [19,29]  # Range for RSI period
            }
        },
        {
            # TS23
            "symbol": "UAL",
            "strategy": "macd_ravi",
            # 2024	{'ravi_short_period_range': 19, 'ravi_long_period_range': 20, 'macd_fast_period_range': 15, 'macd_slow_period_range': 26, 'macd_signal_period_range': 12, 'ravi_entry_thres_range': -20, 'ravi_exit_thres_range': -50}	
            # 2025	{'ravi_short_period_range': 16, 'ravi_long_period_range': 20, 'macd_fast_period_range': 15, 'macd_slow_period_range': 23, 'macd_signal_period_range': 12, 'ravi_entry_thres_range': -30, 'ravi_exit_thres_range': -50}	
            "param_ranges": {
                # Define dynamic ranges for RAVI and MACD periods
                'ravi_short_period_range' : [16,19],  # Range for RAVI short period
                'ravi_long_period_range' : [20],   # Range for RAVI long period
                'macd_fast_period_range' : [15],     # Range for MACD fast period
                'macd_slow_period_range' : [23,26],    # Range for MACD slow period
                'macd_signal_period_range' : [12],   # Range for MACD signal period
                'ravi_entry_thres_range' : [-20,-30],  # Range for RAVI entry threshold
                'ravi_exit_thres_range' : [-50]  # Range for RAVI exit threshold
            }
        },
        {
            # TS24
            "symbol": "VST",
            "strategy": "bollinger",
            # 2024	{'period_range': 23, 'std_dev_range': 3.0, 'squeeze_period_range': 6}	
            # 2025	{'period_range': 19, 'std_dev_range': 2.5, 'squeeze_period_range': 10}	
            "param_ranges": {
                # Define dynamic ranges for short and long windows
                'period_range' : [19,23],  # Range for Bollinger Bands period
                'std_dev_range' : [2.5, 3.0],  # Range for standard deviation (multiplier)
                'squeeze_period_range' : [6,10]  # Range for rolling window in squeeze condition
            }
        },
        {
            # TS25
            "symbol": "WMT",
            "strategy": "hma_heikin_ashi",
            # 2024	{'entry_shift_1_range': 7, 'entry_shift_2_range': 5, 'exit_shift_1_range': 5, 'exit_shift_2_range': 5, 'hma_period_range': 13}	
            # 2025	{'entry_shift_1_range': 3, 'entry_shift_2_range': 9, 'exit_shift_1_range': 9, 'exit_shift_2_range': 3, 'hma_period_range': 11}	
            "param_ranges": {
                'entry_shift_1_range' : [3,7],
                'entry_shift_2_range' : [5,9],
                'exit_shift_1_range' : [5,9],
                'exit_shift_2_range' : [3,5],
                'hma_period_range' : [11,13]
            }
        } 

    ]
}


portfolio_us_trading_2026 = {
    "Title": "US Trading 2026",  # Titolo del portafoglio
    "params_global": {
        "ratio": '4:1',  # Default ratio 4:1
        # "end_date": datetime.now(),
        "operating_freq": "1D",  # Frequenza operativa
        "init_cash": 100_000,  # Capitale iniziale
        "fees": 0.001,  # Commissioni trading
        "slippage": 0, # 0.002: Attenzione: utilizzato da APR 10, 2025
        "sl_stop": None, # 0.003: Attenzione: non utilizzato
        "price_col" : "Open",
        "benchmark" : "CSSPX.MI"
    },
    "trading_systems": [
        {
            # TS1
            "symbol": "AMD",
            "strategy": "spy_mr_crsi_bbands",
            "param_ranges": {
                # 2025	{'use_trend_rng': 0, 'sma_l_rng': 150, 'bb_p_rng': 25, 'bb_k10_rng': 20, 'crsi_thr_rng': 20, 'rsi2_exit_rng': 85}
                # 2026	{'use_trend_rng': 0, 'sma_l_rng': 150, 'bb_p_rng': 25, 'bb_k10_rng': 20, 'crsi_thr_rng': 20, 'rsi2_exit_rng': 85}	
                'use_trend_rng' : [0],  
                'sma_l_rng' : [150],  
                'bb_p_rng' : [25], 
                'bb_k10_rng' : [20],  
                'crsi_thr_rng' : [20],
                'rsi2_exit_rng' : [85] 
            },
        },
        {
            # TS2
            "symbol": "AMGN",
            "strategy": "kvo_cci",
                "param_ranges": {
                    # 2025	{'fast_range': 15, 'slow_range': 30, 'signal_range': 20, 'cci_period_range': 20, 'cci_shift_range': 1}
                    # 2026	{'fast_range': 15, 'slow_range': 35, 'signal_range': 20, 'cci_period_range': 10, 'cci_shift_range': 1}
                    'fast_range': [15],
                    'slow_range': [30,35],
                    'signal_range': [20],
                    'cci_period_range': [10,20],
                    'cci_shift_range': [1]
          }
        },
        {
            # TS3
            "symbol": "C",
            "strategy": "tema_heikin_ashi",
            "param_ranges": {
                # 2025	{'entry_shift_1_range': 1, 'entry_shift_2_range': 1, 'exit_shift_1_range': 7, 'exit_shift_2_range': 1, 'tema_period_range': 16}	
                # 2026	{'entry_shift_1_range': 1, 'entry_shift_2_range': 1, 'exit_shift_1_range': 7, 'exit_shift_2_range': 1, 'tema_period_range': 16}
                'entry_shift_1_range' : [1],  
                'entry_shift_2_range' : [2],  
                'exit_shift_1_range' : [7],  
                'exit_shift_2_range' : [1],  
                'tema_period_range' : [16] 
            }
        },
        {
            # TS4
            "symbol": "CAT",
            "strategy": "pcr_ma",
            "param_ranges": {
                # 2025	{'pcr_upper_bp_range': 145, 'pcr_lower_bp_range': 55, 'pcr_smooth_range': 8, 'ma_len_range': 20, 'direction_mode_range': 1}	
                # 2026	{'pcr_upper_bp_range': 145, 'pcr_lower_bp_range': 70, 'pcr_smooth_range': 8, 'ma_len_range': 20, 'direction_mode_range': 1}
                'pcr_upper_bp_range' : [145],
                'pcr_lower_bp_range' : [55,70],
                'pcr_smooth_range' : [8],
                'ma_len_range' : [20],
                'direction_mode_range' : [1]
            }
        },
        { 
            # TS5
            "symbol": "COF",
            "strategy": "momentum_rank1d",
            "param_ranges": {
                # 2025	{'lb1_range': 20, 'lb2_range': 120, 'lb3_range': 120, 'ema_span_range': 0, 'thresh_range': 0.05, 'use_median_range': True}	
                # 2026	{'lb1_range': 60, 'lb2_range': 120, 'lb3_range': 120, 'ema_span_range': 0, 'thresh_range': 0.05, 'use_median_range': True}	
                'lb1_range' : [20,60],
                'lb2_range' : [120],
                'lb3_range' : [0],
                'ema_span_range' : [0],
                'thresh_range' : [0.05],
                'use_median_range' : [True]
            }
        },
        {
            # TS6
            "symbol": "CSCO",
            "strategy": "zscore_mr_momo",
            "param_ranges": {
                # 2025	{'ema_trend_span_range': 30, 'adx_period_range': 14, 'adx_thresh_range': 20, 'z_window_range': 40, 'entry_z_range': 2.0, 'exit_z_range': 2.0, 'z_shift_range': 1, 'donchian_range': 20, 'momo_shift_range': 1}
                # 2026	{'ema_trend_span_range': 30, 'adx_period_range': 10, 'adx_thresh_range': 20, 'z_window_range': 40, 'entry_z_range': 2.0, 'exit_z_range': 2.0, 'z_shift_range': 1, 'donchian_range': 20, 'momo_shift_range': 1}
                'ema_trend_span_range' : [30],  
                'adx_period_range' : [10,14],  
                'adx_thresh_range' : [20],  
                'z_window_range' : [40],  
                'entry_z_range' : [2.0],  
                'exit_z_range' : [2.0],  
                'z_shift_range' : [1],  
                'donchian_range' : [20],  
                'momo_shift_range' : [1]  
            }
        },
        {
            # TS7
            "symbol": "GM",
            "strategy": "spy_circuit_mix",
            "param_ranges": {
                # 2025	{'one_day_drop_bp_range': 800, 'three_day_drop_bp_range': 1200, 'dd_fast_bp_range': 800, 'vratio_up_tenths_range': 18, 'wk_ema_weeks_range': 18, 'wk_roc_weeks_range': 13, 'wk_roc_neg_thr_bp_range': 600, 'wk_brk_weeks_range': 10, 'wk_persist_out_range': 2, 'ma_fast_range': 10, 'vratio_dn_tenths_range': 12, 'ftd_bp_range': 120, 'reentry_cooldown_range': 0, 'reentry_mode_range': 2}
                # 2026	{'one_day_drop_bp_range': 900, 'three_day_drop_bp_range': 1200, 'dd_fast_bp_range': 800, 'vratio_up_tenths_range': 18, 'wk_ema_weeks_range': 18, 'wk_roc_weeks_range': 13, 'wk_roc_neg_thr_bp_range': 800, 'wk_brk_weeks_range': 10, 'wk_persist_out_range': 1, 'ma_fast_range': 10, 'vratio_dn_tenths_range': 12, 'ftd_bp_range': 120, 'reentry_cooldown_range': 0, 'reentry_mode_range': 0}
                'one_day_drop_bp_range' : [800,900],
                'three_day_drop_bp_range' : [1200],
                'dd_fast_bp_range' : [800],
                'vratio_up_tenths_range' : [18],
                'wk_ema_weeks_range' : [18],
                'wk_roc_weeks_range' : [13],
                'wk_roc_neg_thr_bp_range' : [600,800],
                'wk_brk_weeks_range' : [10],
                'wk_persist_out_range' : [1,2],
                'ma_fast_range' : [10],
                'vratio_dn_tenths_range' : [12],
                'ftd_bp_range' : [120],
                'reentry_cooldown_range' : [0],
                'reentry_mode_range' : [0,2]
            }
        },
        {
            # TS8
            "symbol": "GOOGL",
            "strategy": "tp_ma_crossover",
            "param_ranges": {
                # 2025	{'short_window_range': 45, 'long_window_range': 104}
                # 2026	{'short_window_range': 31, 'long_window_range': 106}	
                'short_window_range' :[31,45],
                'long_window_range' : [104,106]
            }
        },
        {
            # TS9
            "symbol": "MS",
            "strategy": "bollinger",
            "param_ranges": {
                # 2025	{'period_range': 6, 'std_dev_range': 1.5, 'squeeze_period_range': 9}
                # 2026	{'period_range': 6, 'std_dev_range': 1.5, 'squeeze_period_range': 5}
                'period_range' : [6], 
                'std_dev_range' : [1.5], 
                'squeeze_period_range' : [5,0] 
            }
        },
        {
            # TS10
            "symbol": "LLY",
            "strategy": "holding"
        }
        
    ]
}

portfolio_euro_trading_2026 = {
    "Title": "Euro Trading 2026",  # Titolo del portafoglio
    "params_global": {
        "ratio": '4:1',  # Default ratio 4:1
        "end_date": datetime.now(),
        "operating_freq": "1D",  # Frequenza operativa
        "init_cash": 100_000,  # Capitale iniziale
        "fees": 0.001,  # Commissioni trading
        "slippage": 0, # 0.002: Attenzione: utilizzato da APR 10, 2025
        "sl_stop": None, # 0.003: Attenzione: non utilizzato
        "price_col" : "Open",
        "benchmark" : "^STOXX50E"
        # "benchmark" : "TLT"
    },
    "trading_systems": [
        {
            # TS1
            "symbol": "ACS.MC",
            "strategy": "spy_quarter_switch",
            "param_ranges": {
                # 2025	{'ema_slow_range': 180, 'roc_win_range': 63, 'roc_neg_thr_bp_range': 750, 'rv_upper_pct_range': 75, 'persist_days_range': 5, 'breakout_win_range': 100, 'slope_lb_range': 20, 'reentry_mode_range': 1}	
                # 2026	{'ema_slow_range': 180, 'roc_win_range': 63, 'roc_neg_thr_bp_range': 300, 'rv_upper_pct_range': 75, 'persist_days_range': 5, 'breakout_win_range': 100, 'slope_lb_range': 20, 'reentry_mode_range': 1}
                'ema_slow_range' : [180],  
                'roc_win_range' : [63],  
                'roc_neg_thr_bp_range' : [300,750], 
                'rv_upper_pct_range' : [75],  
                'persist_days_range' : [5],
                'breakout_win_range' : [100],
                'slope_lb_range' : [20],
                'reentry_mode_range' : [1] 
            },
        },
        {
            # TS2
            "symbol": "BAYN.DE",
            "strategy": "wad_cks",
                "param_ranges": {
                    # 2025	{'atr_period': 29, 'stop_period': 11, 'multiplier': 2.0, 'rise_shift': 5, 'fall_shift': 15}
                    # 2026	{'atr_period': 23, 'stop_period': 11, 'multiplier': 2.0, 'rise_shift': 20, 'fall_shift': 30}
                    'atr_period': [23,29],
                    'stop_period': [11],
                    'multiplier': [2.0],
                    'rise_shift': [5,20],
                    'fall_shift': [15,30]
          }
        },
        {
            # TS3
            "symbol": "IBE.MC",
            "strategy": "macd_daily_weekly",
            "param_ranges": {
                # 2025	{'short_window_range': 7, 'long_window_range': 20, 'signal_window_range': 7}
                # 2026	{'short_window_range': 6, 'long_window_range': 23, 'signal_window_range': 8}
                'short_window_range' : [6,7],  
                'long_window_range' : [20,23],  
                'signal_window_range' : [7,8]
            }
        },
        {
            # TS4
            "symbol": "IFX.DE",
            "strategy": "ko_bb",
            "param_ranges": {
                # 2025	{'fast_period_range': 32, 'slow_period_range': 56, 'period_range': 17, 'std_dev_range': 1.5, 'squeeze_window_range': 14}
                # 2026	{'fast_period_range': 38, 'slow_period_range': 58, 'period_range': 17, 'std_dev_range': 1.5, 'squeeze_window_range': 10}
                'fast_period_range' : [32,38],
                'slow_period_range' : [56,58],
                'period_range' : [17],
                'std_dev_range' : [1.5],
                'squeeze_window_range' : [10,14]
            }
        },
        {
            # TS5
            "symbol": "ITX.MC",
            "strategy": "mom_trend_atr_vix",
            "param_ranges": {
                # 2025	{'sma_l_range': 220, 'roc_lb_range': 150, 'atr_p_range': 14, 'k_entry10_rng': 10, 'k_exit10_rng': 25, 'risk_ma_rng': 5, 'risk_thr10_rng': 0, 'risk_persist_rng': 0}
                # 2026	{'sma_l_range': 170, 'roc_lb_range': 140, 'atr_p_range': 14, 'k_entry10_rng': 10, 'k_exit10_rng': 25, 'risk_ma_rng': 5, 'risk_thr10_rng': 0, 'risk_persist_rng': 0}
                'sma_l_range' : [170,220],
                'roc_lb_range' : [140,150],
                'atr_p_range' : [14],
                'k_entry10_rng' : [10],
                'k_exit10_rng' : [25],
                'risk_ma_rng' : [5],
                'risk_thr10_rng' : [0],
                'risk_persist_rng' : [0]
            }
        },
        {
            # TS6
            "symbol": "MC.PA",
            "strategy": "cts_dpo",
            "param_ranges": {
                # 2025	{'cts_period_range': 26, 'dpo_period_range': 7, 'cts_upper_range': 20, 'cts_lower_range': -20}
                # 2026	{'cts_period_range': 26, 'dpo_period_range': 7, 'cts_upper_range': 20, 'cts_lower_range': -20}
                'cts_period_range' : [26],  
                'dpo_period_range' : [7],  
                'cts_upper_range' : [20],  
                'cts_lower_range' : [-20]
            }
        },
        {
            # TS7
            "symbol": "MRK.DE",
            "strategy": "macd_kvo",
            "param_ranges": {
                # 2025	{'short_period_range': 30, 'long_period_range': 65, 'signal_period_range': 5, 'fast_period_range': 10, 'slow_period_range': 20, 'macd_signal_range': 13}
                # 2026	{'short_period_range': 30, 'long_period_range': 60, 'signal_period_range': 10, 'fast_period_range': 10, 'slow_period_range': 20, 'macd_signal_range': 11}
                'short_period_range' : [30],
                'long_period_range' : [60,65],
                'signal_period_range' : [5,10],
                'fast_period_range' : [10],
                'slow_period_range' : [20],
                'macd_signal_range' : [11,13]
            }
        },
        {
            # TS8
            "symbol": "RWE.DE",
            "strategy": "macd_daily_weekly",
            "param_ranges": {
                # 2025	{'short_window_range': 9, 'long_window_range': 21, 'signal_window_range': 10}
                # 2026	{'short_window_range': 6, 'long_window_range': 24, 'signal_window_range': 12}	
                'short_window_range' :[6,9],
                'long_window_range' :[21,24],
                'signal_window_range' : [10,12]
            }
        },
        {
            # TS9
            "symbol": "SAN.MC",
            "strategy": "spy_2of3_switch",
            "param_ranges": {
                # 2025	{'sma_fast_range': 50, 'sma_slow_range': 240, 'slope_lb_range': 30, 'persist_in_range': 1, 'persist_out_range': 7, 'roc_win_range': 63, 'roc_neg_thr_bp_range': 1000, 'brk_win_range': 100, 'reentry_mode_range': 1}
                # 2026	{'sma_fast_range': 40, 'sma_slow_range': 220, 'slope_lb_range': 30, 'persist_in_range': 1, 'persist_out_range': 7, 'roc_win_range': 63, 'roc_neg_thr_bp_range': 400, 'brk_win_range': 100, 'reentry_mode_range': 1}
                'sma_fast_range' : [40,50], 
                'sma_slow_range' : [240], 
                'slope_lb_range' : [30], 
                'persist_in_range' : [1], 
                'persist_out_range' : [7], 
                'roc_win_range' : [63], 
                'roc_neg_thr_bp_range' : [400,1000], 
                'brk_win_range' : [100], 
                'reentry_mode_range' : [1] 
            }
        },
        {
            # TS10
            "symbol": "STLAM.MI",
            "strategy": "dma",
            "param_ranges": {
                # 2025	{'dma_fast_period_range': 51, 'dma_slow_period_range': 81, 'dma_fast_shift_range': 16, 'dma_slow_shift_range': 6}
                # 2026	{'dma_fast_period_range': 51, 'dma_slow_period_range': 81, 'dma_fast_shift_range': 16, 'dma_slow_shift_range': 6} 
                'dma_fast_period_range' : [51], 
                'dma_slow_period_range' : [81], 
                'dma_fast_shift_range' : [16], 
                'dma_slow_shift_range' : [6]
            }
        }
    ]
}





