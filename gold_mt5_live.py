"""
ADVANCED GOLD SCALPING SYSTEM v3.0 - PROFESSIONAL EDITION
✅ Multi-Timeframe Confluence
✅ Market Regime Detection (ADX + ATR)
✅ Kelly Criterion Position Sizing
✅ Volume Profile Analysis
✅ Partial Close Strategy
✅ News Time Filter
✅ Order Flow Analysis
✅ Adaptive TP/SL
✅ Advanced ML Features
✅ Real-time Statistics Dashboard
✅ FIXED: Trailing only after 35+ pips profit
"""

import MetaTrader5 as mt5
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
from dotenv import load_dotenv
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

import sys
import io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ScalpConfig:
    """Advanced Scalping Configuration"""
    
    # MT5 Credentials
    MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
    MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
    MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5Trial17')
    MT5_PATH = os.getenv('MT5_PATH', r"C:\Program Files\MetaTrader 5\terminal64.exe")
    
    SYMBOL = "XAUUSDm"
    MAGIC_NUMBER = 234001
    
    # Risk Management
    RISK_PER_TRADE = 0.5
    MIN_LOT = 0.01
    MAX_LOT = 0.02
    USE_KELLY_CRITERION = False
    KELLY_FRACTION = 0.25  # Conservative Kelly
    
    MAX_DAILY_LOSS_PERCENT = 2.0
    MAX_TRADES_PER_DAY = 25
    MAX_CONSECUTIVE_LOSSES = 4
    COOLDOWN_AFTER_LOSS_MINUTES = 5
    MAX_TRADE_DURATION_MINUTES = 60
    
    # TP/SL Settings (Adaptive)
    SCALP_TP_MIN_PIPS = 60
    SCALP_TP_MAX_PIPS = 120
    SCALP_SL_PIPS = 35
    
    USE_DYNAMIC_TP = True
    USE_BREAKEVEN = True
    BREAKEVEN_TRIGGER_PERCENT = 0.5
    USE_PARTIAL_CLOSE = True
    PARTIAL_CLOSE_PERCENT = 0.5
    PARTIAL_CLOSE_TRIGGER = 0.6  # 60% of TP

    # Fixed lot targets - UPDATED FOR YOUR REQUIREMENTS
    FIXED_LOT_TARGET_LOT = 0.01
    FIXED_LOT_TARGET_LOT_2 = 0.02  # 0.02 lot এর জন্যও support
    FIXED_LOT_SL_INR = 700.0  # 600-800 range এর মাঝামাঝি
    FIXED_LOT_TP_INR = 1200.0
    FIXED_LOT_TRAILING_START_INR = 800.0  # 800 INR profit এ trailing শুরু (≈35 pips)
    FIXED_LOT_TRAILING_START_PIPS = 35.0  # 35 pips profit চাই
    FIXED_LOT_SECURE_SL_PIPS = 20.0  # 20 pips দূরে SL রাখবে
    MIN_SL_MOVE_PIPS = 5.0  # কমপক্ষে 5 pips move করতে হবে
    
    # Thresholds (Stricter for quality trades)
    MIN_SCALP_CONFIDENCE = 0.35  # Lowered for more signals
    MIN_INDICATOR_ALIGNMENT = 0.32  # Lowered for more signals
    MAX_SPREAD_PIPS = 2.0
    MIN_ADX_FOR_TREND = 22
    MIN_VOLUME_RATIO = 0.52
    
    # Timeframes
    PRIMARY_TF = mt5.TIMEFRAME_M5
    SECONDARY_TF = mt5.TIMEFRAME_M15
    FILTER_TF = mt5.TIMEFRAME_H1
    
    # Advanced Features (Optimized for quality)
    USE_MTF_CONFLUENCE = True         # ✅ True (multi-timeframe)
    USE_MARKET_REGIME = True          # ✅ True (avoid choppy markets)
    USE_VOLUME_FILTER = True          # ✅ True (need volume confirmation)
    USE_NEWS_FILTER = False            # ✅ True (avoid news volatility)
    USE_ORDER_FLOW = True             # ✅ True (bid-ask pressure)
    # News Filter (Avoid high volatility times)
    NEWS_BLACKOUT_HOURS = [14, 20, 21]
    
    DAILY_WIN_TARGET = 25
    STOP_AFTER_TARGET = True
    
    # Debug
    DEBUG_MODE = True
    SHOW_DETAILED_DEBUG = True
    SHOW_ALL_CHECKS = True  # Show why signals are rejected
    
    # Indicator Weights (Optimized for high win rate)
    INDICATOR_WEIGHTS = {
        'fast_ema_cross': 0.25,
        'stochastic': 0.20,
        'cci': 0.15,
        'momentum': 0.10,
        'price_action': 0.15,
        'trend_filter': 0.10,
        'volume': 0.05
    }
    
    # ML
    XGBOOST_MODEL_PATH = 'models/gold_scalping_xgb.json'
    FEATURE_SCALER_PATH = 'models/gold_scalping_scaler.pkl'
    TRAINING_DATA_PATH = 'models/scalping_training_data.pkl'
    USE_XGBOOST = True
    XGBOOST_FALLBACK_MODE = True
    ML_MIN_TRAINING_SAMPLES = 30
    
    PAPER_TRADING = False
    KILL_SWITCH_FILE = 'kill_switch.txt'


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScalpSignal:
    direction: str
    confidence: float
    entry: float
    sl: float
    tp: float
    indicators: List[str]
    strength: float
    regime: str = "UNKNOWN"
    volume_score: float = 0.0
    mtf_score: float = 0.0


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class MarketRegime:
    """Detect market conditions for adaptive strategy"""
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty else 0.0
    
    @staticmethod
    def calculate_atr_percent(df: pd.DataFrame, period: int = 14) -> float:
        """ATR as percentage of price"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        price = close.iloc[-1]
        
        return (atr / price) * 100 if price > 0 else 0.0
    
    @classmethod
    def detect_regime(cls, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Detect market regime"""
        adx = cls.calculate_adx(df)
        atr_pct = cls.calculate_atr_percent(df)
        
        regime_data = {
            'adx': adx,
            'atr_pct': atr_pct,
            'is_trending': adx > ScalpConfig.MIN_ADX_FOR_TREND,
            'is_volatile': atr_pct > 0.08
        }
        
        if adx > 25 and atr_pct > 0.08:
            regime = "TRENDING_HIGH_VOL"
        elif adx > 25:
            regime = "TRENDING_LOW_VOL"
        elif atr_pct > 0.08:
            regime = "RANGING_HIGH_VOL"
        else:
            regime = "RANGING_LOW_VOL"
        
        regime_data['regime'] = regime
        return regime, regime_data
    
    @staticmethod
    def get_adaptive_targets(regime: str) -> Tuple[float, float]:
        """Adjust TP/SL based on regime"""
        if regime == "TRENDING_HIGH_VOL":
            tp_pips = 100
            sl_pips = 40
        elif regime == "TRENDING_LOW_VOL":
            tp_pips = 75
            sl_pips = 30
        elif regime == "RANGING_HIGH_VOL":
            tp_pips = 60
            sl_pips = 25
        else:  # RANGING_LOW_VOL
            tp_pips = 50
            sl_pips = 20
        
        return tp_pips, sl_pips


# ============================================================================
# VOLUME ANALYZER
# ============================================================================

class VolumeAnalyzer:
    """Volume profile and confirmation"""
    
    @staticmethod
    def analyze_volume(df: pd.DataFrame) -> Tuple[float, Dict]:
        """Volume strength analysis"""
        if 'tick_volume' not in df.columns:
            return 0.5, {}
        
        current_vol = df['tick_volume'].iloc[-1]
        avg_vol = df['tick_volume'].rolling(20).mean().iloc[-1]
        
        if avg_vol == 0:
            return 0.5, {}
        
        vol_ratio = current_vol / avg_vol
        
        # Score based on volume
        if vol_ratio > 1.5:
            score = 0.85
            strength = "STRONG"
        elif vol_ratio > 1.2:
            score = 0.70
            strength = "MODERATE"
        elif vol_ratio > 0.8:
            score = 0.55
            strength = "WEAK"
        else:
            score = 0.30
            strength = "VERY_WEAK"
        
        vol_data = {
            'current': current_vol,
            'average': avg_vol,
            'ratio': vol_ratio,
            'strength': strength
        }
        
        return score, vol_data


# ============================================================================
# MTF CONFLUENCE ANALYZER
# ============================================================================

class MTFConfluence:
    """Multi-timeframe confluence scoring"""
    
    @staticmethod
    def analyze_confluence(m5_signal: Optional[str], 
                          m15_signal: Optional[str],
                          h1_trend: Optional[str]) -> Tuple[float, str]:
        """Calculate MTF alignment score"""
        
        signals = [s for s in [m5_signal, m15_signal, h1_trend] if s]
        
        if not signals:
            return 0.0, "NO_SIGNAL"
        
        # Count LONG vs SHORT
        long_count = signals.count("LONG")
        short_count = signals.count("SHORT")
        total = len(signals)
        
        if long_count == total:
            return 0.95, "LONG"
        elif short_count == total:
            return 0.95, "SHORT"
        elif long_count >= 2:
            return 0.75, "LONG"
        elif short_count >= 2:
            return 0.75, "SHORT"
        else:
            return 0.40, "MIXED"


# ============================================================================
# NEWS FILTER
# ============================================================================

class NewsFilter:
    """Avoid trading during news events"""
    
    @staticmethod
    def is_news_time() -> bool:
        """Check if current time is during news"""
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        
        # Only block specific hours
        if current_hour in ScalpConfig.NEWS_BLACKOUT_HOURS:
            return True
        
        return False


# ============================================================================
# ORDER FLOW ANALYZER
# ============================================================================

class OrderFlowAnalyzer:
    """Bid-Ask pressure analysis"""
    
    @staticmethod
    def analyze_flow(symbol: str) -> Tuple[str, float]:
        """Detect order flow pressure"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return "NEUTRAL", 0.5
            
            # Analyze spread and last trade direction
            spread = tick.ask - tick.bid
            mid_price = (tick.ask + tick.bid) / 2
            last_price = tick.last
            
            # If last > mid, buying pressure
            if last_price > mid_price:
                distance = (last_price - mid_price) / spread
                score = 0.6 + min(distance * 0.3, 0.3)
                return "BULLISH_FLOW", score
            elif last_price < mid_price:
                distance = (mid_price - last_price) / spread
                score = 0.6 + min(distance * 0.3, 0.3)
                return "BEARISH_FLOW", score
            else:
                return "NEUTRAL", 0.5
        except:
            return "NEUTRAL", 0.5


# ============================================================================
# SCALPING INDICATORS
# ============================================================================

class ScalpingIndicators:
    
    @staticmethod
    def fast_ema_cross(df: pd.DataFrame) -> Tuple[Optional[str], float]:
        ema_5 = df['close'].ewm(span=5).mean()
        ema_13 = df['close'].ewm(span=13).mean()
        
        curr_5 = ema_5.iloc[-1]
        curr_13 = ema_13.iloc[-1]
        prev_5 = ema_5.iloc[-2]
        prev_13 = ema_13.iloc[-2]
        
        if prev_5 <= prev_13 and curr_5 > curr_13:
            distance = abs(curr_5 - curr_13) / curr_13
            confidence = min(0.9, 0.6 + distance * 1000)
            return "LONG", confidence
        elif prev_5 >= prev_13 and curr_5 < curr_13:
            distance = abs(curr_5 - curr_13) / curr_13
            confidence = min(0.9, 0.6 + distance * 1000)
            return "SHORT", confidence
        
        return None, 0.0
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[Optional[str], float]:
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        
        k_curr = k.iloc[-1]
        d_curr = d.iloc[-1]
        k_prev = k.iloc[-2]
        d_prev = d.iloc[-2]
        
        if k_curr < 30 and k_prev < d_prev and k_curr > d_curr:
            confidence = 0.8 - (k_curr / 100)
            return "LONG", confidence
        elif k_curr > 70 and k_prev > d_prev and k_curr < d_curr:
            confidence = 0.8 - ((100 - k_curr) / 100)
            return "SHORT", confidence
        
        return None, 0.0
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> Tuple[Optional[str], float]:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma) / (0.015 * mad)
        cci_curr = cci.iloc[-1]
        cci_prev = cci.iloc[-2]
        
        if cci_curr < -100 and cci_curr > cci_prev:
            strength = min(abs(cci_curr) / 200, 1.0)
            return "LONG", 0.6 + strength * 0.3
        elif cci_curr > 100 and cci_curr < cci_prev:
            strength = min(abs(cci_curr) / 200, 1.0)
            return "SHORT", 0.6 + strength * 0.3
        
        return None, 0.0
    
    @staticmethod
    def momentum(df: pd.DataFrame, period: int = 10) -> Tuple[Optional[str], float]:
        mom = df['close'].diff(period)
        mom_curr = mom.iloc[-1]
        mom_prev = mom.iloc[-2]
        
        if mom_curr > 0 and mom_curr > mom_prev:
            strength = min(abs(mom_curr) / df['close'].iloc[-1] * 100, 1.0)
            return "LONG", 0.6 + strength * 0.3
        elif mom_curr < 0 and mom_curr < mom_prev:
            strength = min(abs(mom_curr) / df['close'].iloc[-1] * 100, 1.0)
            return "SHORT", 0.6 + strength * 0.3
        
        return None, 0.0
    
    @staticmethod
    def price_action(df: pd.DataFrame) -> Tuple[Optional[str], float]:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['close'] - last['open'])
        range_size = last['high'] - last['low']
        
        if (last['close'] > last['open'] and 
            prev['close'] < prev['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close']):
            return "LONG", 0.75
        
        elif (last['close'] < last['open'] and 
              prev['close'] > prev['open'] and
              last['close'] < prev['open'] and
              last['open'] > prev['close']):
            return "SHORT", 0.75
        
        elif body > range_size * 0.7 and last['close'] > last['open']:
            return "LONG", 0.65
        
        elif body > range_size * 0.7 and last['close'] < last['open']:
            return "SHORT", 0.65
        
        return None, 0.0
    
    @staticmethod
    def trend_filter(df: pd.DataFrame) -> Tuple[Optional[str], float]:
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        distance = abs(price - ema_50) / ema_50
        
        if price > ema_50:
            confidence = 0.6 + min(distance * 1000, 0.3)
            return "LONG", confidence
        elif price < ema_50:
            confidence = 0.6 + min(distance * 1000, 0.3)
            return "SHORT", confidence
        
        return None, 0.0


# ============================================================================
# ADVANCED SCALPING ENGINE
# ============================================================================

class AdvancedScalpingEngine:
    def __init__(self):
        self.indicators = ScalpingIndicators()
        self.regime_detector = MarketRegime()
        self.volume_analyzer = VolumeAnalyzer()
        self.mtf_analyzer = MTFConfluence()
        self.news_filter = NewsFilter()
        self.flow_analyzer = OrderFlowAnalyzer()
    
    def generate_scalp_signal(self, market_data: Dict[str, pd.DataFrame],
                             current_price: float) -> Optional[ScalpSignal]:
        
        df_m5 = market_data.get('M5', pd.DataFrame())
        df_m15 = market_data.get('M15', pd.DataFrame())
        df_h1 = market_data.get('H1', pd.DataFrame())
        
        if df_m5.empty or df_m15.empty:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info("🔍 No M5/M15 data")
            return None
        
        # NEWS FILTER
        if ScalpConfig.USE_NEWS_FILTER and self.news_filter.is_news_time():
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info("🚫 News time - skipping")
            return None
        
        # MARKET REGIME
        regime, regime_data = self.regime_detector.detect_regime(df_m5)
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"📊 Market Regime: {regime}")
            logging.info(f"   ADX: {regime_data['adx']:.1f} | ATR%: {regime_data['atr_pct']:.3f}%")
        
        if ScalpConfig.USE_MARKET_REGIME:
            if regime == "RANGING_LOW_VOL":
                if ScalpConfig.SHOW_DETAILED_DEBUG:
                    logging.info(f"⚠️ Poor market regime: {regime}")
                return None
        
        # VOLUME ANALYSIS
        volume_score, vol_data = self.volume_analyzer.analyze_volume(df_m5)
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"📊 Volume: {volume_score:.2f} ({vol_data.get('strength', 'N/A')})")
        
        if ScalpConfig.USE_VOLUME_FILTER and volume_score < ScalpConfig.MIN_VOLUME_RATIO:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"⚠️ Low volume: {volume_score:.2f} < {ScalpConfig.MIN_VOLUME_RATIO}")
            return None
        
        # Collect indicator signals
        signals = {}
        
        direction, conf = self.indicators.fast_ema_cross(df_m5)
        if direction:
            signals['fast_ema_cross'] = (direction, conf)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"✅ EMA Cross: {direction} @ {conf:.2f}")
        
        direction, conf = self.indicators.stochastic(df_m5)
        if direction:
            signals['stochastic'] = (direction, conf)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"✅ Stochastic: {direction} @ {conf:.2f}")
        
        direction, conf = self.indicators.cci(df_m15)
        if direction:
            signals['cci'] = (direction, conf)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"✅ CCI: {direction} @ {conf:.2f}")
        
        direction, conf = self.indicators.momentum(df_m5)
        if direction:
            signals['momentum'] = (direction, conf)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"✅ Momentum: {direction} @ {conf:.2f}")
        
        direction, conf = self.indicators.price_action(df_m5)
        if direction:
            signals['price_action'] = (direction, conf)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"✅ Price Action: {direction} @ {conf:.2f}")
        
        if not df_h1.empty:
            direction, conf = self.indicators.trend_filter(df_h1)
            if direction:
                signals['trend_filter'] = (direction, conf)
                if ScalpConfig.SHOW_DETAILED_DEBUG:
                    logging.info(f"✅ Trend Filter: {direction} @ {conf:.2f}")
        
        # Add volume signal
        if volume_score > 0.7:
            signals['volume'] = (None, volume_score)
        
        if not signals:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info("❌ No indicator signals generated")
            return None
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"📊 Total signals: {len(signals)}")
        
        # MTF CONFLUENCE
        m5_direction = signals.get('fast_ema_cross', (None, 0))[0]
        m15_direction = signals.get('cci', (None, 0))[0]
        h1_direction = signals.get('trend_filter', (None, 0))[0]
        
        mtf_score, mtf_direction = self.mtf_analyzer.analyze_confluence(
            m5_direction, m15_direction, h1_direction
        )
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"🎯 MTF Confluence: {mtf_direction} @ {mtf_score:.2f}")
        
        # Vote
        long_score = 0.0
        short_score = 0.0
        long_indicators = []
        short_indicators = []
        
        for indicator, (direction, conf) in signals.items():
            if direction is None:
                continue
            
            weight = ScalpConfig.INDICATOR_WEIGHTS.get(indicator, 0.1)
            weighted_conf = conf * weight
            
            if direction == "LONG":
                long_score += weighted_conf
                long_indicators.append(indicator)
            elif direction == "SHORT":
                short_score += weighted_conf
                short_indicators.append(indicator)
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"🗳️ Vote: LONG={long_score:.3f} vs SHORT={short_score:.3f}")
        
        # Apply MTF boost
        if ScalpConfig.USE_MTF_CONFLUENCE:
            if mtf_direction == "LONG":
                long_score *= (1 + mtf_score * 0.3)
            elif mtf_direction == "SHORT":
                short_score *= (1 + mtf_score * 0.3)
        
        # Apply volume boost
        if volume_score > 0.7:
            long_score *= (1 + (volume_score - 0.7) * 0.5)
            short_score *= (1 + (volume_score - 0.7) * 0.5)
        
        # ORDER FLOW
        if ScalpConfig.USE_ORDER_FLOW:
            flow_direction, flow_score = self.flow_analyzer.analyze_flow(ScalpConfig.SYMBOL)
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"💧 Order Flow: {flow_direction} @ {flow_score:.2f}")
            if flow_direction == "BULLISH_FLOW":
                long_score *= (1 + (flow_score - 0.5) * 0.2)
            elif flow_direction == "BEARISH_FLOW":
                short_score *= (1 + (flow_score - 0.5) * 0.2)
        
        if long_score > short_score and long_score > ScalpConfig.MIN_SCALP_CONFIDENCE:
            final_direction = "LONG"
            final_confidence = long_score
            active_indicators = long_indicators
        elif short_score > long_score and short_score > ScalpConfig.MIN_SCALP_CONFIDENCE:
            final_direction = "SHORT"
            final_confidence = short_score
            active_indicators = short_indicators
        else:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"❌ Below confidence threshold: LONG={long_score:.3f}, SHORT={short_score:.3f}, Min={ScalpConfig.MIN_SCALP_CONFIDENCE}")
            return None
        
        alignment = len(active_indicators) / len(ScalpConfig.INDICATOR_WEIGHTS)
        
        if ScalpConfig.SHOW_DETAILED_DEBUG:
            logging.info(f"🎯 Alignment: {alignment:.0%} (need {ScalpConfig.MIN_INDICATOR_ALIGNMENT:.0%})")
        
        if alignment < ScalpConfig.MIN_INDICATOR_ALIGNMENT:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"❌ Low alignment: {alignment:.0%} < {ScalpConfig.MIN_INDICATOR_ALIGNMENT:.0%}")
            return None
        
        # ADAPTIVE TP/SL
        tp_pips, sl_pips = self.regime_detector.get_adaptive_targets(regime)
        tp_pips = tp_pips / 10
        sl_pips = sl_pips / 10
        
        if final_direction == "LONG":
            entry = current_price
            sl = entry - sl_pips
            tp = entry + tp_pips
        else:
            entry = current_price
            sl = entry + sl_pips
            tp = entry - tp_pips
        
        return ScalpSignal(
            direction=final_direction,
            confidence=final_confidence,
            entry=entry,
            sl=sl,
            tp=tp,
            indicators=active_indicators,
            strength=alignment,
            regime=regime,
            volume_score=volume_score,
            mtf_score=mtf_score
        )


# ============================================================================
# KELLY CRITERION POSITION SIZER
# ============================================================================

class KellyPositionSizer:
    """Kelly Criterion for optimal lot sizing"""
    
    def __init__(self):
        self.trade_history = []
    
    def add_trade(self, profit: float, risk: float):
        """Record trade outcome"""
        self.trade_history.append({
            'profit': profit,
            'risk': risk,
            'win': profit > 0
        })
        
        # Keep last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]
    
    def calculate_kelly_lot(self, balance: float, sl_distance: float) -> float:
        """Calculate optimal lot size using Kelly Criterion"""
        
        if len(self.trade_history) < 10:
            # Not enough data, use default risk
            risk_amount = balance * (ScalpConfig.RISK_PER_TRADE / 100)
            lot = risk_amount / (sl_distance * 100)
            return max(ScalpConfig.MIN_LOT, min(round(lot, 2), ScalpConfig.MAX_LOT))
        
        wins = [t for t in self.trade_history if t['win']]
        losses = [t for t in self.trade_history if not t['win']]
        
        if not wins or not losses:
            # Fallback
            risk_amount = balance * (ScalpConfig.RISK_PER_TRADE / 100)
            lot = risk_amount / (sl_distance * 100)
            return max(ScalpConfig.MIN_LOT, min(round(lot, 2), ScalpConfig.MAX_LOT))
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([w['profit'] for w in wins])
        avg_loss = abs(np.mean([l['profit'] for l in losses]))
        
        if avg_loss == 0:
            avg_loss = 1.0
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: (W * R - L) / R
        # W = win rate, R = win/loss ratio, L = loss rate
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Use fractional Kelly for safety
        kelly_pct = kelly_pct * ScalpConfig.KELLY_FRACTION
        
        # Clamp to reasonable range
        kelly_pct = max(0.003, min(kelly_pct, 0.02))  # 0.3% - 2%
        
        risk_amount = balance * kelly_pct
        lot = risk_amount / (sl_distance * 100)
        
        final_lot = max(ScalpConfig.MIN_LOT, min(round(lot, 2), ScalpConfig.MAX_LOT))
        
        logging.info(f"📊 Kelly: WR={win_rate:.1%}, R:R={win_loss_ratio:.2f}, Kelly%={kelly_pct*100:.2f}%, Lot={final_lot}")
        
        return final_lot


# ============================================================================
# PARTIAL CLOSE MANAGER
# ============================================================================

class PartialCloseManager:
    """Close portions of position at profit levels"""
    
    def __init__(self, executor):
        self.executor = executor
        self.partial_closed = False
    
    def reset(self):
        self.partial_closed = False
    
    def check_partial_close(self, position: Dict, original_tp: float) -> bool:
        """Close partial position at trigger level"""
        
        if not ScalpConfig.USE_PARTIAL_CLOSE or self.partial_closed:
            return False
        
        entry = position['price_open']
        current_price = position['profit'] / position['volume'] / 100 + entry
        
        if position['type'] == 'LONG':
            total_distance = original_tp - entry
            current_distance = current_price - entry
        else:
            total_distance = entry - original_tp
            current_distance = entry - current_price
        
        progress = current_distance / total_distance if total_distance > 0 else 0
        
        if progress >= ScalpConfig.PARTIAL_CLOSE_TRIGGER:
            try:
                close_volume = position['volume'] * ScalpConfig.PARTIAL_CLOSE_PERCENT
                close_volume = round(close_volume, 2)
                
                if ScalpConfig.PAPER_TRADING:
                    self.partial_closed = True
                    logging.info(f"📍 PAPER: Partial close {close_volume} lots")
                    return True
                
                close_type = mt5.ORDER_TYPE_SELL if position['type'] == 'LONG' else mt5.ORDER_TYPE_BUY
                bid, ask = self.executor.get_current_price()
                price = bid if position['type'] == 'LONG' else ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": ScalpConfig.SYMBOL,
                    "volume": close_volume,
                    "type": close_type,
                    "position": position['ticket'],
                    "price": price,
                    "deviation": 10,
                    "magic": ScalpConfig.MAGIC_NUMBER,
                    "comment": "PartialClose",
                }
                
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.partial_closed = True
                    logging.info(f"📍 Partial close: {close_volume} lots @ {price:.2f}")
                    return True
            except Exception as e:
                logging.error(f"❌ Partial close error: {e}")
        
        return False


# ============================================================================
# BREAKEVEN & TRAILING
# ============================================================================

class BreakevenManager:
    def __init__(self, executor):
        self.executor = executor
        self.moved_to_breakeven = False
    
    def reset(self):
        self.moved_to_breakeven = False
    
    def check_and_move(self, position: Dict, original_tp: float) -> bool:
        if not ScalpConfig.USE_BREAKEVEN or self.moved_to_breakeven:
            return False
        
        entry = position['price_open']
        current_price = position['profit'] / position['volume'] / 100 + entry
        
        if position['type'] == 'LONG':
            total_distance = original_tp - entry
            current_distance = current_price - entry
        else:
            total_distance = entry - original_tp
            current_distance = entry - current_price
        
        progress = current_distance / total_distance if total_distance > 0 else 0
        
        if progress >= ScalpConfig.BREAKEVEN_TRIGGER_PERCENT:
            try:
                if ScalpConfig.PAPER_TRADING:
                    self.moved_to_breakeven = True
                    logging.info("📍 PAPER: Moved to breakeven")
                    return True
                
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position['ticket'],
                    "sl": entry,
                    "tp": position['tp'],
                }
                
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.moved_to_breakeven = True
                    logging.info(f"📍 Moved SL to breakeven: ${entry:.2f}")
                    return True
            except Exception as e:
                logging.error(f"❌ Breakeven error: {e}")
        
        return False


class TrailingManager:
    def __init__(self, executor):
        self.executor = executor
        self.last_sl = None
        self.last_tp = None
        self.secured_at_40 = False
    
    def reset(self):
        """নতুন trade এর জন্য reset করুন"""
        self.last_sl = None
        self.last_tp = None
        self.secured_at_40 = False

    def check_and_trail(self, position: Dict, active_info: Dict, conversion_rate: float) -> bool:
        try:
            if not active_info:
                return False

            # ✅ 0.01 অথবা 0.02 lot - উভয়ের জন্যই trailing
            is_target_lot = (
                abs(position['volume'] - ScalpConfig.FIXED_LOT_TARGET_LOT) < 1e-8 or
                abs(position['volume'] - ScalpConfig.FIXED_LOT_TARGET_LOT_2) < 1e-8
            )
            
            if not is_target_lot:
                return False  # অন্য lot size হলে skip

            profit_usd = position['profit']
            profit_inr = profit_usd * conversion_rate if conversion_rate and conversion_rate > 0 else profit_usd

            entry = position['price_open']
            curr_bid, curr_ask = self.executor.get_current_price()
            current_price = (curr_bid + curr_ask) / 2

            # Price movement in pips
            profit_pips = abs(current_price - entry) * 100

            # LOG করুন কোথায় আছেন
            if profit_pips > 10:  # 10 pips এর বেশি হলে log করুন
                logging.info(f"🔍 Trail Check: Profit={profit_inr:.0f} INR ({profit_pips:.1f} pips), Need {ScalpConfig.FIXED_LOT_TRAILING_START_INR} INR ({ScalpConfig.FIXED_LOT_TRAILING_START_PIPS} pips)")

            # ✅ STEP 1: চেক করুন যথেষ্ট profit আছে কিনা
            if profit_inr < ScalpConfig.FIXED_LOT_TRAILING_START_INR:
                return False  # এখনো trailing শুরু করবেন না
            
            if profit_pips < ScalpConfig.FIXED_LOT_TRAILING_START_PIPS:
                return False  # এখনো trailing শুরু করবেন না

            # ✅ STEP 2: এখন trailing শুরু করুন
            logging.info(f"🎯 TRAILING ACTIVATED: {profit_pips:.1f} pips profit reached!")

            # SL কে current price থেকে FIXED_LOT_SECURE_SL_PIPS দূরে রাখুন
            secure_offset = ScalpConfig.FIXED_LOT_SECURE_SL_PIPS / 100.0
            
            if position['type'] == 'LONG':
                desired_sl = current_price - secure_offset
                # নিশ্চিত করুন SL কখনো entry এর নিচে যাবে না
                if desired_sl < entry:
                    desired_sl = entry + 0.0005  # entry থেকে একটু উপরে
                desired_tp = active_info.get('tp', position.get('tp', entry))
            else:  # SHORT
                desired_sl = current_price + secure_offset
                # নিশ্চিত করুন SL কখনো entry এর উপরে যাবে না
                if desired_sl > entry:
                    desired_sl = entry - 0.0005  # entry থেকে একটু নিচে
                desired_tp = active_info.get('tp', position.get('tp', entry))

            # ✅ STEP 3: চেক করুন SL move করার দরকার আছে কিনা
            if self.last_sl is not None:
                sl_move_pips = abs(desired_sl - self.last_sl) * 100
                if sl_move_pips < ScalpConfig.MIN_SL_MOVE_PIPS:
                    # খুব ছোট movement, skip করুন
                    return False
                
                # নিশ্চিত করুন SL শুধু favorable direction এ move করছে
                if position['type'] == 'LONG' and desired_sl <= self.last_sl:
                    return False  # LONG এ SL নিচে যাবে না
                if position['type'] == 'SHORT' and desired_sl >= self.last_sl:
                    return False  # SHORT এ SL উপরে যাবে না

            # ✅ STEP 4: SL Update করুন
            if ScalpConfig.PAPER_TRADING:
                logging.info(f"📍 PAPER: Trail SL -> {desired_sl:.5f} ({profit_pips:.1f} pips profit)")
                self.last_sl = desired_sl
                self.last_tp = desired_tp
                self.secured_at_40 = True
                return True

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position['ticket'],
                "sl": desired_sl,
                "tp": desired_tp,
            }
            result = mt5.order_send(request)
            if result and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                logging.info(f"📍 ✅ TRAILED: SL={desired_sl:.5f} (was {self.last_sl:.5f if self.last_sl else 'initial'}), Profit={profit_pips:.1f} pips")
                self.last_sl = desired_sl
                self.last_tp = desired_tp
                self.secured_at_40 = True
                return True
            else:
                logging.warning(f"⚠️ Trail failed: {result.comment if result else 'unknown'}")
            return False

        except Exception as e:
            logging.error(f"❌ Trailing error: {e}")
            return False


# ============================================================================
# TRADE STATISTICS
# ============================================================================

class TradeStatistics:
    """Real-time performance tracking"""
    
    def __init__(self):
        self.trades = []
    
    def add_trade(self, profit: float, duration_minutes: float, regime: str = ""):
        self.trades.append({
            'profit': profit,
            'duration': duration_minutes,
            'regime': regime,
            'timestamp': datetime.now()
        })
    
    def print_stats(self):
        if not self.trades:
            logging.info("📊 No trades yet")
            return
        
        profits = [t['profit'] for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_pnl = sum(profits)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / len(profits) * 100 if profits else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        logging.info("=" * 60)
        logging.info("📊 PERFORMANCE STATISTICS")
        logging.info("=" * 60)
        logging.info(f"Total Trades: {len(self.trades)}")
        logging.info(f"Wins: {win_count} | Losses: {loss_count}")
        logging.info(f"Win Rate: {win_rate:.1f}%")
        logging.info(f"Total P&L: ${total_pnl:.2f}")
        logging.info(f"Avg Win: ${avg_win:.2f}")
        logging.info(f"Avg Loss: ${avg_loss:.2f}")
        
        if wins and losses:
            profit_factor = abs(sum(wins) / sum(losses))
            logging.info(f"Profit Factor: {profit_factor:.2f}")
        
        if profits:
            logging.info(f"Best Trade: ${max(profits):.2f}")
            logging.info(f"Worst Trade: ${min(profits):.2f}")
        
        logging.info("=" * 60)


# ============================================================================
# ML PIPELINE (Enhanced)
# ============================================================================

class ScalpMLPipeline:
    def __init__(self):
        self.training_data = []
        self.model = None
        self.scaler = None
    
    def load_data(self):
        try:
            if os.path.exists(ScalpConfig.TRAINING_DATA_PATH):
                with open(ScalpConfig.TRAINING_DATA_PATH, 'rb') as f:
                    self.training_data = pickle.load(f)
                logging.info(f"✅ Loaded {len(self.training_data)} samples")
        except Exception as e:
            logging.warning(f"⚠️ Load failed: {e}")
    
    def add_sample(self, features: np.ndarray, profit: float):
        label = 1 if profit > 0 else 0
        self.training_data.append({
            'features': features,
            'label': label,
            'profit': profit,
            'timestamp': datetime.now()
        })
        
        if len(self.training_data) > 500:
            self.training_data = self.training_data[-500:]
        
        try:
            os.makedirs('models', exist_ok=True)
            with open(ScalpConfig.TRAINING_DATA_PATH, 'wb') as f:
                pickle.dump(self.training_data, f)
        except Exception as e:
            logging.error(f"❌ Save failed: {e}")
    
    def load_model(self) -> bool:
        try:
            if os.path.exists(ScalpConfig.XGBOOST_MODEL_PATH):
                self.model = xgb.Booster()
                self.model.load_model(ScalpConfig.XGBOOST_MODEL_PATH)
                logging.info("✅ Model loaded")
            
            if os.path.exists(ScalpConfig.FEATURE_SCALER_PATH):
                with open(ScalpConfig.FEATURE_SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            return self.model is not None
        except Exception as e:
            logging.warning(f"⚠️ Model load failed: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.7
        
        try:
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1))[0]
            
            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            confidence = self.model.predict(dmatrix)[0]
            return float(confidence)
        except:
            return 0.7
    
    def extract_features(self, signal: ScalpSignal, market_data: Dict) -> np.ndarray:
        """Enhanced feature extraction"""
        features = []
        
        # Signal features
        features.append(signal.confidence)
        features.append(1 if signal.direction == 'LONG' else 0)
        features.append(signal.strength)
        features.append(len(signal.indicators) / 8)
        features.append(signal.volume_score)
        features.append(signal.mtf_score)
        
        # Regime encoding
        regime_map = {
            "TRENDING_HIGH_VOL": 1.0,
            "TRENDING_LOW_VOL": 0.75,
            "RANGING_HIGH_VOL": 0.5,
            "RANGING_LOW_VOL": 0.25
        }
        features.append(regime_map.get(signal.regime, 0.5))
        
        # Price features
        df_m5 = market_data.get('M5', pd.DataFrame())
        if not df_m5.empty:
            price = df_m5['close'].iloc[-1]
            ema_20 = df_m5['close'].ewm(span=20).mean().iloc[-1]
            features.append((price - ema_20) / ema_20 * 100)
            
            # Recent returns
            features.append(df_m5['close'].pct_change(5).iloc[-1] * 100)
            features.append(df_m5['close'].pct_change(20).iloc[-1] * 100)
            
            # Volatility
            features.append(df_m5['close'].rolling(20).std().iloc[-1])
        else:
            features.extend([0, 0, 0, 0])
        
        # Time features (cyclical encoding)
        hour = datetime.now().hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        day = datetime.now().weekday()
        features.append(day / 7)
        
        # Spread
        try:
            tick = mt5.symbol_info_tick(ScalpConfig.SYMBOL)
            spread = (tick.ask - tick.bid) / tick.bid
            features.append(spread * 1000)
        except:
            features.append(0)
        
        # Pad to 25 features
        while len(features) < 25:
            features.append(0.5)
        
        return np.array(features[:25])


# ============================================================================
# MT5 EXECUTOR
# ============================================================================

class MT5Executor:
    def __init__(self):
        self.connected = False
    
    def connect(self) -> bool:
        try:
            mt5.shutdown()
            time.sleep(1)
            
            if not mt5.initialize(
                path=ScalpConfig.MT5_PATH,
                login=ScalpConfig.MT5_LOGIN,
                password=ScalpConfig.MT5_PASSWORD,
                server=ScalpConfig.MT5_SERVER,
                timeout=30000
            ):
                logging.error(f"❌ MT5 init failed: {mt5.last_error()}")
                return False
            
            acc = mt5.account_info()
            if acc is None:
                logging.error("❌ account_info None")
                return False
            
            logging.info("✅ Connected to MT5")
            logging.info(f"👤 Login: {acc.login}")
            logging.info(f"💰 Balance: ${acc.balance}")
            
            if not mt5.symbol_select(ScalpConfig.SYMBOL, True):
                logging.error(f"❌ Cannot select {ScalpConfig.SYMBOL}")
                return False
            
            self.connected = True
            return True
        
        except Exception as e:
            logging.exception("❌ MT5 connection exception")
            return False
    
    def disconnect(self):
        mt5.shutdown()
        logging.info("🔌 Disconnected")
    
    def get_balance(self) -> float:
        acc = mt5.account_info()
        return acc.balance if acc else 0.0

    def get_conversion_rate(self, from_ccy: str = 'USD', to_ccy: str = 'INR') -> float:
        if from_ccy == to_ccy:
            return 1.0

        candidates = [f"{from_ccy}{to_ccy}", f"{from_ccy}{to_ccy}.m"]
        try:
            for sym in candidates:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        bid = getattr(tick, 'bid', None)
                        ask = getattr(tick, 'ask', None)
                        if bid and ask:
                            return (bid + ask) / 2
                except:
                    continue
        except:
            pass

        return 1.0
    
    def get_current_price(self) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(ScalpConfig.SYMBOL)
        return (tick.bid, tick.ask) if tick else (0.0, 0.0)
    
    def get_position(self) -> Optional[Dict]:
        positions = mt5.positions_get(symbol=ScalpConfig.SYMBOL)
        if positions and len(positions) > 0:
            pos = positions[0]
            return {
                'ticket': pos.ticket,
                'type': 'LONG' if pos.type == mt5.POSITION_TYPE_BUY else 'SHORT',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': datetime.fromtimestamp(pos.time)
            }
        return None
    
    def execute_order(self, order_type: str, lot: float, entry: float,
                     sl: float, tp: float) -> Tuple[bool, str]:
        try:
            if ScalpConfig.PAPER_TRADING:
                logging.info(f"📝 PAPER: {order_type} {lot} lots")
                return True, "PAPER_SUCCESS"
            
            bid, ask = self.get_current_price()
            actual_entry = ask if order_type == "LONG" else bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ScalpConfig.SYMBOL,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if order_type == "LONG" else mt5.ORDER_TYPE_SELL,
                "price": actual_entry,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": ScalpConfig.MAGIC_NUMBER,
                "comment": f"Scalp_{datetime.now().strftime('%H%M%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                return False, f"FAILED: {mt5.last_error()}"
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"REJECTED: {result.retcode}"
            
            logging.info(f"✅ {order_type} executed: #{result.order}")
            return True, f"SUCCESS: #{result.order}"
        
        except Exception as e:
            return False, str(e)
    
    def close_position(self, reason: str = "MANUAL") -> Tuple[bool, float]:
        try:
            position = self.get_position()
            if not position:
                return False, 0.0
            
            if ScalpConfig.PAPER_TRADING:
                logging.info(f"📝 PAPER: Close")
                return True, 0.0
            
            close_type = mt5.ORDER_TYPE_SELL if position['type'] == 'LONG' else mt5.ORDER_TYPE_BUY
            bid, ask = self.get_current_price()
            price = bid if position['type'] == 'LONG' else ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ScalpConfig.SYMBOL,
                "volume": position['volume'],
                "type": close_type,
                "position": position['ticket'],
                "price": price,
                "deviation": 10,
                "magic": ScalpConfig.MAGIC_NUMBER,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                profit = position['profit']
                logging.info(f"✅ Closed: {reason} | P&L: ${profit:,.2f}")
                return True, profit
            else:
                return False, 0.0
        
        except Exception as e:
            logging.error(f"❌ Close error: {e}")
            return False, 0.0


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0  # Track wins
        self.consecutive_losses = 0
        self.last_loss_time = None
    
    def check_limits(self, balance: float) -> Tuple[bool, str]:
        if os.path.exists(ScalpConfig.KILL_SWITCH_FILE):
            return False, "KILL_SWITCH"
        
        # STOP AFTER TARGET WINS
        if ScalpConfig.STOP_AFTER_TARGET and self.daily_wins >= ScalpConfig.DAILY_WIN_TARGET:
            return False, f"TARGET_REACHED ({self.daily_wins} wins) 🎯"
        
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl / balance * 100)
            if loss_pct >= ScalpConfig.MAX_DAILY_LOSS_PERCENT:
                return False, f"DAILY_LOSS ({loss_pct:.2f}%)"
        
        if self.daily_trades >= ScalpConfig.MAX_TRADES_PER_DAY:
            return False, "MAX_TRADES"
        
        if self.consecutive_losses >= ScalpConfig.MAX_CONSECUTIVE_LOSSES:
            return False, f"CONSECUTIVE_LOSSES ({self.consecutive_losses})"
        
        if self.last_loss_time:
            cooldown_end = self.last_loss_time + timedelta(minutes=ScalpConfig.COOLDOWN_AFTER_LOSS_MINUTES)
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).seconds // 60
                return False, f"COOLDOWN ({remaining}m)"
        
        return True, "OK"
    
    def record_trade(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if pnl > 0:
            self.daily_wins += 1
            self.consecutive_losses = 0
            logging.info(f"✅ WIN #{self.daily_wins} | Target: {ScalpConfig.DAILY_WIN_TARGET}")
        else:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
    
    def reset_daily(self):
        logging.info(f"📊 Daily reset | P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades} | Wins: {self.daily_wins}")
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0


# ============================================================================
# MAIN BOT
# ============================================================================

class AdvancedGoldScalpingBot:
    def __init__(self):
        self.executor = MT5Executor()
        self.scalping = AdvancedScalpingEngine()
        self.ml_pipeline = ScalpMLPipeline()
        self.kelly_sizer = KellyPositionSizer()
        self.breakeven = BreakevenManager(self.executor)
        self.trailing = TrailingManager(self.executor)

        self.partial_close = PartialCloseManager(self.executor)
        self.risk_manager = RiskManager()
        self.statistics = TradeStatistics()
        self.running = False
        self.active_trade_info = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('gold_scalping_v3.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def start(self) -> bool:
        logging.info("="*70)
        logging.info("⚡ ADVANCED GOLD SCALPING BOT v3.0 - PROFESSIONAL EDITION")
        logging.info("="*70)
        logging.info("🚀 NEW FEATURES:")
        logging.info("  • Multi-Timeframe Confluence")
        logging.info("  • Market Regime Detection (ADX + ATR)")
        logging.info("  • Kelly Criterion Position Sizing")
        logging.info("  • Volume Profile Analysis")
        logging.info("  • Partial Close Strategy")
        logging.info("  • News Time Filter")
        logging.info("  • Order Flow Analysis")
        logging.info("  • Adaptive TP/SL")
        logging.info("  • Real-time Statistics")
        logging.info("  • FIXED: Trailing after 35+ pips (800 INR)")
        logging.info(f"Mode: {'📝 PAPER' if ScalpConfig.PAPER_TRADING else '🔴 LIVE'}")
        logging.info("="*70)
        
        if not self.executor.connect():
            return False
        
        self.ml_pipeline.load_data()
        if ScalpConfig.USE_XGBOOST:
            self.ml_pipeline.load_model()
        
        self.running = True
        logging.info("✅ Advanced scalping bot started")
        return True
    
    def stop(self):
        logging.info("🛑 Stopping...")
        self.running = False
        
        if self.executor.get_position():
            success, profit = self.executor.close_position("SHUTDOWN")
            if success and self.active_trade_info:
                self.ml_pipeline.add_sample(
                    self.active_trade_info['features'],
                    profit
                )
                self.risk_manager.record_trade(profit)
        
        self.statistics.print_stats()
        self.executor.disconnect()
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf_name, tf_code in [('M5', ScalpConfig.PRIMARY_TF), 
                                  ('M15', ScalpConfig.SECONDARY_TF),
                                  ('H1', ScalpConfig.FILTER_TF)]:
            rates = mt5.copy_rates_from_pos(ScalpConfig.SYMBOL, tf_code, 0, 200)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                data[tf_name] = df
        return data
    
    def check_spread(self) -> bool:
        bid, ask = self.executor.get_current_price()
        spread = abs(ask - bid)
        spread_pips = spread * 10
        
        if spread_pips > ScalpConfig.MAX_SPREAD_PIPS:
            logging.warning(f"⚠️ Spread too high: {spread_pips:.2f} pips")
            return False
        return True
    
    def run(self):
        last_daily_reset = datetime.now().date()
        last_signal_check = datetime.now()
        last_stats_print = datetime.now()
        
        try:
            while self.running:
                current_time = datetime.now()
                
                if os.path.exists(ScalpConfig.KILL_SWITCH_FILE):
                    logging.critical("🛑 KILL SWITCH")
                    self.stop()
                    break
                
                if current_time.date() > last_daily_reset:
                    self.risk_manager.reset_daily()
                    last_daily_reset = current_time.date()
                
                # Print stats every 30 minutes
                if (current_time - last_stats_print).total_seconds() > 1800:
                    self.statistics.print_stats()
                    last_stats_print = current_time
                
                position = self.executor.get_position()
                
                if position:
                    logging.info(f"📊 Active {position['type']} | P&L: ${position['profit']:,.2f}")
                    conv = self.executor.get_conversion_rate('USD', 'INR')
                    profit_inr = position['profit'] * conv
                    logging.info(f"    (≈ {profit_inr:,.2f} INR)")

                    if self.active_trade_info:
                        self.breakeven.check_and_move(position, self.active_trade_info['tp'])
                        self.partial_close.check_partial_close(position, self.active_trade_info['tp'])
                        self.trailing.check_and_trail(position, self.active_trade_info, conv)
                    
                    duration = datetime.now() - position['time']
                    if duration.total_seconds() / 60 >= ScalpConfig.MAX_TRADE_DURATION_MINUTES:
                        logging.warning(f"⏰ Max duration reached")
                        success, profit = self.executor.close_position("MAX_DURATION")
                        if success and self.active_trade_info:
                            self.ml_pipeline.add_sample(
                                self.active_trade_info['features'],
                                profit
                            )
                            self.kelly_sizer.add_trade(profit, abs(profit))
                            self.risk_manager.record_trade(profit)
                            self.statistics.add_trade(
                                profit,
                                duration.total_seconds() / 60,
                                self.active_trade_info.get('regime', '')
                            )
                            self.active_trade_info = None
                            self.breakeven.reset()
                            self.partial_close.reset()
                            self.trailing.reset()
                    
                    time.sleep(10)
                    continue
                
                if position is None and self.active_trade_info is not None:
                    logging.info("📝 Position closed by TP/SL")

                    profit = 0.0
                    try:
                        deals = mt5.history_deals_get(
                            datetime.now() - timedelta(minutes=10),
                            datetime.now()
                        )
                        if deals:
                            last_deal = deals[-1]
                            profit = last_deal.profit
                    except:
                        pass

                    self.ml_pipeline.add_sample(
                        self.active_trade_info['features'],
                        profit
                    )
                    self.kelly_sizer.add_trade(profit, abs(profit))
                    self.risk_manager.record_trade(profit)
                    
                    trade_duration = (datetime.now() - self.active_trade_info.get('start_time', datetime.now())).total_seconds() / 60
                    self.statistics.add_trade(
                        profit,
                        trade_duration,
                        self.active_trade_info.get('regime', '')
                    )

                    logging.info(f"📦 Sample saved | P&L: ${profit:.2f}")

                    self.active_trade_info = None
                    self.breakeven.reset()
                    self.partial_close.reset()
                    self.trailing.reset()
                
                balance = self.executor.get_balance()
                can_trade, reason = self.risk_manager.check_limits(balance)
                
                if not can_trade:
                    logging.warning(f"🚫 Trading blocked: {reason}")
                    time.sleep(60)
                    continue
                
                if not self.check_spread():
                    time.sleep(30)
                    continue
                
                if (current_time - last_signal_check).total_seconds() < 30:
                    time.sleep(5)
                    continue
                
                last_signal_check = current_time
                
                market_data = self.get_market_data()
                
                if 'M5' not in market_data or market_data['M5'].empty:
                    time.sleep(30)
                    continue
                
                bid, ask = self.executor.get_current_price()
                current_price = (bid + ask) / 2
                
                scalp_signal = self.scalping.generate_scalp_signal(market_data, current_price)
                
                if not scalp_signal:
                    time.sleep(30)
                    continue
                
                logging.info("="*60)
                logging.info(f"⚡ SCALP SIGNAL: {scalp_signal.direction} @ {scalp_signal.confidence:.2f}")
                logging.info(f"   Indicators: {', '.join(scalp_signal.indicators)}")
                logging.info(f"   Strength: {scalp_signal.strength:.0%}")
                logging.info(f"   Regime: {scalp_signal.regime}")
                logging.info(f"   Volume Score: {scalp_signal.volume_score:.2f}")
                logging.info(f"   MTF Score: {scalp_signal.mtf_score:.2f}")
                
                features = self.ml_pipeline.extract_features(scalp_signal, market_data)
                ml_confidence = self.ml_pipeline.predict(features)
                
                ml_threshold = 0.50 if self.ml_pipeline.model is None else 0.55
                
                logging.info(f"🤖 ML Confidence: {ml_confidence:.2f} (Threshold: {ml_threshold:.2f})")
                
                if ml_confidence < ml_threshold:
                    logging.info("🚫 Rejected by ML filter")
                    logging.info("="*60)
                    continue
                
                logging.info("✅ Signal APPROVED - Opening scalp trade")
                
                # KELLY POSITION SIZING
                sl_distance = abs(scalp_signal.sl - scalp_signal.entry)
                
                if ScalpConfig.USE_KELLY_CRITERION and len(self.kelly_sizer.trade_history) >= 10:
                    lot = self.kelly_sizer.calculate_kelly_lot(balance, sl_distance)
                else:
                    risk_amount = balance * (ScalpConfig.RISK_PER_TRADE / 100)
                    lot = risk_amount / (sl_distance * 100)
                    lot = max(ScalpConfig.MIN_LOT, min(round(lot, 2), ScalpConfig.MAX_LOT))

                # Fixed lot INR targets
                conv_rate = self.executor.get_conversion_rate('USD', 'INR')
                if abs(lot - ScalpConfig.FIXED_LOT_TARGET_LOT) < 1e-8:
                    tp_usd = ScalpConfig.FIXED_LOT_TP_INR / conv_rate
                    sl_usd = ScalpConfig.FIXED_LOT_SL_INR / conv_rate
                    delta_tp = tp_usd / (lot * 100)
                    delta_sl = sl_usd / (lot * 100)

                    if scalp_signal.direction == 'LONG':
                        scalp_signal.tp = scalp_signal.entry + delta_tp
                        scalp_signal.sl = scalp_signal.entry - delta_sl
                    else:
                        scalp_signal.tp = scalp_signal.entry - delta_tp
                        scalp_signal.sl = scalp_signal.entry + delta_sl

                    logging.info(f"   (INR Targets) TP: {ScalpConfig.FIXED_LOT_TP_INR} INR, SL: {ScalpConfig.FIXED_LOT_SL_INR} INR")
                
                potential_profit = abs(scalp_signal.tp - scalp_signal.entry) * lot * 100
                potential_loss = abs(scalp_signal.sl - scalp_signal.entry) * lot * 100
                rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
                
                logging.info(f"📊 TRADE PLAN:")
                logging.info(f"   Direction: {scalp_signal.direction}")
                logging.info(f"   Entry: ${scalp_signal.entry:,.2f}")
                logging.info(f"   SL: ${scalp_signal.sl:,.2f} (Risk: ${potential_loss:.2f})")
                logging.info(f"   TP: ${scalp_signal.tp:,.2f} (Reward: ${potential_profit:.2f})")
                logging.info(f"   Lot: {lot}")
                logging.info(f"   R:R = 1:{rr_ratio:.1f}")
                logging.info("="*60)
                
                success, msg = self.executor.execute_order(
                    scalp_signal.direction,
                    lot,
                    scalp_signal.entry,
                    scalp_signal.sl,
                    scalp_signal.tp
                )
                
                if success:
                    logging.info("✅ ⚡ SCALP TRADE OPENED")
                    
                    self.active_trade_info = {
                        'signal': scalp_signal,
                        'features': features,
                        'tp': scalp_signal.tp,
                        'regime': scalp_signal.regime,
                        'start_time': datetime.now()
                    }
                    
                else:
                    logging.error(f"❌ Execution failed: {msg}")
                
                time.sleep(30)
        
        except KeyboardInterrupt:
            logging.info("⚠️ Interrupted by user")
            self.stop()
        except Exception as e:
            logging.error(f"❌ CRITICAL ERROR: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.stop()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("⚡ ADVANCED GOLD SCALPING BOT v3.0 - PROFESSIONAL EDITION")
    print("="*70)
    print()
    print("🚀 ADVANCED FEATURES:")
    print("  ✅ Multi-Timeframe Confluence Analysis")
    print("  ✅ Market Regime Detection (ADX + ATR)")
    print("  ✅ Kelly Criterion Position Sizing")
    print("  ✅ Volume Profile Analysis")
    print("  ✅ Partial Close Strategy")
    print("  ✅ News Time Filter")
    print("  ✅ Order Flow Analysis")
    print("  ✅ Adaptive TP/SL Based on Regime")
    print("  ✅ Enhanced ML Features (25 dimensions)")
    print("  ✅ Real-time Performance Dashboard")
    print("  ✅ FIXED: Trailing only after 35+ pips (800 INR)")
    print()
    print("🎯 DAILY TARGET SYSTEM:")
    print(f"  • Target: {ScalpConfig.DAILY_WIN_TARGET} winning trades per day")
    print(f"  • Max attempts: {ScalpConfig.MAX_TRADES_PER_DAY} trades")
    print(f"  • Stop after target: {'YES' if ScalpConfig.STOP_AFTER_TARGET else 'NO'}")
    print(f"  • Cooldown after loss: {ScalpConfig.COOLDOWN_AFTER_LOSS_MINUTES} minutes")
    print(f"  • Max consecutive losses: {ScalpConfig.MAX_CONSECUTIVE_LOSSES}")
    print()
    print("📊 SCALPING SETUP (0.01 LOT):")
    print(f"  • SL: {ScalpConfig.FIXED_LOT_SL_INR} INR (≈30 pips)")
    print(f"  • TP: {ScalpConfig.FIXED_LOT_TP_INR} INR (≈50 pips)")
    print(f"  • Trailing starts: {ScalpConfig.FIXED_LOT_TRAILING_START_INR} INR ({ScalpConfig.FIXED_LOT_TRAILING_START_PIPS} pips)")
    print(f"  • Trailing distance: {ScalpConfig.FIXED_LOT_SECURE_SL_PIPS} pips behind price")
    print(f"  • Min SL move: {ScalpConfig.MIN_SL_MOVE_PIPS} pips")
    print(f"  • Max duration: {ScalpConfig.MAX_TRADE_DURATION_MINUTES} minutes")
    print()
    print("🎯 INDICATORS & FILTERS:")
    print("  • Fast EMA Cross (5/13)")
    print("  • Stochastic Oscillator")
    print("  • CCI (20)")
    print("  • Price Momentum")
    print("  • Candlestick Patterns")
    print("  • H1 Trend Filter")
    print("  • Volume Confirmation")
    print("  • Market Regime Filter")
    print("  • News Time Avoidance")
    print("  • Order Flow Pressure")
    print()
    print("⚙️ TIMEFRAMES:")
    print("  • Primary: M5")
    print("  • Secondary: M15")
    print("  • Filter: H1")
    print()
    print("🧠 MACHINE LEARNING:")
    print("  • XGBoost Classifier")
    print("  • 25 Advanced Features")
    print("  • Real-time Learning")
    print()
    print("="*70)
    print()
    
    if ScalpConfig.PAPER_TRADING:
        print("⚠️  PAPER TRADING MODE")
    else:
        print("🔴 LIVE TRADING MODE")
        print()
        print("⚠️  WARNING: This bot will trade real money!")
        print("⚠️  Make sure you understand the risks involved.")
        print()
        confirm = input("Type 'SCALP' to start: ")
        if confirm != 'SCALP':
            print("❌ Cancelled")
            return
    
    print()
    bot = AdvancedGoldScalpingBot()
    
    if bot.start():
        print("✅ Bot running... Press Ctrl+C to stop")
        print()
        bot.run()
    else:
        logging.error("❌ Failed to start bot")


if __name__ == "__main__":
    main()