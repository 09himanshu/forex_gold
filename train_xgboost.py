"""
ADVANCED GOLD SCALPING SYSTEM v2.1 - OPTIMIZED & FIXED
✅ 50-100 pip TP (Scalping Mode)
✅ Tight SL (20-40 pip)
✅ M5/M15 Timeframes
✅ High Frequency Trading
✅ Quick In/Out Strategy
✅ Safe Risk Management
✅ Scalping Indicators (Stochastic, CCI, Fast EMAs)
✅ FIXED: All syntax errors resolved
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
# SCALPING CONFIGURATION
# ============================================================================

class ScalpConfig:
    """Scalping-specific configuration"""
    
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
    
    MAX_DAILY_LOSS_PERCENT = 1.5
    MAX_TRADES_PER_DAY = 65
    MAX_CONSECUTIVE_LOSSES = 5
    COOLDOWN_AFTER_LOSS_MINUTES = 15
    MAX_TRADE_DURATION_MINUTES = 60
    
    # TP/SL Settings
    SCALP_TP_MIN_PIPS = 50
    SCALP_TP_MAX_PIPS = 100
    SCALP_SL_PIPS = 30
    
    USE_DYNAMIC_TP = True
    USE_BREAKEVEN = True
    BREAKEVEN_TRIGGER_PERCENT = 0.5

    # INR specific targets for 0.01 lot (user request)
    FIXED_LOT_TARGET_LOT = 0.01
    FIXED_LOT_SL_INR = 600.0
    FIXED_LOT_TP_INR = 1200.0
    FIXED_LOT_TRAILING_START_INR = 400.0
    
    # Thresholds
    MIN_SCALP_CONFIDENCE = 0.30
    MIN_INDICATOR_ALIGNMENT = 0.30
    MAX_SPREAD_PIPS = 2.0
    
    # Timeframes
    PRIMARY_TF = mt5.TIMEFRAME_M5
    SECONDARY_TF = mt5.TIMEFRAME_M15
    FILTER_TF = mt5.TIMEFRAME_H1
    
    # Debug
    DEBUG_MODE = True
    SHOW_DETAILED_DEBUG = True
    
    # Indicator Weights
    INDICATOR_WEIGHTS = {
        'fast_ema_cross': 0.25,
        'stochastic': 0.20,
        'cci': 0.15,
        'momentum': 0.15,
        'price_action': 0.15,
        'trend_filter': 0.10
    }
    
    # ML
    XGBOOST_MODEL_PATH = 'models/gold_scalping_xgb.json'
    FEATURE_SCALER_PATH = 'models/gold_scalping_scaler.pkl'
    TRAINING_DATA_PATH = 'models/scalping_training_data.pkl'
    USE_XGBOOST = True
    XGBOOST_FALLBACK_MODE = True
    ML_MIN_TRAINING_SAMPLES = 30
    ML_AUTO_RETRAIN = False
    
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
        
        # Bullish engulfing
        if (last['close'] > last['open'] and 
            prev['close'] < prev['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close']):
            return "LONG", 0.75
        
        # Bearish engulfing
        elif (last['close'] < last['open'] and 
              prev['close'] > prev['open'] and
              last['close'] < prev['open'] and
              last['open'] > prev['close']):
            return "SHORT", 0.75
        
        # Strong bullish candle
        elif body > range_size * 0.7 and last['close'] > last['open']:
            return "LONG", 0.65
        
        # Strong bearish candle
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
# SCALPING ENGINE
# ============================================================================

class ScalpingEngine:
    def __init__(self):
        self.indicators = ScalpingIndicators()
    
    def generate_scalp_signal(self, market_data: Dict[str, pd.DataFrame],
                             current_price: float) -> Optional[ScalpSignal]:
        
        df_m5 = market_data.get('M5', pd.DataFrame())
        df_m15 = market_data.get('M15', pd.DataFrame())
        df_h1 = market_data.get('H1', pd.DataFrame())
        
        if df_m5.empty or df_m15.empty:
            return None
        
        signals = {}
        
        # Collect all indicator signals
        direction, conf = self.indicators.fast_ema_cross(df_m5)
        if direction:
            signals['fast_ema_cross'] = (direction, conf)
        
        direction, conf = self.indicators.stochastic(df_m5)
        if direction:
            signals['stochastic'] = (direction, conf)
        
        direction, conf = self.indicators.cci(df_m15)
        if direction:
            signals['cci'] = (direction, conf)
        
        direction, conf = self.indicators.momentum(df_m5)
        if direction:
            signals['momentum'] = (direction, conf)
        
        direction, conf = self.indicators.price_action(df_m5)
        if direction:
            signals['price_action'] = (direction, conf)
        
        if not df_h1.empty:
            direction, conf = self.indicators.trend_filter(df_h1)
            if direction:
                signals['trend_filter'] = (direction, conf)
        
        if not signals:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info("🔍 No indicator signals")
            return None
        
        # Vote
        long_score = 0.0
        short_score = 0.0
        long_indicators = []
        short_indicators = []
        
        for indicator, (direction, conf) in signals.items():
            weight = ScalpConfig.INDICATOR_WEIGHTS.get(indicator, 0.1)
            weighted_conf = conf * weight
            
            if direction == "LONG":
                long_score += weighted_conf
                long_indicators.append(indicator)
            elif direction == "SHORT":
                short_score += weighted_conf
                short_indicators.append(indicator)
        
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
                logging.info(f"🧮 Vote detail → LONG={long_score:.3f}, SHORT={short_score:.3f}, Signals={len(signals)}")
            return None
        
        alignment = len(active_indicators) / len(ScalpConfig.INDICATOR_WEIGHTS)
        if alignment < ScalpConfig.MIN_INDICATOR_ALIGNMENT:
            if ScalpConfig.SHOW_DETAILED_DEBUG:
                logging.info(f"⚠️ Low alignment: {alignment:.0%}")
            return None
        
        # Calculate TP/SL
        atr = self._calculate_atr(df_m5)
        
        if ScalpConfig.USE_DYNAMIC_TP:
            tp_pips = np.clip(
                atr * 100 * 0.8,   # ATR → pips
                ScalpConfig.SCALP_TP_MIN_PIPS,
                ScalpConfig.SCALP_TP_MAX_PIPS
            ) / 10
        else:
            tp_pips = (ScalpConfig.SCALP_TP_MIN_PIPS + ScalpConfig.SCALP_TP_MAX_PIPS) / 20
        
        sl_pips = ScalpConfig.SCALP_SL_PIPS / 10
        
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
            strength=alignment
        )
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        return pd.Series(tr).rolling(period).mean().iloc[-1]


# ============================================================================
# ML PIPELINE
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
        features = []
        
        features.append(signal.confidence)
        features.append(1 if signal.direction == 'LONG' else 0)
        features.append(signal.strength)
        features.append(len(signal.indicators) / 6)
        
        df_m5 = market_data.get('M5', pd.DataFrame())
        if not df_m5.empty:
            price = df_m5['close'].iloc[-1]
            ema_20 = df_m5['close'].ewm(span=20).mean().iloc[-1]
            features.append((price - ema_20) / ema_20 * 100)
        else:
            features.append(0)
        
        while len(features) < 20:
            features.append(0.5)
        
        return np.array(features[:20])


# ============================================================================
# BREAKEVEN MANAGER
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
    """Move SL/TP progressively once profit crosses a configured INR threshold."""
    def __init__(self, executor):
        self.executor = executor
        self.last_sl = None
        self.last_tp = None

    def check_and_trail(self, position: Dict, active_info: Dict, conversion_rate: float) -> bool:
        try:
            if not active_info:
                return False

            if position['volume'] != ScalpConfig.FIXED_LOT_TARGET_LOT:
                return False

            profit_usd = position['profit']
            profit_inr = profit_usd * conversion_rate if conversion_rate and conversion_rate > 0 else profit_usd

            if profit_inr < ScalpConfig.FIXED_LOT_TRAILING_START_INR:
                return False

            entry = position['price_open']
            curr_bid, curr_ask = self.executor.get_current_price()
            current_price = (curr_bid + curr_ask) / 2

            # For LONG positions
            if position['type'] == 'LONG':
                # move SL to halfway between entry and current price to lock profit
                new_sl = entry + (current_price - entry) * 0.5
                # extend TP by the gained distance so TP moves up too
                orig_tp = active_info.get('tp', position.get('tp', entry))
                new_tp = orig_tp + (current_price - entry)
            else:
                new_sl = entry - (entry - current_price) * 0.5
                orig_tp = active_info.get('tp', position.get('tp', entry))
                new_tp = orig_tp - (entry - current_price)

            # Avoid tiny updates
            eps = 1e-6
            if (self.last_sl is not None and abs(new_sl - self.last_sl) < eps) and (
                self.last_tp is not None and abs(new_tp - self.last_tp) < eps):
                return False

            # Send SL/TP update
            if ScalpConfig.PAPER_TRADING:
                logging.info(f"📝 PAPER: Trailing SL -> {new_sl:.5f}, TP -> {new_tp:.5f} | P&L INR: {profit_inr:.2f}")
                self.last_sl = new_sl
                self.last_tp = new_tp
                return True

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position['ticket'],
                "sl": new_sl,
                "tp": new_tp,
            }
            result = mt5.order_send(request)
            if result and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                logging.info(f"📍 Trailed SL to {new_sl:.5f} and TP to {new_tp:.5f} | P&L INR: {profit_inr:.2f}")
                self.last_sl = new_sl
                self.last_tp = new_tp
                return True
            else:
                logging.debug(f"Trailing update skipped or failed: {result}")
                return False
        except Exception as e:
            logging.error(f"❌ Trailing error: {e}")
            return False


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
        """Attempt to find a symbol providing the conversion rate (tries common variants).
        Returns mid price (ask+bid)/2 or 1.0 on failure.
        """
        if from_ccy == to_ccy:
            return 1.0

        candidates = [f"{from_ccy}{to_ccy}", f"{from_ccy}{to_ccy}.m", f"{from_ccy}{to_ccy}i", f"{from_ccy}/{to_ccy}"]
        try:
            # also search available symbols for substring match
            symbols = mt5.symbols_get()
            for sym in candidates:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        bid = getattr(tick, 'bid', None)
                        ask = getattr(tick, 'ask', None)
                        if bid and ask:
                            return (bid + ask) / 2
                except Exception:
                    continue

            # fallback: substring search across available symbols
            for s in symbols:
                name = getattr(s, 'name', '') or getattr(s, 'symbol', '')
                if from_ccy + to_ccy in name.replace(' ', '').upper():
                    tick = mt5.symbol_info_tick(name)
                    if tick:
                        bid = getattr(tick, 'bid', None)
                        ask = getattr(tick, 'ask', None)
                        if bid and ask:
                            return (bid + ask) / 2
        except Exception:
            pass

        logging.warning(f"⚠️ Conversion rate {from_ccy}->{to_ccy} not found, defaulting to 1.0")
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
    
    def calculate_lot_size(self, balance: float, entry: float, sl: float) -> float:
        try:
            risk_amount = balance * (ScalpConfig.RISK_PER_TRADE / 100)
            sl_distance = abs(entry - sl)
            lot_size = risk_amount / (sl_distance * 100)
            lot_size = round(lot_size, 2)
            return max(ScalpConfig.MIN_LOT, min(lot_size, ScalpConfig.MAX_LOT))
        except Exception as e:
            logging.error(f"❌ Lot calc error: {e}")
            return ScalpConfig.MIN_LOT
    
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
        self.consecutive_losses = 0
        self.last_loss_time = None
    
    def check_limits(self, balance: float) -> Tuple[bool, str]:
        if os.path.exists(ScalpConfig.KILL_SWITCH_FILE):
            return False, "KILL_SWITCH"
        
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
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self):
        logging.info(f"📊 Daily reset | P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades}")
        self.daily_pnl = 0.0
        self.daily_trades = 0


# ============================================================================
# MAIN BOT
# ============================================================================

class GoldScalpingBot:
    def __init__(self):
        self.executor = MT5Executor()
        self.scalping = ScalpingEngine()
        self.ml_pipeline = ScalpMLPipeline()
        self.breakeven = BreakevenManager(self.executor)
        self.trailing = TrailingManager(self.executor)
        self.risk_manager = RiskManager()
        self.running = False
        self.active_trade_info = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('gold_scalping.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def start(self) -> bool:
        logging.info("="*70)
        logging.info("⚡ ADVANCED GOLD SCALPING BOT v2.1")
        logging.info("="*70)
        logging.info(f"🎯 TP: {ScalpConfig.SCALP_TP_MIN_PIPS}-{ScalpConfig.SCALP_TP_MAX_PIPS} pips (${ScalpConfig.SCALP_TP_MIN_PIPS/10:.1f}-${ScalpConfig.SCALP_TP_MAX_PIPS/10:.1f})")
        logging.info(f"🛡️ SL: {ScalpConfig.SCALP_SL_PIPS} pips (${ScalpConfig.SCALP_SL_PIPS/10:.1f})")
        logging.info(f"📊 Timeframes: M5/M15")
        logging.info(f"⚡ Max Duration: {ScalpConfig.MAX_TRADE_DURATION_MINUTES} minutes")
        logging.info(f"🔥 Max Trades/Day: {ScalpConfig.MAX_TRADES_PER_DAY}")
        logging.info(f"Mode: {'📝 PAPER' if ScalpConfig.PAPER_TRADING else '🔴 LIVE'}")
        logging.info("="*70)
        
        if not self.executor.connect():
            return False
        
        self.ml_pipeline.load_data()
        if ScalpConfig.USE_XGBOOST:
            self.ml_pipeline.load_model()
        
        self.running = True
        logging.info("✅ Scalping bot started")
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
            logging.warning(f"⚠️ Spread too high: {spread_pips:.2f} pips > {ScalpConfig.MAX_SPREAD_PIPS}")
            return False
        return True
    
    def run(self):
        last_daily_reset = datetime.now().date()
        last_signal_check = datetime.now()
        
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
                
                position = self.executor.get_position()
                
                if position:
                    logging.info(f"📊 Active {position['type']} | P&L: ${position['profit']:,.2f}")
                    # convert USD profit to INR for display and logic
                    conv = self.executor.get_conversion_rate('USD', 'INR')
                    try:
                        profit_inr = position['profit'] * conv
                    except Exception:
                        profit_inr = position['profit']

                    logging.info(f"    (≈ {profit_inr:,.2f} INR)")

                    if self.active_trade_info:
                        self.breakeven.check_and_move(position, self.active_trade_info['tp'])
                        # trailing manager will move SL/TP when profit_inr threshold crossed
                        try:
                            self.trailing.check_and_trail(position, self.active_trade_info, conv)
                        except Exception:
                            pass
                    
                    duration = datetime.now() - position['time']
                    if duration.total_seconds() / 60 >= ScalpConfig.MAX_TRADE_DURATION_MINUTES:
                        logging.warning(f"⏰ Max duration reached ({ScalpConfig.MAX_TRADE_DURATION_MINUTES}m)")
                        success, profit = self.executor.close_position("MAX_DURATION")
                        if success and self.active_trade_info:
                            self.ml_pipeline.add_sample(
                                self.active_trade_info['features'],
                                profit
                            )
                            self.risk_manager.record_trade(profit)
                            self.active_trade_info = None
                            self.breakeven.reset()
                    
                    time.sleep(10)
                    continue
                
                if position is None and self.active_trade_info is not None:
                    logging.info("📝 Position closed by TP/SL")

                    # 👉 LIVE MODE: fetch last closed deal profit
                    profit = 0.0
                    try:
                        deals = mt5.history_deals_get(
                            datetime.now() - timedelta(minutes=10),
                            datetime.now()
                        )
                        if deals:
                            last_deal = deals[-1]
                            profit = last_deal.profit
                    except Exception as e:
                        logging.error(f"❌ Deal fetch error: {e}")

                    # ✅ SAVE SAMPLE (LIVE + PAPER both)
                    self.ml_pipeline.add_sample(
                        self.active_trade_info['features'],
                        profit
                    )

                    self.risk_manager.record_trade(profit)

                    logging.info(
                        f"📦 LIVE SAMPLE SAVED | Profit: ${profit:.2f} | Total samples: {len(self.ml_pipeline.training_data)}"
                    )

                    self.active_trade_info = None
                    self.breakeven.reset()
                
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
                
                logging.info(f"⚡ SCALP SIGNAL: {scalp_signal.direction} @ {scalp_signal.confidence:.2f}")
                logging.info(f"   Indicators: {', '.join(scalp_signal.indicators)}")
                logging.info(f"   Strength: {scalp_signal.strength:.0%}")
                
                features = self.ml_pipeline.extract_features(scalp_signal, market_data)
                ml_confidence = self.ml_pipeline.predict(features)
                
                ml_threshold = 0.45 if self.ml_pipeline.model is None else 0.55
                
                logging.info(f"🤖 ML Confidence: {ml_confidence:.2f} (Threshold: {ml_threshold:.2f})")
                
                if ml_confidence < ml_threshold:
                    logging.info("🚫 Rejected by ML filter")
                    continue
                
                logging.info("✅ Signal APPROVED - Opening scalp trade")
                
                lot = self.executor.calculate_lot_size(
                    balance,
                    scalp_signal.entry,
                    scalp_signal.sl
                )

                # If using fixed 0.01 lot target, set SL/TP to hit INR targets requested by user
                conv_rate = self.executor.get_conversion_rate('USD', 'INR')
                if abs(lot - ScalpConfig.FIXED_LOT_TARGET_LOT) < 1e-8:
                    # amount in USD needed = INR_target / conv_rate
                    tp_usd = ScalpConfig.FIXED_LOT_TP_INR / conv_rate
                    sl_usd = ScalpConfig.FIXED_LOT_SL_INR / conv_rate
                    # price delta for this symbol: delta_price = usd_amount / (lot * 100)
                    # note: for 0.01 lot, lot*100 == 1 → delta = usd_amount
                    delta_tp = tp_usd / (lot * 100)
                    delta_sl = sl_usd / (lot * 100)

                    if scalp_signal.direction == 'LONG':
                        scalp_signal.tp = scalp_signal.entry + delta_tp
                        scalp_signal.sl = scalp_signal.entry - delta_sl
                    else:
                        scalp_signal.tp = scalp_signal.entry - delta_tp
                        scalp_signal.sl = scalp_signal.entry + delta_sl

                    logging.info(f"   (INR Targets applied) TP delta USD: {tp_usd:.4f}, SL delta USD: {sl_usd:.4f} | Rate: {conv_rate:.4f}")
                
                risk_amount = balance * (ScalpConfig.RISK_PER_TRADE / 100)
                potential_profit = abs(scalp_signal.tp - scalp_signal.entry) * lot * 100
                potential_loss = abs(scalp_signal.sl - scalp_signal.entry) * lot * 100
                rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
                
                logging.info(f"📊 SCALP TRADE PLAN:")
                logging.info(f"   Direction: {scalp_signal.direction}")
                logging.info(f"   Entry: ${scalp_signal.entry:,.2f}")
                logging.info(f"   SL: ${scalp_signal.sl:,.2f} (${potential_loss:.2f} risk)")
                logging.info(f"   TP: ${scalp_signal.tp:,.2f} (${potential_profit:.2f} reward)")
                logging.info(f"   Lot: {lot}")
                logging.info(f"   R:R = 1:{rr_ratio:.1f}")
                logging.info(f"   Risk: ${risk_amount:.2f} ({ScalpConfig.RISK_PER_TRADE}%)")
                
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
                        'tp': scalp_signal.tp
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
    print("⚡ ADVANCED GOLD SCALPING BOT v2.1")
    print("="*70)
    print()
    print("🎯 SCALPING FEATURES:")
    print(f"  • TP: {ScalpConfig.SCALP_TP_MIN_PIPS}-{ScalpConfig.SCALP_TP_MAX_PIPS} pips (${ScalpConfig.SCALP_TP_MIN_PIPS/10:.1f}-${ScalpConfig.SCALP_TP_MAX_PIPS/10:.1f})")
    print(f"  • SL: {ScalpConfig.SCALP_SL_PIPS} pips (${ScalpConfig.SCALP_SL_PIPS/10:.1f})")
    print(f"  • Risk per trade: {ScalpConfig.RISK_PER_TRADE}%")
    print(f"  • Max lot: {ScalpConfig.MAX_LOT}")
    print(f"  • Max trades/day: {ScalpConfig.MAX_TRADES_PER_DAY}")
    print(f"  • Max duration: {ScalpConfig.MAX_TRADE_DURATION_MINUTES} minutes")
    print(f"  • Breakeven: {ScalpConfig.BREAKEVEN_TRIGGER_PERCENT:.0%} of TP")
    print()
    print("📊 INDICATORS:")
    print("  • Fast EMA Cross (5/13)")
    print("  • Stochastic Oscillator")
    print("  • CCI (Commodity Channel Index)")
    print("  • Price Momentum")
    print("  • Candlestick Patterns")
    print("  • H1 Trend Filter")
    print()
    print("⚙️ TIMEFRAMES:")
    print("  • Primary: M5")
    print("  • Secondary: M15")
    print("  • Filter: H1")
    print()
    print("="*70)
    print()
    
    if ScalpConfig.PAPER_TRADING:
        print("⚠️  PAPER TRADING MODE")
    else:
        print("🔴 LIVE TRADING MODE")
        print()
        confirm = input("Type 'SCALP' to start: ")
        if confirm != 'SCALP':
            print("❌ Cancelled")
            return
    
    print()
    bot = GoldScalpingBot()
    
    if bot.start():
        print("✅ Bot running... Press Ctrl+C to stop")
        print()
        bot.run()
    else:
        logging.error("❌ Failed to start bot")


if __name__ == "__main__":
    main()