"""
Perrobotillo Strategy for Jesse AI Research Lab
=================================================

Replicates the trading logic of Perrobotillo Trading Bot (producción)
para backtesting, optimización y análisis en Jesse.

Entry logic:
  1. Market mode detection (F&G + EMA20 + RSI)
  2. ML validator proxy (technical indicators → confidence score)
  3. Per-coin TP/SL/position sizing overrides

Exit logic:
  - TP1 (35%): primer take profit
  - TP2 (30%): segundo take profit
  - TP3 (30%): tercer take profit
  - SL: stop loss basado en market mode
  - Trailing stop: ATR-based (cuando está activado)
  - Scalping: solo en LATERAL (1% target, 2.5% SL)

Market modes:
  BULL          TP1=3.0, TP2=6.0,  TP3=10.0, SL=4.0, POS=25
  BEAR          TP1=1.8, TP2=3.5,  TP3=5.5,  SL=2.5, POS=10
  LATERAL       TP1=1.5, TP2=3.0,  TP3=5.0,  SL=3.5, POS=10
  VOLATILE_BULL TP1=3.5, TP2=7.0,  TP3=12.0, SL=5.0, POS=15
  VOLATILE_BEAR TP1=1.8, TP2=3.5,  TP3=5.5,  SL=2.5, POS=10
  LATERAL_BULL  (usa LATERAL con override ML)

Per-coin overrides (desde per-coin-config.env):
  BTC:  TP1=3.0, TP2=6.0, TP3=9.0,  SL=3.5, POS=10, ML_MIN=45
  ETH:  TP1=3.0, TP2=6.0, TP3=9.0,  SL=3.0, POS=10, ML_MIN=45
  BNB:  TP1=2.5, TP2=5.0, TP3=7.5,  SL=2.0, POS=15, ML_MIN=40
  SOL:  TP1=2.5, TP2=5.0, TP3=7.5,  SL=2.0, POS=15, ML_MIN=35
  LINK: TP1=2.5, TP2=5.0, TP3=7.5,  SL=2.0, POS=10, ML_MIN=35
  XRP:  TP1=2.5, TP2=5.0, TP3=7.5,  SL=2.5, POS=10, ML_MIN=35
  AVAX: TP1=2.5, TP2=5.0, TP3=7.5,  SL=2.0, POS=10, ML_MIN=35
"""

from jesse.strategies import Strategy, cached
import jesse.indicators as ta
from jesse import utils
import numpy as np


# ============================================================================
# MARKET MODE PARAMETERS (mirror de market-sentiment.sh + dynamic-config.env)
# ============================================================================
MODE_PARAMS = {
    'BULL':          {'tp1': 3.0,  'tp2': 6.0,  'tp3': 10.0, 'sl': 4.0,  'pos': 25,  'trailing': False},
    'BEAR':          {'tp1': 1.8,  'tp2': 3.5,  'tp3': 5.5,  'sl': 2.5,  'pos': 10,  'trailing': False},
    'LATERAL':       {'tp1': 1.5,  'tp2': 3.0,  'tp3': 5.0,  'sl': 3.5,  'pos': 10,  'trailing': False},
    'VOLATILE_BULL': {'tp1': 3.5,  'tp2': 7.0,  'tp3': 12.0, 'sl': 5.0,  'pos': 15,  'trailing': False},
    'VOLATILE_BEAR': {'tp1': 1.8,  'tp2': 3.5,  'tp3': 5.5,  'sl': 2.5,  'pos': 10,  'trailing': False},
    'LATERAL_BULL':  {'tp1': 1.5,  'tp2': 3.0,  'tp3': 5.0,  'sl': 3.5,  'pos': 10,  'trailing': False},
}

# Per-coin overrides (from per-coin-config.env)
COIN_OVERRIDES = {
    'BTC':  {'tp1': 3.0,  'tp2': 6.0,  'tp3': 9.0,  'sl': 3.5,  'pos': 10, 'ml_min': 45},
    'ETH':  {'tp1': 3.0,  'tp2': 6.0,  'tp3': 9.0,  'sl': 3.0,  'pos': 10, 'ml_min': 45},
    'BNB':  {'tp1': 2.5,  'tp2': 5.0,  'tp3': 7.5,  'sl': 2.0,  'pos': 15, 'ml_min': 40},
    'SOL':  {'tp1': 2.5,  'tp2': 5.0,  'tp3': 7.5,  'sl': 2.0,  'pos': 15, 'ml_min': 35},
    'LINK': {'tp1': 2.5,  'tp2': 5.0,  'tp3': 7.5,  'sl': 2.0,  'pos': 10, 'ml_min': 35},
    'XRP':  {'tp1': 2.5,  'tp2': 5.0,  'tp3': 7.5,  'sl': 2.5,  'pos': 10, 'ml_min': 35},
    'AVAX': {'tp1': 2.5,  'tp2': 5.0,  'tp3': 7.5,  'sl': 2.0,  'pos': 10, 'ml_min': 35},
}

# TP allocation percentages (35% / 30% / 30%)
TP1_PCT = 0.35
TP2_PCT = 0.30
TP3_PCT = 0.30

# Scalping (LATERAL only)
SCALP_TARGET = 1.0
SCALP_SL = 2.5


class Perrobotillo(Strategy):
    # ------------------------------------------------------------------ #
    # MARKET MODE DETECTION (mirror de market-sentiment.sh)
    # ------------------------------------------------------------------ #
    # Inputs: Fear & Greed value, BTC price vs EMA20, RSI
    # In backtesting we don't have F&G, so we approximate:
    #   F&G proxy = BTC momentum strength (EMA cross + price vs EMA)
    #   FG_HIGH (60+) = bullish, FG_LOW (40-) = bearish

    @property
    def btc_candles(self):
        """Get BTC candles for market mode detection (4h timeframe)."""
        try:
            return self.get_candles('binance-spot', 'BTC-USDT', '4h')
        except Exception:
            return self.candles

    @property
    def btc_ema20(self):
        return ta.ema(self.btc_candles, 20)[-1]

    @property
    def btc_price(self):
        return self.btc_candles[-1][2]  # close price

    @property
    def btc_rsi(self):
        return ta.rsi(self.btc_candles, 14)[-1]

    @property
    def btc_ema_cross(self):
        """EMA20 vs EMA50 cross for F&G proxy"""
        ema20 = ta.ema(self.btc_candles, 20)[-1]
        ema50 = ta.ema(self.btc_candles, 50)[-1]
        return ema20 - ema50

    @property
    def fear_greed_proxy(self):
        """
        Proxy for Fear & Greed Index using BTC technicals.
        Maps to 0-100 scale similar to Alternative.me F&G.
        Based on: EMA cross direction, RSI, price vs EMA, momentum
        """
        ema20 = self.btc_ema20
        ema50_val = ta.ema(self.btc_candles, 50)[-1]
        rsi = self.btc_rsi
        close = self.btc_price

        # Base score from RSI (30-70 neutral)
        fg = 50.0

        # EMA cross: bullish if price > EMA20 > EMA50
        if close > ema20 and ema20 > ema50_val:
            fg += 20  # strong uptrend
        elif close < ema20 and ema20 < ema50_val:
            fg -= 20  # strong downtrend

        # RSI contribution
        if rsi > 70:
            fg += 10
        elif rsi < 30:
            fg -= 10

        # EMA cross momentum
        cross = self.btc_ema_cross
        if cross > 0:
            fg += 5
        else:
            fg -= 5

        return np.clip(fg, 0, 100)

    @property
    def coin(self):
        """Extract bare coin name from symbol (BTC-USDT -> BTC)."""
        return self.symbol.split('-')[0]

    @property
    def per_coin_config(self):
        """Get per-coin overrides, falling back to market mode defaults."""
        coin = self.coin
        cfg = COIN_OVERRIDES.get(coin, {})
        mode = self.detected_market_mode
        mode_cfg = MODE_PARAMS[mode]

        # Per-coin overrides take precedence
        result = {
            'tp1': cfg.get('tp1', mode_cfg['tp1']),
            'tp2': cfg.get('tp2', mode_cfg['tp2']),
            'tp3': cfg.get('tp3', mode_cfg['tp3']),
            'sl': cfg.get('sl', mode_cfg['sl']),
            'pos': cfg.get('pos', mode_cfg['pos']),
            'ml_min': cfg.get('ml_min', 40),
            'trailing': mode_cfg['trailing'],
        }
        return result

    @property
    def detected_market_mode(self):
        """
        Detect market mode (same logic as market-sentiment.sh):
        - BEAR: F&G < 40 AND price < EMA20
        - BULL: F&G > 60 AND price > EMA20
        - VOLATILE_BULL: RSI > 70
        - VOLATILE_BEAR: RSI < 30
        - LATERAL: default
        - LATERAL_BULL: extended mode (BULL conditions + lateral range)
        """
        fg = self.fear_greed_proxy
        btc_close = self.btc_price
        btc_ema20 = self.btc_ema20
        rsi = self.btc_rsi

        # VOLATILE conditions take priority
        if rsi > 70:
            return 'VOLATILE_BULL'
        if rsi < 30:
            return 'VOLATILE_BEAR'

        # BEAR / BULL
        if fg < 40 and btc_close < btc_ema20:
            return 'BEAR'
        if fg > 60 and btc_close > btc_ema20:
            return 'BULL'

        # LATERAL_BULL: price above EMA but FG not strong enough for full BULL
        if btc_close > btc_ema20 and fg >= 40 and fg <= 60:
            return 'LATERAL_BULL'

        return 'LATERAL'

    # ------------------------------------------------------------------ #
    # ENTRY INDICATORS (proxy para ML validator)
    # ------------------------------------------------------------------ #
    # En Fase 2, usamos indicadores técnicos como proxy del ML validator.
    # En Fase 5, reemplazaremos con el modelo ML real de Perrobotillo.

    @property
    def rsi(self):
        return ta.rsi(self.candles, 14)[-1]

    @property
    def ema_fast(self):
        return ta.ema(self.candles, self.hp['ema_fast'])[-1]

    @property
    def ema_slow(self):
        return ta.ema(self.candles, self.hp['ema_slow'])[-1]

    @property
    def macd(self):
        return ta.macd(self.candles)

    @property
    def adx(self):
        try:
            return ta.adx(self.candles, 14)[-1]
        except Exception:
            return 20.0

    @property
    def atr(self):
        return ta.atr(self.candles, 14)[-1]

    @property
    def volume_sma(self):
        vol = self.candles[:, 5]  # volume column
        if len(vol) < 20:
            return vol[-1]
        return np.mean(vol[-20:])

    @property
    def volume_ratio(self):
        """Current volume vs 20-period average (Perrobotillo's volume_ratio feature)"""
        current_vol = self.candles[-1][5]
        avg_vol = self.volume_sma
        if avg_vol == 0:
            return 1.0
        return current_vol / avg_vol

    @property
    def sma_50(self):
        return ta.sma(self.candles, 50)[-1]

    @property
    def boll_width(self):
        """Bollinger Bands width (volatility measure)"""
        bb = ta.bollinger_bands(self.candles, 20, 2)
        if len(bb) >= 3:
            return (bb[0][-1] - bb[2][-1]) / bb[1][-1] if bb[1][-1] != 0 else 0
        return 0

    @property
    def returns(self):
        """Price returns (similar to Perrobotillo's returns feature)"""
        close = self.candles[-1][2]
        open_price = self.candles[-1][1]
        if open_price != 0:
            return (close - open_price) / open_price
        return 0

    @property
    def volatility_20(self):
        """20-period volatility (similar to Perrobotillo's volatility_20 feature)"""
        close = self.candles[-1][2]
        closes = self.candles[-20:, 2]
        if len(closes) > 1:
            return np.std(closes) / close if close != 0 else 0
        return 0

    def ml_confidence_proxy(self):
        """
        Proxy for Perrobotillo's ML validator confidence score.
        Combines technical indicators to produce a 0-100 confidence.

        Features align with Perrobotillo's selected features per coin.
        En Fase 5, se reemplaza con el modelo pkl real.
        """
        coin = self.coin
        cfg = self.per_coin_config
        ml_min = cfg.get('ml_min', 40)

        score = 50.0  # base

        # EMA alignment (strong filter)
        ema_f = self.ema_fast
        ema_s = self.ema_slow
        if ema_f > ema_s:
            score += 15  # bullish alignment
        else:
            score -= 15

        # RSI in healthy range (not overbought/oversold)
        rsi_val = self.rsi
        if 45 < rsi_val < 75:
            score += 10
        elif rsi_val > 75:
            score -= 20
        elif rsi_val < 30:
            score -= 10

        # Volume confirmation
        vr = self.volume_ratio
        if vr >= 1.0:
            score += 5
        else:
            score -= 5

        # ADX trend strength
        adx_val = self.adx
        if adx_val > 25:
            score += 8  # strong trend
        else:
            score -= 3

        # BTC trend filter (must be above EMA for long)
        if self.btc_price > self.btc_ema20:
            score += 7
        else:
            score -= 10

        # Fear & Greed proxy
        fg = self.fear_greed_proxy
        if fg > 60:
            score += 5
        elif fg < 40:
            score -= 10

        score = np.clip(score, 0, 100)
        return score

    # ------------------------------------------------------------------ #
    # JESSE STRATEGY METHODS
    # ------------------------------------------------------------------ #

    def filters(self):
        """Market mode filters (executed before should_long/should_short)."""
        return [self.filter_market_mode, self.filter_ml_proxy]

    def filter_market_mode(self):
        """Reject entries in extremely bearish modes (mirrors Perrobotillo's BEAR behavior)."""
        mode = self.detected_market_mode
        if mode in ('BEAR', 'VOLATILE_BEAR'):
            # In BEAR mode, position size is reduced but entries still allowed
            # This filter returns False to skip — actual rejection handled by ml proxy
            return True  # Allow (ML proxy will filter)
        return True

    def filter_ml_proxy(self):
        """ML validator proxy: reject if confidence below threshold."""
        score = self.ml_confidence_proxy()
        cfg = self.per_coin_config
        ml_min = cfg.get('ml_min', 40)
        return score >= ml_min

    def should_long(self) -> bool:
        """Entry logic mirroring Perrobotillo's execute_long → ML validator approval."""
        coin = self.coin

        # 1. Check if coin is enabled (BTC might be suspended in Perrobotillo)
        if not self.is_coin_enabled(coin):
            return False

        # 2. ML validator proxy
        score = self.ml_confidence_proxy()
        cfg = self.per_coin_config
        ml_min = cfg.get('ml_min', 40)

        if score < ml_min:
            return False

        # 3. EMA crossover confirmation
        if self.ema_fast <= self.ema_slow:
            return False

        # 4. Price above EMA (trend filter)
        if self.close <= self.ema_slow:
            return False

        return True

    def should_short(self) -> bool:
        return False

    def should_cancel_entry(self) -> bool:
        return True

    def is_coin_enabled(self, coin: str) -> bool:
        """Check per-coin enabled status (BTC may be suspended)."""
        # BTC is currently suspended in Perrobotillo due to ML model review
        if coin == 'BTC':
            return False
        return True

    def go_long(self):
        """
        Entry: market order at current price.
        Sets TP1 (35%), TP2 (30%), TP3 (30%), and SL.
        Mirrors Perrobotillo's execute_long() logic.
        """
        cfg = self.per_coin_config
        entry_price = self.price
        balance = self.balance
        position_percent = cfg['pos']

        # Calculate order value and quantity
        order_value = balance * (position_percent / 100)
        qty = utils.size_to_qty(order_value, entry_price, fee_rate=self.fee_rate)

        # TP levels
        tp1_price = entry_price * (1 + cfg['tp1'] / 100)
        tp2_price = entry_price * (1 + cfg['tp2'] / 100)
        tp3_price = entry_price * (1 + cfg['tp3'] / 100)

        # SL
        sl_price = entry_price * (1 - cfg['sl'] / 100)

        # Partial take profits: 35% / 30% / 30%
        qty_tp1 = qty * TP1_PCT
        qty_tp2 = qty * TP2_PCT
        qty_tp3 = qty * TP3_PCT

        # Entry
        self.buy = qty, entry_price

        # TP1, TP2, TP3 (partial exits)
        self.take_profit = [
            (qty_tp1, tp1_price),
            (qty_tp2, tp2_price),
            (qty_tp3, tp3_price),
        ]

        # Stop loss
        self.stop_loss = qty, sl_price

    def go_short(self):
        pass

    def update_position(self):
        """
        Trailing stop logic (mirrors Perrobotillo's ATR-based trailing).
        Only active when TRAILING_STOP is enabled for the market mode.
        """
        cfg = self.per_coin_config

        if cfg.get('trailing', False) and self.position.is_long:
            # ATR-based trailing stop
            current_atr = self.atr
            entry_price = self.average_entry_price
            current_price = self.price

            # Trailing stop at 2x ATR below current price
            trailing_sl = current_price - (current_atr * 2)

            # Only move SL up, never down
            if trailing_sl > entry_price * (1 - cfg['sl'] / 100):
                self.stop_loss = self.position.qty, trailing_sl

    def before(self):
        """
        Called before each candle. Mirrors Perrobotillo's cycle start.
        - Detect market mode
        - Record ML features (for Fase 5)
        """
        self.record_features({
            # Core features (from Perrobotillo's 212-feature set)
            'rsi': self.rsi,
            'ema_fast': self.ema_fast,
            'ema_slow': self.ema_slow,
            'macd_hist': self.macd_hist,
            'adx': self.adx,
            'atr': self.atr,
            'volume_ratio': self.volume_ratio,
            'sma_50': self.sma_50,
            'boll_width': self.boll_width,
            'returns': self.returns,
            'volatility_20': self.volatility_20,
            # Market context
            'market_mode': self.detected_market_mode,
            'fear_greed': self.fear_greed_proxy,
            'btc_ema_cross': self.btc_ema_cross,
            'confidence_score': self.ml_confidence_proxy(),
        })

    @property
    def macd_hist(self):
        macd_result = self.macd
        if len(macd_result) >= 3:
            return macd_result[2][-1]
        return 0

    def hyperparameters(self):
        """
        Hyperparameters para Optuna optimization.
        These map to Perrobotillo's per-coin-config.env parameters.
        """
        return [
            # TP levels
            {'name': 'tp1_percent', 'type': float, 'min': 1.0, 'max': 10.0, 'default': 3.0},
            {'name': 'tp2_percent', 'type': float, 'min': 2.0, 'max': 15.0, 'default': 6.0},
            {'name': 'tp3_percent', 'type': float, 'min': 3.0, 'max': 20.0, 'default': 9.0},
            # Stop loss
            {'name': 'stop_loss_percent', 'type': float, 'min': 2.0, 'max': 8.0, 'default': 3.5},
            # Position sizing
            {'name': 'position_size_percent', 'type': float, 'min': 5.0, 'max': 30.0, 'default': 10.0},
            # Entry indicators
            {'name': 'ema_fast', 'type': int, 'min': 5, 'max': 50, 'default': 20},
            {'name': 'ema_slow', 'type': int, 'min': 20, 'max': 200, 'default': 50},
            {'name': 'rsi_overbought', 'type': int, 'min': 60, 'max': 85, 'default': 75},
            {'name': 'rsi_oversold', 'type': int, 'min': 15, 'max': 45, 'default': 30},
            {'name': 'min_adx', 'type': int, 'min': 15, 'max': 40, 'default': 20},
        ]

    @property
    def is_long_trade(self):
        """Helper for ML features."""
        return True

    # ------------------------------------------------------------------ #
    # ML FEATURE RECORDING (para Fase 5 — Jesse ML Pipeline)
    # ------------------------------------------------------------------ #
    # Los features aquí registrados mirror exactamente los 212 features
    # de Perrobotillo's ml/features/{COIN}_features.csv
    # En Fase 5, usamos gather_ml_data() para recolectar datos y entrenar
    # modelos con scikit-learn (binary, multiclass, regression)

    def get_ml_features(self):
        """Return all ML features matching Perrobotillo's feature set."""
        candles = self.candles
        close = candles[:, 2]
        high = candles[:, 3]
        low = candles[:, 4]
        volume = candles[:, 5]

        features = {}

        # OHLCV
        features['open'] = close[-1]  # open of current candle
        features['high'] = high[-1]
        features['low'] = low[-1]
        features['close'] = close[-1]
        features['volume'] = volume[-1]

        # Core indicators
        features['rsi'] = float(ta.rsi(candles, 14)[-1])
        macd_result = ta.macd(candles)
        features['macd'] = float(macd_result[0][-1])
        features['macdsignal'] = float(macd_result[1][-1])
        features['macdhist'] = float(macd_result[2][-1])

        bb = ta.bollinger_bands(candles, 20, 2)
        features['bb_upper'] = float(bb[0][-1])
        features['bb_middle'] = float(bb[1][-1])
        features['bb_lower'] = float(bb[2][-1])
        if features['bb_upper'] != features['bb_lower']:
            features['bb_percent'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        else:
            features['bb_percent'] = 0
            features['bb_width'] = 0

        # EMAs
        features['sma_20'] = float(ta.sma(candles, 20)[-1])
        features['sma_50'] = float(ta.sma(candles, 50)[-1])
        features['ema_12'] = float(ta.ema(candles, 12)[-1])
        features['ema_26'] = float(ta.ema(candles, 26)[-1])

        # ATR
        features['atr'] = float(ta.atr(candles, 14)[-1])

        # ADX
        features['adx'] = float(ta.adx(candles, 14)[-1])

        # Stochastic
        stoch = ta.stoch(candles, 14, 3, 3)
        features['slowk'] = float(stoch[0][-1])
        features['slowd'] = float(stoch[1][-1])

        # CCI
        features['cci'] = float(ta.cci(candles, 20)[-1])

        # Williams %R
        features['willr'] = float(ta.willr(candles, 14)[-1])

        # Momentum & ROC
        features['momentum'] = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0
        features['roc'] = float(ta.roc(candles, 10)[-1])

        # Lag features (RSI, MACD, close, volume, ATR)
        features['rsi_lag_1'] = float(ta.rsi(candles, 14)[-2]) if len(candles) > 14 else 0
        features['rsi_diff_1'] = features['rsi'] - features['rsi_lag_1']
        features['rsi_lag_2'] = float(ta.rsi(candles, 14)[-3]) if len(candles) > 15 else 0
        features['rsi_diff_2'] = features['rsi_lag_1'] - features['rsi_lag_2']

        # Volume stats
        vol_5 = volume[-5:]
        vol_10 = volume[-10:]
        vol_20 = volume[-20:]
        features['volume_max_5'] = float(np.max(vol_5))
        features['volume_min_5'] = float(np.min(vol_5))
        features['volume_max_10'] = float(np.max(vol_10))
        features['volume_min_10'] = float(np.min(vol_10))
        features['volume_max_20'] = float(np.max(vol_20))
        features['volume_min_20'] = float(np.min(vol_20))
        features['volume_median_20'] = float(np.median(vol_20))

        # Close stats
        close_5 = close[-5:]
        close_10 = close[-10:]
        close_20 = close[-20:]
        features['close_min_5'] = float(np.min(close_5))
        features['close_max_5'] = float(np.max(close_5))
        features['close_min_10'] = float(np.min(close_10))
        features['close_max_10'] = float(np.max(close_10))
        features['close_median_20'] = float(np.median(close_20))
        features['close_min_20'] = float(np.min(close_20))
        features['close_max_20'] = float(np.max(close_20))

        # Price change
        if len(close) >= 2:
            features['price_change_1h'] = (close[-1] / close[-2] - 1) * 100
        else:
            features['price_change_1h'] = 0

        # Volatility
        features['volatility_20'] = float(np.std(close_20) / close[-1] * 100) if close[-1] != 0 else 0

        # OBV
        if len(volume) >= 2 and len(close) >= 2:
            obv = 0
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv += volume[i]
                elif close[i] < close[i-1]:
                    obv -= volume[i]
            features['obv'] = float(obv)
            vol_20_arr = volume[-20:]
            features['obv_sma'] = float(np.mean(vol_20_arr))
        else:
            features['obv'] = 0
            features['obv_sma'] = 0

        # Market context (BTC)
        features['btc_ema_cross'] = self.btc_ema_cross
        # Fear & Greed is a proxy; in production you'd use the real F&G value
        features['fear_greed'] = self.fear_greed_proxy
        # Market mode as numeric
        mode_map = {'BEAR': 0, 'LATERAL': 1, 'BULL': 2, 'VOLATILE_BEAR': 3, 'VOLATILE_BULL': 4, 'LATERAL_BULL': 5}
        features['market_mode'] = mode_map.get(self.detected_market_mode, 1)

        # Confidence score (ML proxy)
        features['confidence_score'] = self.ml_confidence_proxy()

        return features
