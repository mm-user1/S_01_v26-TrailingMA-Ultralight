# 📊 АНАЛИЗ ТЕКУЩЕЙ АРХИТЕКТУРЫ И ПРОБЛЕМЫ

## Текущее состояние

**Критическая проблема**: Вся логика стратегии S_01 жестко зашита в коде инфраструктуры:

1. **backtest_engine.py::run_strategy()** (~270 строк) - содержит:
   - Логику входа (счетчик закрытий выше/ниже MA)
   - ATR-based стопы с RR и lookback period
   - Trailing MA exits с активацией
   - Max stop % и max days фильтры
   - Risk management на основе riskPerTrade
2. **optimizer_engine.py::_simulate_combination()** (~400 строк) - дублирует всю логику run_strategy() для производительности
3. **StrategyParams** - датакласс с 28 полями, специфичными для S_01

**Все остальное - это инфраструктура:**

- Оптимизаторы (grid, optuna)
- WFA engine
- MA функции (11 типов)
- Preset система
- CSV export/import
- Flask API

## Анализ стратегий

**S_01 (TrailingMA):**

- Trend-following с защитными механизмами
- ~28 параметров
- Сложная логика управления позицией
- Risk-based position sizing

**S_03 (Reversal):**

- Reversal система, всегда в рынке
- ~10-15 параметров
- БЕЗ стопов/таргетов
- Equity-based position sizing (100% капитала)
- Опциональный фильтр по дням недели

**Ключевое различие**: Это две ПРИНЦИПИАЛЬНО разные системы управления позицией.

------

# 💡 ИДЕИ ИЗ JESSE.TRADE

Изучил Jesse.Trade - вот что можно взять:

## 1. **Базовый класс Strategy (ABC паттерн)**

```python
class Strategy(ABC):
    @abstractmethod
    def should_long(self) -> bool:
        """Логика сигнала на вход в лонг"""
        
    @abstractmethod
    def should_short(self) -> bool:
        """Логика сигнала на вход в шорт"""
        
    @abstractmethod
    def go_long(self, ...):
        """Выполнение входа в лонг"""
        
    # и т.д.
```

**Преимущества:**

- Четкий контракт для всех стратегий
- Невозможно забыть реализовать критичные методы
- Type safety и IDE подсказки

## 2. **Методы жизненного цикла**

```python
def before() -> None:
    """Хук перед началом бара"""
    
def after() -> None:
    """Хук после обработки бара"""
    
def on_open_position(trade: Trade) -> None:
    """Событие при открытии позиции"""
    
def on_close_position(trade: Trade) -> None:
    """Событие при закрытии позиции"""
```

## 3. **Hyperparameters для оптимизации**

```python
def hyperparameters(self):
    return [
        {'name': 'ma_length', 'type': int, 'min': 10, 'max': 200},
        {'name': 'stop_atr', 'type': float, 'min': 1.0, 'max': 5.0, 'step': 0.1}
    ]
```

Это АВТОМАТИЧЕСКИ генерирует UI и параметры оптимизации!

## 4. **Свойства доступа к данным**

```python
@property
def close(self) -> np.ndarray:
    """Массив цен закрытия"""
    
@property
def candles(self) -> pd.DataFrame:
    """Весь OHLCV датафрейм"""
```

## 5. **Встроенный риск-менеджмент**

```python
utils.size_to_qty(
    capital=equity,
    price=close,
    risk_pct=2.0,
    stop_distance=stop_distance
)
```

------

# 🏗️ ПРЕДЛАГАЕМАЯ АРХИТЕКТУРА

## Ключевой принцип разделения

```
┌─────────────────────────────────────────────────────┐
│          INFRASTRUCTURE (Unchangeable)              │
├─────────────────────────────────────────────────────┤
│ • MA functions (11 types)                           │
│ • ATR, indicators                                   │
│ • Optimizer engines (grid, optuna)                  │
│ • WFA engine                                        │
│ • Flask API                                         │
│ • CSV export/import                                 │
│ • Preset system                                     │
└─────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────┐
│           STRATEGY INTERFACE (Contract)             │
├─────────────────────────────────────────────────────┤
│ class BaseStrategy(ABC):                            │
│     @abstractmethod                                 │
│     def should_long(...) -> bool                    │
│     @abstractmethod                                 │
│     def should_short(...) -> bool                   │
│     @abstractmethod                                 │
│     def calculate_position_size(...) -> float       │
│     @abstractmethod                                 │
│     def get_exit_signals(...) -> ExitSignals        │
│     def get_hyperparameters() -> List[Param]        │
└─────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────┐
│         CONCRETE STRATEGIES (Changeable)            │
├─────────────────────────────────────────────────────┤
│ S_01_TrailingMA(BaseStrategy)                       │
│ S_03_Reversal(BaseStrategy)                         │
│ S_XX_YourStrategy(BaseStrategy)  ← Легко добавить! │
└─────────────────────────────────────────────────────┘
```

## Структура файлов

```
src/
├── Strategies/              ← НОВАЯ ПАПКА
│   ├── __init__.py
│   ├── base_strategy.py     ← Базовый класс + типы
│   ├── S_01_TrailingMA.py   ← Текущая стратегия
│   ├── S_03_Reversal.py     ← Новая стратегия
│   └── README.md            ← ГАЙД для будущих агентов
├── backtest_engine.py       ← РЕФАКТОРИНГ: убрать логику стратегии
├── optimizer_engine.py      ← РЕФАКТОРИНГ: убрать логику стратегии
├── optuna_engine.py         ← Минимальные изменения
├── walkforward_engine.py    ← Минимальные изменения
├── server.py                ← Добавить strategy selector
└── indicators.py            ← НОВЫЙ: вынести MA functions сюда
```

------

# 📋 ДЕТАЛЬНЫЙ ПЛАН РЕФАКТОРИНГА

## ЭТАП 1: ПОДГОТОВКА ИНФРАСТРУКТУРЫ

### 1.1. Создать модуль indicators.py

**Цель**: Вынести все функции индикаторов из backtest_engine.py

```python
# indicators.py
def ema(series: pd.Series, length: int) -> pd.Series: ...
def sma(series: pd.Series, length: int) -> pd.Series: ...
# ... все 11 MA типов
def atr(high, low, close, period: int) -> pd.Series: ...
def get_ma(series, ma_type, length, **kwargs) -> pd.Series: ...
```

**Зачем**: Индикаторы - это общий код, используемый всеми стратегиями.

### 1.2. Создать базовый контракт стратегии

```python
# Strategies/base_strategy.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class MarketData:
    """Все данные рынка, доступные стратегии"""
    df: pd.DataFrame           # Полный OHLCV
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    volume: np.ndarray
    times: pd.DatetimeIndex
    current_idx: int           # Текущий бар
    
    @property
    def current_close(self) -> float:
        return self.close[self.current_idx]
    
    # ... другие хелперы

@dataclass
class PositionState:
    """Текущее состояние позиции"""
    position: int              # 1 = long, -1 = short, 0 = flat
    entry_price: float
    entry_time: pd.Timestamp
    position_size: float
    realized_equity: float
    # ... все что нужно для управления позицией

@dataclass
class ExitSignals:
    """Сигналы на выход из позиции"""
    should_exit: bool
    exit_price: Optional[float] = None
    exit_reason: str = ""      # "stop", "target", "trailing", "max_days"

@dataclass
class StrategyParameter:
    """Описание одного параметра стратегии"""
    name: str
    display_name: str
    type: type                 # int, float, str
    default: any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None  # Для categorical
    
class BaseStrategy(ABC):
    """
    Базовый класс для всех торговых стратегий.
    
    Все стратегии ДОЛЖНЫ наследоваться от этого класса
    и реализовать абстрактные методы.
    """
    
    def __init__(self, params: dict):
        """
        params: словарь со всеми параметрами стратегии
        """
        self.params = params
        self._validate_params()
    
    @abstractmethod
    def _validate_params(self) -> None:
        """Валидация параметров стратегии"""
        pass
    
    @abstractmethod
    def should_long(
        self, 
        market: MarketData,
        position: PositionState
    ) -> bool:
        """
        Проверка условий для входа в лонг.
        
        ВАЖНО: Метод вызывается каждый бар.
        Логика фильтров (dateFilter, prev_position и т.д.)
        должна быть включена здесь.
        
        Returns:
            True если все условия для лонга выполнены
        """
        pass
    
    @abstractmethod
    def should_short(
        self, 
        market: MarketData,
        position: PositionState
    ) -> bool:
        """Проверка условий для входа в шорт"""
        pass
    
    @abstractmethod
    def calculate_entry(
        self,
        market: MarketData,
        position: PositionState,
        direction: str  # "long" or "short"
    ) -> Tuple[float, float, float]:
        """
        Расчет параметров входа.
        
        Returns:
            (entry_price, stop_price, target_price)
            
        Если стратегия не использует стопы/таргеты,
        можно вернуть (entry_price, nan, nan)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        market: MarketData,
        position: PositionState,
        direction: str,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Расчет размера позиции.
        
        Может использовать:
        - Fixed contract size
        - Risk-based sizing
        - Percent of equity
        """
        pass
    
    @abstractmethod
    def get_exit_signals(
        self,
        market: MarketData,
        position: PositionState
    ) -> ExitSignals:
        """
        Проверка условий выхода из позиции.
        
        Вызывается каждый бар если есть открытая позиция.
        Здесь должна быть вся логика:
        - Стоп-лоссы
        - Тейк-профиты  
        - Trailing stops
        - Max days
        - Reversal сигналы
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> List[StrategyParameter]:
        """
        Возвращает список всех параметров стратегии
        для автогенерации UI и оптимизации.
        
        Пример:
            return [
                StrategyParameter(
                    name="ma_length",
                    display_name="MA Length",
                    type=int,
                    default=45,
                    min_value=10,
                    max_value=200,
                    step=5
                ),
                ...
            ]
        """
        pass
    
    # Хуки жизненного цикла (опциональные)
    
    def on_bar_start(self, market: MarketData) -> None:
        """Вызывается в начале каждого бара (опционально)"""
        pass
    
    def on_bar_end(self, market: MarketData) -> None:
        """Вызывается в конце каждого бара (опционально)"""
        pass
    
    def on_position_opened(
        self, 
        market: MarketData,
        position: PositionState
    ) -> None:
        """Событие открытия позиции (опционально)"""
        pass
    
    def on_position_closed(
        self,
        market: MarketData,
        position: PositionState,
        trade_pnl: float
    ) -> None:
        """Событие закрытия позиции (опционально)"""
        pass
```

**Почему именно так:**

1. **MarketData** - инкапсуляция всех рыночных данных, легко расширять
2. **PositionState** - вся инфа о позиции в одном месте
3. **ExitSignals** - четкий контракт выхода
4. **StrategyParameter** - самодокументирующиеся параметры
5. **Хуки** - для кастомной логики без изменения базового класса

------

## ЭТАП 2: ИЗВЛЕЧЕНИЕ S_01 В ОТДЕЛЬНЫЙ ФАЙЛ

### 2.1. Создать S_01_TrailingMA.py

```python
# Strategies/S_01_TrailingMA.py

from .base_strategy import (
    BaseStrategy, MarketData, PositionState, 
    ExitSignals, StrategyParameter
)
from indicators import get_ma, atr
import math
import pandas as pd

class S_01_TrailingMA(BaseStrategy):
    """
    Trailing Moving Average Strategy
    
    Trend-following стратегия с:
    - MA crossover entry
    - ATR-based stops
    - Trailing MA exits
    - Risk-based position sizing
    """
    
    STRATEGY_NAME = "S_01_TrailingMA"
    STRATEGY_VERSION = "v26"
    
    def __init__(self, params: dict):
        super().__init__(params)
        
        # Кэш для индикаторов (вычисляются один раз)
        self._ma_cache = {}
        self._atr_cache = None
        self._trail_ma_long_cache = None
        self._trail_ma_short_cache = None
        
        # State для trailing stops
        self.trail_price_long = math.nan
        self.trail_price_short = math.nan
        self.trail_activated_long = False
        self.trail_activated_short = False
        
        # State для счетчиков
        self.counter_close_trend_long = 0
        self.counter_close_trend_short = 0
        self.counter_trade_long = 0
        self.counter_trade_short = 0
    
    def _validate_params(self) -> None:
        required = [
            'ma_type', 'ma_length', 'close_count_long', 'close_count_short',
            'stop_long_atr', 'stop_long_rr', 'stop_long_lp',
            'stop_short_atr', 'stop_short_rr', 'stop_short_lp',
            'stop_long_max_pct', 'stop_short_max_pct',
            'stop_long_max_days', 'stop_short_max_days',
            'trail_rr_long', 'trail_rr_short',
            'trail_ma_long_type', 'trail_ma_long_length', 'trail_ma_long_offset',
            'trail_ma_short_type', 'trail_ma_short_length', 'trail_ma_short_offset',
            'risk_per_trade_pct', 'contract_size', 'commission_rate'
        ]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
    
    def _compute_indicators(self, market: MarketData) -> None:
        """Вычисление всех индикаторов один раз"""
        if self._ma_cache:
            return  # Уже вычислены
        
        df = market.df
        
        # Trend MA
        self._ma_cache['trend'] = get_ma(
            df['Close'],
            self.params['ma_type'],
            self.params['ma_length'],
            df['Volume'],
            df['High'],
            df['Low']
        ).to_numpy()
        
        # ATR
        self._atr_cache = atr(
            df['High'],
            df['Low'],
            df['Close'],
            self.params.get('atr_period', 14)
        ).to_numpy()
        
        # Trailing MAs
        self._trail_ma_long_cache = get_ma(
            df['Close'],
            self.params['trail_ma_long_type'],
            self.params['trail_ma_long_length'],
            df['Volume'],
            df['High'],
            df['Low']
        ).to_numpy() * (1 + self.params['trail_ma_long_offset'] / 100.0)
        
        self._trail_ma_short_cache = get_ma(
            df['Close'],
            self.params['trail_ma_short_type'],
            self.params['trail_ma_short_length'],
            df['Volume'],
            df['High'],
            df['Low']
        ).to_numpy() * (1 + self.params['trail_ma_short_offset'] / 100.0)
    
    def should_long(self, market: MarketData, position: PositionState) -> bool:
        self._compute_indicators(market)
        
        i = market.current_idx
        c = market.close[i]
        ma_value = self._ma_cache['trend'][i]
        
        # Update counters
        if not math.isnan(ma_value):
            if c > ma_value:
                self.counter_close_trend_long += 1
                self.counter_close_trend_short = 0
            elif c < ma_value:
                self.counter_close_trend_short += 1
                self.counter_close_trend_long = 0
            else:
                self.counter_close_trend_long = 0
                self.counter_close_trend_short = 0
        
        # Update trade counters
        if position.position > 0:
            self.counter_trade_long = 1
            self.counter_trade_short = 0
        elif position.position < 0:
            self.counter_trade_long = 0
            self.counter_trade_short = 1
        
        # Check conditions
        up_trend = (
            self.counter_close_trend_long >= self.params['close_count_long']
            and self.counter_trade_long == 0
        )
        
        can_open = (
            up_trend
            and position.position == 0
            and not math.isnan(self._atr_cache[i])
        )
        
        return can_open
    
    def should_short(self, market: MarketData, position: PositionState) -> bool:
        # Аналогично should_long, но для шорта
        # ... (код опущен для краткости)
        pass
    
    def calculate_entry(
        self, 
        market: MarketData,
        position: PositionState,
        direction: str
    ) -> Tuple[float, float, float]:
        i = market.current_idx
        c = market.close[i]
        atr_value = self._atr_cache[i]
        
        if direction == "long":
            # Вычисляем lowest low за lookback period
            lookback = self.params['stop_long_lp']
            lowest = market.low[max(0, i-lookback+1):i+1].min()
            
            stop_size = atr_value * self.params['stop_long_atr']
            stop_price = lowest - stop_size
            stop_distance = c - stop_price
            target_price = c + stop_distance * self.params['stop_long_rr']
            
            # Check max stop %
            stop_pct = (stop_distance / c) * 100
            max_stop_pct = self.params['stop_long_max_pct']
            if max_stop_pct > 0 and stop_pct > max_stop_pct:
                return (math.nan, math.nan, math.nan)  # Skip entry
            
            return (c, stop_price, target_price)
        
        else:  # short
            # ... аналогично для short
            pass
    
    def calculate_position_size(
        self,
        market: MarketData,
        position: PositionState,
        direction: str,
        entry_price: float,
        stop_price: float
    ) -> float:
        if math.isnan(entry_price) or math.isnan(stop_price):
            return 0.0
        
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return 0.0
        
        risk_cash = position.realized_equity * (self.params['risk_per_trade_pct'] / 100)
        qty = risk_cash / stop_distance
        
        # Round to contract size
        contract_size = self.params['contract_size']
        if contract_size > 0:
            qty = math.floor(qty / contract_size) * contract_size
        
        return qty
    
    def get_exit_signals(
        self,
        market: MarketData,
        position: PositionState
    ) -> ExitSignals:
        i = market.current_idx
        h = market.high[i]
        l = market.low[i]
        c = market.close[i]
        current_time = market.times[i]
        
        if position.position > 0:  # Long position
            # Trailing stop activation
            if not self.trail_activated_long:
                activation_price = (
                    position.entry_price +
                    (position.entry_price - position.stop_price) * self.params['trail_rr_long']
                )
                if h >= activation_price:
                    self.trail_activated_long = True
                    self.trail_price_long = position.stop_price
            
            # Update trailing price
            trail_value = self._trail_ma_long_cache[i]
            if not math.isnan(trail_value):
                if math.isnan(self.trail_price_long) or trail_value > self.trail_price_long:
                    self.trail_price_long = trail_value
            
            # Check exit conditions
            if self.trail_activated_long:
                if l <= self.trail_price_long:
                    exit_price = h if self.trail_price_long > h else self.trail_price_long
                    return ExitSignals(True, exit_price, "trailing")
            else:
                # Regular stop/target
                if l <= position.stop_price:
                    return ExitSignals(True, position.stop_price, "stop")
                if h >= position.target_price:
                    return ExitSignals(True, position.target_price, "target")
            
            # Max days filter
            max_days = self.params['stop_long_max_days']
            if max_days > 0:
                days_in_trade = int((current_time - position.entry_time).total_seconds() / 86400)
                if days_in_trade >= max_days:
                    return ExitSignals(True, c, "max_days")
        
        elif position.position < 0:  # Short position
            # ... аналогично для short
            pass
        
        return ExitSignals(False)
    
    def get_hyperparameters(self) -> List[StrategyParameter]:
        """Автогенерация параметров для UI"""
        return [
            StrategyParameter("ma_type", "Trend MA Type", str, "EMA",
                            options=["SMA", "EMA", "HMA", "WMA", "ALMA", "KAMA", "TMA", "T3", "DEMA", "VWMA", "VWAP"]),
            StrategyParameter("ma_length", "MA Length", int, 45, 10, 200, 5),
            StrategyParameter("close_count_long", "Close Count Long", int, 7, 1, 20, 1),
            StrategyParameter("close_count_short", "Close Count Short", int, 5, 1, 20, 1),
            # ... все остальные 24 параметра
        ]
    
    def on_position_closed(self, market, position, trade_pnl):
        """Reset trailing stops при закрытии позиции"""
        self.trail_activated_long = False
        self.trail_activated_short = False
        self.trail_price_long = math.nan
        self.trail_price_short = math.nan
```

**Ключевые моменты:**

1. Вся логика S_01 инкапсулирована в одном классе
2. Кэширование индикаторов для производительности
3. State management (counters, trailing stops) внутри класса
4. Полная самодостаточность - можно брать и переносить в другой проект

------

## ЭТАП 3: РЕФАКТОРИНГ BACKTEST ENGINE

### 3.1. Новый run_strategy()

```python
# backtest_engine.py (ПОСЛЕ рефакторинга)

def run_strategy(
    df: pd.DataFrame, 
    strategy: BaseStrategy,  # ← Принимаем объект стратегии!
    params: StrategyParams   # Для compatibility
) -> StrategyResult:
    """
    Универсальный движок бэктестинга.
    Работает с ЛЮБОЙ стратегией, реализующей BaseStrategy.
    """
    
    # Prepare market data
    market = MarketData(
        df=df,
        close=df['Close'].to_numpy(),
        high=df['High'].to_numpy(),
        low=df['Low'].to_numpy(),
        open=df['Open'].to_numpy(),
        volume=df['Volume'].to_numpy(),
        times=df.index,
        current_idx=0
    )
    
    # Initialize position state
    position = PositionState(
        position=0,
        entry_price=math.nan,
        entry_time=None,
        position_size=0.0,
        realized_equity=100.0
    )
    
    prev_position = 0
    trades: List[TradeRecord] = []
    equity_curve: List[float] = []
    
    # MAIN SIMULATION LOOP
    for i in range(len(df)):
        market.current_idx = i
        
        # Hook: bar start
        strategy.on_bar_start(market)
        
        # ═══════════════════════════════════════════════════
        # EXIT LOGIC (если есть позиция)
        # ═══════════════════════════════════════════════════
        
        if position.position != 0:
            exit_signals = strategy.get_exit_signals(market, position)
            
            if exit_signals.should_exit:
                # Расчет PnL
                if position.position > 0:
                    gross_pnl = (exit_signals.exit_price - position.entry_price) * position.position_size
                else:
                    gross_pnl = (position.entry_price - exit_signals.exit_price) * position.position_size
                
                exit_commission = exit_signals.exit_price * position.position_size * params.commission_rate
                net_pnl = gross_pnl - position.entry_commission - exit_commission
                
                position.realized_equity += gross_pnl - exit_commission
                
                # Record trade
                trades.append(TradeRecord(
                    direction="long" if position.position > 0 else "short",
                    entry_time=position.entry_time,
                    exit_time=market.times[i],
                    entry_price=position.entry_price,
                    exit_price=exit_signals.exit_price,
                    size=position.position_size,
                    net_pnl=net_pnl
                ))
                
                # Callback
                strategy.on_position_closed(market, position, net_pnl)
                
                # Reset position
                position.position = 0
                position.position_size = 0.0
                position.entry_price = math.nan
                position.stop_price = math.nan
                position.target_price = math.nan
                position.entry_time = None
                position.entry_commission = 0.0
        
        # ═══════════════════════════════════════════════════
        # ENTRY LOGIC (если нет позиции)
        # ═══════════════════════════════════════════════════
        
        if position.position == 0 and prev_position == 0:
            # Check long
            if strategy.should_long(market, position):
                entry_price, stop_price, target_price = strategy.calculate_entry(
                    market, position, "long"
                )
                
                if not math.isnan(entry_price):
                    qty = strategy.calculate_position_size(
                        market, position, "long", entry_price, stop_price
                    )
                    
                    if qty > 0:
                        position.position = 1
                        position.position_size = qty
                        position.entry_price = entry_price
                        position.stop_price = stop_price
                        position.target_price = target_price
                        position.entry_time = market.times[i]
                        position.entry_commission = entry_price * qty * params.commission_rate
                        position.realized_equity -= position.entry_commission
                        
                        strategy.on_position_opened(market, position)
            
            # Check short (если не зашли в лонг)
            elif strategy.should_short(market, position):
                # ... аналогично для short
                pass
        
        # ═══════════════════════════════════════════════════
        # EQUITY TRACKING
        # ═══════════════════════════════════════════════════
        
        current_equity = position.realized_equity
        if position.position > 0:
            current_equity += (market.close[i] - position.entry_price) * position.position_size
        elif position.position < 0:
            current_equity += (position.entry_price - market.close[i]) * position.position_size
        
        equity_curve.append(current_equity)
        
        # Hook: bar end
        strategy.on_bar_end(market)
        
        prev_position = position.position
    
    # Calculate final metrics
    equity_series = pd.Series(equity_curve, index=df.index)
    net_profit_pct = ((position.realized_equity - 100.0) / 100.0) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)
    
    return StrategyResult(
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=len(trades),
        trades=trades
    )
```

**ЧТО ИЗМЕНИЛОСЬ:**

1. ✅ **Универсальность** - работает с ЛЮБОЙ стратегией
2. ✅ **Чистота** - только инфраструктурный код, никакой бизнес-логики
3. ✅ **Расширяемость** - легко добавить новые фичи
4. ✅ **Тестируемость** - можно мокать стратегию для тестов

------

## ЭТАП 4: РЕФАКТОРИНГ OPTIMIZER ENGINE

### 4.1. Изменения в optimizer_engine.py

**Проблема**: _simulate_combination() дублирует run_strategy()

**Решение**: Использовать ту же универсальную логику!

```python
# optimizer_engine.py

def _simulate_combination(params_dict: Dict[str, Any]) -> OptimizationResult:
    """
    Выполняет симуляцию одной комбинации параметров.
    
    ВАЖНО: Теперь создает объект стратегии и использует
    универсальный simulation loop.
    """
    
    # Получаем имя стратегии из глобального контекста
    global _strategy_class
    
    # Создаем экземпляр стратегии с параметрами
    strategy = _strategy_class(params_dict)
    
    # Prepare market data (как раньше из кэшей)
    market = MarketData(
        df=None,  # Не нужен полный DF
        close=_data_close,
        high=_data_high,
        low=_data_low,
        # ...
    )
    
    # Используем ТУ ЖЕ логику что и run_strategy()!
    # (можно вынести в отдельную функцию _run_simulation_core)
    
    # ... simulation loop ...
    
    return OptimizationResult(...)
```

**КРИТИЧЕСКИ ВАЖНО**: Теперь optimizer и backtest используют ОДНУ И ТУ ЖЕ логику!

### 4.2. Добавить strategy_class в OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    csv_file: IO[Any]
    strategy_class: type  # ← НОВОЕ ПОЛЕ!
    enabled_params: Dict[str, bool]
    # ... остальное без изменений
```

------

## ЭТАП 5: СОЗДАНИЕ S_03_REVERSAL

### 5.1. Реализация S_03

```python
# Strategies/S_03_Reversal.py

class S_03_Reversal(BaseStrategy):
    """
    Reversal Strategy
    
    Простая reversal система:
    - Всегда в рынке (long или short)
    - Reversal на сигнале противоположной стороны
    - Опциональный close count filter
    - Опциональный days of week filter
    """
    
    STRATEGY_NAME = "S_03_Reversal"
    STRATEGY_VERSION = "v07"
    
    def __init__(self, params: dict):
        super().__init__(params)
        self._ma_caches = {}  # 3 MAs
        self.count_close_long = 0
        self.count_close_short = 0
    
    def _validate_params(self) -> None:
        required = [
            'ma1_type', 'ma1_length',
            'ma2_type', 'ma2_length',
            'ma3_type', 'ma3_length',
            'use_close_count', 'close_count_long', 'close_count_short',
            'use_days_filter', 'trade_days',  # list of weekday names
            'contract_size'
        ]
        # ... validation
    
    def should_long(self, market: MarketData, position: PositionState) -> bool:
        """
        Long condition для reversal стратегии.
        
        ВАЖНО: Проверяем также возможность reverse из short!
        """
        i = market.current_idx
        c = market.close[i]
        
        # Compute MAs
        self._compute_indicators(market)
        ma3 = self._ma_caches['ma3'][i]
        
        # Update close count
        if c > ma3:
            self.count_close_long += 1
            self.count_close_short = 0
        elif c < ma3:
            self.count_close_short += 1
            self.count_close_long = 0
        else:
            self.count_close_long = 0
            self.count_close_short = 0
        
        # Check close count condition
        if self.params['use_close_count']:
            count_ok = self.count_close_long >= self.params['close_count_long']
        else:
            count_ok = True
        
        # Check days filter
        if self.params['use_days_filter']:
            current_weekday = market.times[i].strftime('%A')
            days_ok = current_weekday in self.params['trade_days']
        else:
            days_ok = True
        
        # ← КЛЮЧЕВОЕ ОТЛИЧИЕ от S_01:
        # Reversal может входить даже если уже в short!
        return count_ok and days_ok
    
    def should_short(self, market: MarketData, position: PositionState) -> bool:
        # ... аналогично
        pass
    
    def calculate_entry(self, market, position, direction):
        """
        Reversal стратегия НЕ использует stops/targets.
        Возвращаем текущую цену и NaN для stop/target.
        """
        i = market.current_idx
        entry_price = market.close[i]
        return (entry_price, math.nan, math.nan)
    
    def calculate_position_size(self, market, position, direction, entry_price, stop_price):
        """
        100% equity position sizing.
        """
        equity = position.realized_equity
        contract_size = self.params['contract_size']
        
        qty = equity / entry_price
        if contract_size > 0:
            qty = math.floor(qty / contract_size) * contract_size
        
        return qty
    
    def get_exit_signals(self, market, position):
        """
        Reversal стратегия не имеет обычных exit сигналов.
        Выход происходит только через reverse signal.
        
        ВАЖНО: Эта логика будет обрабатываться в backtest_engine
        через проверку противоположных should_long/should_short!
        """
        # Можно добавить exit по концу даты диапазона
        if self.params.get('use_date_filter'):
            if market.times[market.current_idx] > self.params.get('end_date'):
                return ExitSignals(True, market.close[market.current_idx], "end_date")
        
        return ExitSignals(False)
    
    def get_hyperparameters(self):
        return [
            StrategyParameter("ma1_type", "MA1 Type", str, "KAMA", options=[...]),
            StrategyParameter("ma1_length", "MA1 Length", int, 15, 5, 100, 5),
            # ... и так далее
        ]
```

**КРИТИЧЕСКИ ВАЖНО для reversal логики:**

Нужно модифицировать backtest_engine для поддержки reversal:

```python
# В run_strategy(), секция ENTRY LOGIC:

if position.position == 0:
    # ... обычная логика входа
    pass
elif position.position != 0:  # ← НОВОЕ!
    # Check для reversal стратегий
    if hasattr(strategy, 'IS_REVERSAL_STRATEGY') and strategy.IS_REVERSAL_STRATEGY:
        # Если в long и есть short signal -> reverse
        if position.position > 0 and strategy.should_short(market, position):
            # 1. Close long
            # 2. Open short
            pass
        # Если в short и есть long signal -> reverse
        elif position.position < 0 and strategy.should_long(market, position):
            # 1. Close short
            # 2. Open long
            pass
```

------

## ЭТАП 6: ИНТЕГРАЦИЯ С UI И API

### 6.1. Добавить strategy selector в server.py

```python
# server.py

# Регистрация доступных стратегий
AVAILABLE_STRATEGIES = {
    "S_01_TrailingMA": S_01_TrailingMA,
    "S_03_Reversal": S_03_Reversal,
}

@app.get("/api/strategies")
def list_strategies():
    """Список доступных стратегий"""
    return jsonify({
        "strategies": [
            {
                "id": name,
                "name": cls.STRATEGY_NAME,
                "version": cls.STRATEGY_VERSION,
                "parameters": [p.to_dict() for p in cls({}).get_hyperparameters()]
            }
            for name, cls in AVAILABLE_STRATEGIES.items()
        ]
    })

@app.post("/api/optimize")
def run_optimization_endpoint():
    # ... existing code ...
    
    # NEW: Get strategy from request
    strategy_id = request.form.get("strategy_id", "S_01_TrailingMA")
    if strategy_id not in AVAILABLE_STRATEGIES:
        return ("Invalid strategy ID", HTTPStatus.BAD_REQUEST)
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_id]
    
    # Build config с strategy_class
    optimization_config = OptimizationConfig(
        csv_file=data_source,
        strategy_class=strategy_class,  # ← НОВОЕ!
        # ... rest
    )
    
    # Run optimization
    results = run_optimization(optimization_config)
    # ...
```

### 6.2. Обновить UI (index.html)

Добавить dropdown со списком стратегий:

```html
<select id="strategySelector" onchange="loadStrategyParameters()">
    <option value="S_01_TrailingMA">S_01 - Trailing MA (v26)</option>
    <option value="S_03_Reversal">S_03 - Reversal (v07)</option>
</select>

<div id="strategyParameters">
    <!-- Динамически генерируется из get_hyperparameters() -->
</div>
```

JavaScript для динамической загрузки параметров:

```javascript
async function loadStrategyParameters() {
    const strategyId = document.getElementById('strategySelector').value;
    const response = await fetch('/api/strategies');
    const data = await response.json();
    
    const strategy = data.strategies.find(s => s.id === strategyId);
    const parametersDiv = document.getElementById('strategyParameters');
    
    // Очистить старые параметры
    parametersDiv.innerHTML = '';
    
    // Создать UI для каждого параметра
    strategy.parameters.forEach(param => {
        const inputHtml = createParameterInput(param);
        parametersDiv.innerHTML += inputHtml;
    });
}

function createParameterInput(param) {
    switch (param.type) {
        case 'int':
            return `<input type="number" 
                           name="${param.name}" 
                           min="${param.min_value}" 
                           max="${param.max_value}" 
                           step="${param.step}" 
                           value="${param.default}">`;
        case 'str':
            if (param.options) {
                return `<select name="${param.name}">
                    ${param.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                </select>`;
            }
            break;
        // ... и так далее
    }
}
```

------

## ЭТАП 7: СОЗДАНИЕ ДОКУМЕНТАЦИИ ДЛЯ БУДУЩИХ АГЕНТОВ

### 7.1. Strategies/README.md - ГАЙД ПО СОЗДАНИЮ СТРАТЕГИЙ

~~~markdown
# Strategy Development Guide

## Как создать новую стратегию

### Шаг 1: Создать файл стратегии

Создайте новый файл в `src/Strategies/` с именем вида `S_XX_YourStrategyName.py`

### Шаг 2: Наследоваться от BaseStrategy

```python
from .base_strategy import BaseStrategy, MarketData, PositionState, ExitSignals, StrategyParameter

class S_XX_YourStrategy(BaseStrategy):
    STRATEGY_NAME = "S_XX_YourStrategy"
    STRATEGY_VERSION = "v01"
    
    # Если это reversal стратегия (всегда в рынке):
    IS_REVERSAL_STRATEGY = True  # опционально
    
    def __init__(self, params: dict):
        super().__init__(params)
        # Инициализация кэшей, state variables
    
    def _validate_params(self) -> None:
        # Проверка обязательных параметров
        required = ['param1', 'param2', ...]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing: {param}")
~~~

### Шаг 3: Реализовать обязательные методы

**КРИТИЧЕСКИ ВАЖНО**: Вы ДОЛЖНЫ реализовать ВСЕ абстрактные методы:

1. `should_long(market, position) -> bool`
2. `should_short(market, position) -> bool`
3. `calculate_entry(market, position, direction) -> (entry, stop, target)`
4. `calculate_position_size(...) -> float`
5. `get_exit_signals(market, position) -> ExitSignals`
6. `get_hyperparameters() -> List[StrategyParameter]`

### Шаг 4: Использовать кэширование для производительности

```python
def __init__(self, params):
    super().__init__(params)
    self._indicator_cache = {}  # Кэш индикаторов
    
def _compute_indicators(self, market: MarketData):
    """Вычислить все индикаторы ОДИН РАЗ"""
    if self._indicator_cache:
        return  # Уже вычислены
    
    # Вычисляем для всего датафрейма сразу
    from indicators import get_ma
    self._indicator_cache['ma'] = get_ma(
        market.df['Close'],
        self.params['ma_type'],
        self.params['ma_length']
    ).to_numpy()
```

### Шаг 5: Следовать naming conventions

**Обязательно:**

- Параметры в camelCase: `maLength`, `closeCountLong`
- Внутренние переменные в snake_case: `_ma_cache`, `counter_long`
- Константы в UPPER_CASE: `STRATEGY_NAME`

**Соответствие Pine → Python:**

```
Pine                    Python
-----------------------------------------
maType                  ma_type (в params dict используем 'maType')
maLength                ma_length
closeCountLong          close_count_long
stopLongX               stop_long_atr
trailRRLong             trail_rr_long
```

### Шаг 6: Тестирование

```python
# test_your_strategy.py
from Strategies.S_XX_YourStrategy import S_XX_YourStrategy
from backtest_engine import load_data, run_strategy

# Загрузить данные
df = load_data('../data/test.csv')

# Создать стратегию
params = {
    'param1': 100,
    'param2': 2.0,
    # ... все параметры
}
strategy = S_XX_YourStrategy(params)

# Запустить бэктест
result = run_strategy(df, strategy, StrategyParams.from_dict(params))

print(f"Profit: {result.net_profit_pct:.2f}%")
print(f"Trades: {result.total_trades}")
```

### Шаг 7: Регистрация в системе

Добавьте в `server.py`:

```python
from Strategies.S_XX_YourStrategy import S_XX_YourStrategy

AVAILABLE_STRATEGIES = {
    "S_01_TrailingMA": S_01_TrailingMA,
    "S_03_Reversal": S_03_Reversal,
    "S_XX_YourStrategy": S_XX_YourStrategy,  # ← NEW
}
```

## Troubleshooting

### Ошибка: "Missing required parameter"

- Проверьте `_validate_params()` - все ли параметры добавлены в required
- Проверьте `get_hyperparameters()` - все ли параметры описаны

### Оптимизация работает медленно

- Убедитесь что индикаторы кэшируются (`_compute_indicators()`)
- Не вызывайте тяжелые вычисления в `should_long()`/`should_short()`

### Стратегия дает другие результаты чем Pine

- Проверьте логику округления position size
- Проверьте commission calculation
- Проверьте intra-bar order execution logic

```
---

# 🎯 СТАНДАРТИЗАЦИЯ И ПРАВИЛА

## Naming Conventions (СТРОГО!)

### Параметры стратегии

**Frontend (JSON, API, UI):**
```javascript
{
  "maType": "EMA",
  "maLength": 45,
  "closeCountLong": 7,
  "stopLongX": 2.0,
  "trailRRLong": 1.0,
  "trailLongType": "SMA",
  "trailLongLength": 160,
  "trailLongOffset": -1.0
}
```

**Python (внутри стратегии):**

```python
self.params = {
    'ma_type': "EMA",
    'ma_length': 45,
    'close_count_long': 7,
    'stop_long_atr': 2.0,
    'trail_rr_long': 1.0,
    'trail_ma_long_type': "SMA",
    'trail_ma_long_length': 160,
    'trail_ma_long_offset': -1.0
}
```

**Преобразование в базовом классе:**

```python
@staticmethod
def _convert_params(frontend_params: dict) -> dict:
    """Конвертирует camelCase в snake_case"""
    mapping = {
        'maType': 'ma_type',
        'maLength': 'ma_length',
        'closeCountLong': 'close_count_long',
        # ... полный mapping
    }
    return {mapping.get(k, k): v for k, v in frontend_params.items()}
```

## Типы данных (СТРОГО!)

```python
# ПРАВИЛЬНО:
ma_length: int = 45
stop_atr: float = 2.0
ma_type: str = "EMA"

# НЕПРАВИЛЬНО:
ma_length: float = 45.0  # ← WRONG! Length всегда int
stop_atr: int = 2  # ← WRONG! Multiplier всегда float
```

## State Management

**State переменные стратегии:**

```python
# Счетчики
self.counter_close_long: int = 0
self.counter_trade_long: int = 0

# Trailing stops
self.trail_price_long: float = math.nan
self.trail_activated_long: bool = False

# Кэши (вычисляются ОДИН раз)
self._ma_cache: Dict[str, np.ndarray] = {}
self._atr_cache: Optional[np.ndarray] = None
```

------

# ⚠️ ПОДВОДНЫЕ КАМНИ И РЕШЕНИЯ

## 1. Дублирование кода между run_strategy() и _simulate_combination()

**Проблема**: Сейчас есть два почти одинаковых simulation loop.

**Решение**: Вынести общую логику в `_run_simulation_core()`:

```python
def _run_simulation_core(
    strategy: BaseStrategy,
    market_data: MarketData,
    params: dict
) -> Tuple[float, float, int, List[TradeRecord]]:
    """
    Ядро симуляции - общее для backtest и optimization.
    
    Returns:
        (net_profit_pct, max_drawdown_pct, total_trades, trades)
    """
    # ... вся логика симуляции ...
    return (net_profit_pct, max_dd, len(trades), trades)

def run_strategy(df, strategy, params):
    market = MarketData(df=df, ...)
    return _run_simulation_core(strategy, market, params)

def _simulate_combination(params_dict):
    global _strategy_class, _cached_market_data
    strategy = _strategy_class(params_dict)
    return _run_simulation_core(strategy, _cached_market_data, params_dict)
```

## 2. Reversal стратегии требуют другой логики

**Проблема**: S_03 должна закрывать противоположную позицию при reverse signal.

**Решение**: Добавить флаг `IS_REVERSAL_STRATEGY` и специальную обработку:

```python
# В BaseStrategy
IS_REVERSAL_STRATEGY = False  # По умолчанию

# В S_03_Reversal
IS_REVERSAL_STRATEGY = True

# В simulation core
if strategy.IS_REVERSAL_STRATEGY and position.position != 0:
    # Проверка reverse signals
    if position.position > 0 and strategy.should_short(market, position):
        # Close long + Open short
        ...
    elif position.position < 0 and strategy.should_long(market, position):
        # Close short + Open long
        ...
```

## 3. Разные стратегии имеют разные параметры

**Проблема**: StrategyParams сейчас жестко привязан к S_01.

**Решение 1 (простой)**: Сделать StrategyParams гибким dict:

```python
# Вместо фиксированных полей - использовать Any
params: Dict[str, Any]
```

**Решение 2 (правильный)**: Каждая стратегия определяет свой dataclass:

```python
@dataclass
class S_01_Params:
    ma_type: str
    ma_length: int
    # ... 26 других полей

@dataclass
class S_03_Params:
    ma1_type: str
    ma1_length: int
    # ... только нужные поля
```

Рекомендую **Решение 1** для простоты.

## 4. UI должен динамически адаптироваться

**Проблема**: Сейчас UI hardcoded для S_01 параметров.

**Решение**: Генерация UI из `get_hyperparameters()`:

```javascript
// Fetch параметров стратегии
const params = await getStrategyParameters(strategyId);

// Динамическая генерация HTML
params.forEach(param => {
    const html = generateInputForParameter(param);
    container.appendChild(html);
});

function generateInputForParameter(param) {
    if (param.type === 'int' || param.type === 'float') {
        return `<div class="param-row">
            <label>${param.display_name}</label>
            <input type="number" 
                   name="${param.name}"
                   min="${param.min_value}"
                   max="${param.max_value}"
                   step="${param.step}"
                   value="${param.default}">
            <label>Enable Optimization</label>
            <input type="checkbox" name="enable_${param.name}">
        </div>`;
    }
    else if (param.options) {
        return `<select name="${param.name}">
            ${param.options.map(opt => `<option>${opt}</option>`)}
        </select>`;
    }
}
```

## 5. Оптимизация параметров должна быть стратегия-специфичной

**Проблема**: Сейчас PARAMETER_MAP hardcoded.

**Решение**: Получать mapping из стратегии:

```python
class BaseStrategy:
    @classmethod
    def get_parameter_mapping(cls) -> Dict[str, Tuple[str, bool]]:
        """
        Returns mapping: frontend_name -> (python_name, is_int)
        
        Генерируется автоматически из get_hyperparameters()
        """
        mapping = {}
        for param in cls({}).get_hyperparameters():
            mapping[param.name] = (
                param.name,  # python name (snake_case)
                param.type == int
            )
        return mapping
```

## 6. Walk-Forward Analysis должен работать с любой стратегией

**Проблема**: WFA engine использует optuna_engine, который должен знать про стратегию.

**Решение**: Передавать strategy_class через конфиг:

```python
class WalkForwardEngine:
    def __init__(
        self, 
        config: WFConfig, 
        strategy_class: type,  # ← ДОБАВИТЬ
        base_config_template: Dict[str, Any],
        optuna_settings: Dict[str, Any]
    ):
        self.strategy_class = strategy_class
        # ...
    
    def _run_optuna_on_window(self, df_window):
        base_config = OptimizationConfig(
            csv_file=csv_buffer,
            strategy_class=self.strategy_class,  # ← ПЕРЕДАТЬ
            # ...
        )
```

------

# 🚀 ПОРЯДОК ВЫПОЛНЕНИЯ (ПОШАГОВО)

## Фаза 1: Подготовка (1-2 дня)

1. ✅ Создать `src/indicators.py` - вынести все MA функции
2. ✅ Создать `src/Strategies/__init__.py`
3. ✅ Создать `src/Strategies/base_strategy.py` - базовый контракт
4. ✅ Написать полные docstrings и type hints
5. ✅ Создать unit tests для indicators.py

## Фаза 2: Извлечение S_01 (2-3 дня)

1. ✅ Создать `src/Strategies/S_01_TrailingMA.py`
2. ✅ Скопировать всю логику из backtest_engine.py
3. ✅ Реализовать все абстрактные методы
4. ✅ Добавить кэширование индикаторов
5. ✅ Тестирование: сравнить результаты старой и новой версии (ДОЛЖНЫ СОВПАДАТЬ!)

## Фаза 3: Рефакторинг backtest_engine (2-3 дня)

1. ✅ Создать новый `run_strategy()` - универсальный
2. ✅ Вынести общую логику в `_run_simulation_core()`
3. ✅ Добавить поддержку reversal стратегий
4. ✅ Удалить старый код из backtest_engine.py
5. ✅ Тестирование: S_01 через новый engine должен давать те же результаты

## Фаза 4: Рефакторинг optimizer_engine (3-4 дня)

1. ✅ Модифицировать `_simulate_combination()` - использовать strategy class
2. ✅ Добавить `strategy_class` в OptimizationConfig
3. ✅ Обновить `_init_worker()` для поддержки стратегий
4. ✅ Модифицировать `generate_parameter_grid()` - получать параметры из стратегии
5. ✅ Тестирование: оптимизация S_01 должна работать как раньше

## Фаза 5: Создание S_03 (2-3 дня)

1. ✅ Изучить Pine код S_03 детально
2. ✅ Создать `src/Strategies/S_03_Reversal.py`
3. ✅ Реализовать всю логику из Pine
4. ✅ Добавить IS_REVERSAL_STRATEGY = True
5. ✅ Тестирование: сравнить с Pine результатами

## Фаза 6: Обновление API и UI (3-4 дня)

1. ✅ Добавить `/api/strategies` endpoint
2. ✅ Модифицировать `/api/optimize` - принимать strategy_id
3. ✅ Модифицировать `/api/backtest` - принимать strategy_id
4. ✅ Модифицировать `/api/walkforward` - принимать strategy_id
5. ✅ Обновить UI - dropdown для выбора стратегии
6. ✅ Динамическая генерация параметров из get_hyperparameters()
7. ✅ Тестирование UI с обеими стратегиями

## Фаза 7: Документация (1-2 дня)

1. ✅ Написать `Strategies/README.md` - полный гайд
2. ✅ Добавить примеры кода
3. ✅ Документировать naming conventions
4. ✅ Создать template для новых стратегий
5. ✅ Написать инструкции для будущих агентов

## Фаза 8: Финальное тестирование (2-3 дня)

1. ✅ Integration tests: оба стратегии через все режимы
2. ✅ Performance testing: оптимизация 10K комбинаций
3. ✅ WFA с обеими стратегиями
4. ✅ CSV export/import с обеими стратегиями
5. ✅ Проверка preset системы

**Итого: 18-27 дней работы**

------

# 📊 ИТОГОВАЯ АРХИТЕКТУРА

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Strategy  │  │ Parameters │  │   Results  │         │
│  │  Selector  │  │    Form    │  │   Display  │         │
│  └────────────┘  └────────────┘  └────────────┘         │
└──────────────────────────────────────────────────────────┘
                         ↕ HTTP
┌──────────────────────────────────────────────────────────┐
│                    FLASK API (server.py)                 │
│  GET  /api/strategies                                    │
│  POST /api/backtest    (strategy_id, params)            │
│  POST /api/optimize    (strategy_id, config)            │
│  POST /api/walkforward (strategy_id, config)            │
└──────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────────┐
│              OPTIMIZATION ENGINES                        │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Grid Search    │  │ Optuna (Bayes)  │               │
│  │ optimizer_engine│  │ optuna_engine   │               │
│  └─────────────────┘  └─────────────────┘               │
│                                                           │
│              WALK-FORWARD ANALYSIS                       │
│  ┌──────────────────────────────────────┐               │
│  │  walkforward_engine                  │               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────────┐
│              BACKTEST ENGINE (CORE)                      │
│  ┌──────────────────────────────────────┐               │
│  │  run_strategy(strategy, data)        │               │
│  │  _run_simulation_core()              │               │
│  │  compute_max_drawdown()              │               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────────┐
│            STRATEGY INTERFACE (ABC)                      │
│  ┌──────────────────────────────────────┐               │
│  │  BaseStrategy                        │               │
│  │  - should_long()                     │               │
│  │  - should_short()                    │               │
│  │  - calculate_entry()                 │               │
│  │  - calculate_position_size()         │               │
│  │  - get_exit_signals()                │               │
│  │  - get_hyperparameters()             │               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────────┐
│            CONCRETE STRATEGIES                           │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ S_01_TrailingMA  │  │ S_03_Reversal    │             │
│  │ (28 parameters)  │  │ (12 parameters)  │             │
│  └──────────────────┘  └──────────────────┘             │
│                                                           │
│  ┌──────────────────┐                                    │
│  │ S_XX_YourNew     │  ← Легко добавить!                │
│  │ (N parameters)   │                                    │
│  └──────────────────┘                                    │
└──────────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────────┐
│            INDICATORS & UTILITIES                        │
│  ┌──────────────────────────────────────┐               │
│  │  indicators.py                       │               │
│  │  - 11 MA types (SMA, EMA, HMA...)    │               │
│  │  - ATR                               │               │
│  │  - get_ma() - unified interface      │               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
```

------

Вот мой детальный анализ и план. Это действительно БОЛЬШОЙ апдейт, но я постарался учесть все нюансы:

✅ **Надежность** - используем ABC паттерн и type hints ✅ **Логичность** - четкое разделение ответственности ✅ **Лаконичность** - вынесли дублирование, общий simulation core ✅ **Понятность** - подробная документация и naming conventions ✅ **Расширяемость** - легко добавить S_04, S_05... S_XX