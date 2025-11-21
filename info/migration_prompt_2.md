# Migration Prompt 2: Create Base Strategy Contract

## Context

**Current Phase:** Phase 2 of 6
**Previous Phase:** Phase 1 ✅ (indicators.py extracted)
**Migration Checklist:** See `migration_checklist.md` - Phase 2
**Duration:** 1-1.5 days

**Goal:** Create the `BaseStrategy` ABC class that defines the contract all strategies must implement. Also create `StrategyRegistry` for strategy management.

---

## Task Overview

Create two new files:
1. `/src/strategies/base_strategy.py` - Abstract base class
2. `/src/strategy_registry.py` - Strategy registration system

**No changes to existing backtest code yet** - we're just establishing the foundation.

---

## Step 1: Create Strategies Folder

```bash
cd src
mkdir -p strategies
touch strategies/__init__.py
```

In `strategies/__init__.py`:
```python
"""Trading strategies module"""
from .base_strategy import BaseStrategy

__all__ = ['BaseStrategy']
```

---

## Step 2: Create base_strategy.py

Create `/src/strategies/base_strategy.py`:

```python
"""
Base Strategy Contract

This module defines the abstract base class that ALL trading strategies
must implement. It provides:
- Contract enforcement via ABC
- Single simulation entry point (eliminates duplication)
- Parameter definition interface for UI generation
- Cache requirements interface for optimization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import math


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    ALL strategies MUST inherit from this class and implement
    all abstract methods.

    Design Principles:
    - Single simulate() method for both backtest and optimization
    - Separation of concerns (entry, exit, sizing, data prep)
    - DRY parameter definitions
    - Type safety via ABC
    """

    # Strategy metadata (override in subclass)
    STRATEGY_ID: str = "base"
    STRATEGY_NAME: str = "Base Strategy"
    VERSION: str = "1.0"

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize strategy with parameters.

        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params
        self._validate_params()

        # Data storage (set by _prepare_data)
        self.df: Optional[pd.DataFrame] = None
        self.close: Optional[np.ndarray] = None
        self.high: Optional[np.ndarray] = None
        self.low: Optional[np.ndarray] = None
        self.open: Optional[np.ndarray] = None
        self.volume: Optional[np.ndarray] = None
        self.times: Optional[pd.DatetimeIndex] = None

    # ════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS (must be implemented by all strategies)
    # ════════════════════════════════════════════════════════════════

    @abstractmethod
    def _validate_params(self) -> None:
        """
        Validate strategy parameters.

        Raise ValueError if any required parameter is missing
        or has invalid value.

        Example:
            required = ['maLength', 'stopAtr', 'riskPct']
            for param in required:
                if param not in self.params:
                    raise ValueError(f"Missing parameter: {param}")
        """
        pass

    @abstractmethod
    def should_long(self, idx: int) -> bool:
        """
        Check if long entry conditions are met at given bar.

        Args:
            idx: Current bar index

        Returns:
            True if should enter long, False otherwise

        Note:
            This method is called every bar when position is flat
            or when checking for reversal.
        """
        pass

    @abstractmethod
    def should_short(self, idx: int) -> bool:
        """
        Check if short entry conditions are met at given bar.

        Args:
            idx: Current bar index

        Returns:
            True if should enter short, False otherwise
        """
        pass

    @abstractmethod
    def calculate_entry(
        self,
        idx: int,
        direction: str
    ) -> Tuple[float, float, float]:
        """
        Calculate entry price, stop price, and target price.

        Args:
            idx: Current bar index
            direction: "long" or "short"

        Returns:
            Tuple of (entry_price, stop_price, target_price)

        Note:
            - For strategies without stops/targets, return (entry, nan, nan)
            - Return (nan, nan, nan) if entry should be skipped (e.g., max stop % exceeded)
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        idx: int,
        direction: str,
        entry_price: float,
        stop_price: float,
        equity: float
    ) -> float:
        """
        Calculate position size based on strategy rules.

        Args:
            idx: Current bar index
            direction: "long" or "short"
            entry_price: Entry price
            stop_price: Stop price (may be nan)
            equity: Current equity

        Returns:
            Position size in contracts/units

        Examples:
            - Risk-based: risk_cash = equity * 0.02; size = risk_cash / stop_distance
            - Fixed %: size = equity * 0.5 / entry_price
            - Fixed contracts: size = 1.0
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        idx: int,
        position_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[float], str]:
        """
        Check if exit conditions are met.

        Args:
            idx: Current bar index
            position_info: Dict with keys:
                - 'direction': 1 (long) or -1 (short)
                - 'entry_price': float
                - 'stop_price': float (may be nan)
                - 'target_price': float (may be nan)
                - 'entry_idx': int (bar index of entry)
                - 'size': float

        Returns:
            Tuple of (should_exit, exit_price, reason)
                - should_exit: bool
                - exit_price: float or None (if should_exit=False)
                - reason: str ("stop", "target", "trailing", "max_days", etc.)

        Note:
            Called every bar while in position.
        """
        pass

    @abstractmethod
    def _prepare_data(
        self,
        df: pd.DataFrame,
        cached_data: Optional[Dict] = None
    ) -> None:
        """
        Prepare all indicators needed for strategy.

        Args:
            df: OHLCV DataFrame
            cached_data: Pre-computed indicators (for optimization)
                         None for single backtest (compute on-the-fly)

        Responsibilities:
            - Store reference to df and extract OHLC arrays
            - Compute or retrieve indicators from cache
            - Store indicators in instance variables

        Example:
            if cached_data:
                # Use pre-computed values from optimizer
                self._ma = cached_data['ma_cache'][(self.params['maType'], self.params['maLength'])]
            else:
                # Compute on-the-fly
                from indicators import get_ma
                self._ma = get_ma(df['Close'], self.params['maType'], self.params['maLength']).to_numpy()
        """
        pass

    @classmethod
    @abstractmethod
    def get_param_definitions(cls) -> Dict[str, Dict]:
        """
        Define all strategy parameters for UI and optimization.

        Returns:
            Dict with structure:
            {
                'camelCaseName': {  # Use camelCase for API/frontend compatibility
                    'type': 'int' | 'float' | 'bool' | 'categorical',
                    'default': value,
                    'min': value,  # for int/float
                    'max': value,  # for int/float
                    'step': value,  # optional
                    'choices': [...],  # for categorical
                    'description': 'Parameter description'
                }
            }

        Example:
            return {
                'maLength': {  # camelCase key
                    'type': 'int',
                    'default': 45,
                    'min': 10,
                    'max': 200,
                    'step': 5,
                    'description': 'Moving average period'
                },
                'maType': {  # camelCase key
                    'type': 'categorical',
                    'choices': ['SMA', 'EMA', 'HMA'],
                    'default': 'EMA',
                    'description': 'Moving average type'
                }
            }
        """
        pass

    # ════════════════════════════════════════════════════════════════
    # CONCRETE METHODS (implemented in base class)
    # ════════════════════════════════════════════════════════════════

    def simulate(
        self,
        df: pd.DataFrame,
        cached_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main simulation entry point.

        This is the SINGLE method called for both:
        - Single backtest (cached_data=None)
        - Optimization (cached_data={...})

        Args:
            df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            cached_data: Pre-computed indicators (optional, for optimization)

        Returns:
            Dict with keys:
                - net_profit_pct: float
                - max_drawdown_pct: float
                - total_trades: int
                - winning_trades: int
                - losing_trades: int
                - sharpe_ratio: float
                - profit_factor: float
                - trades: List[Dict]
                - equity_curve: List[float]
        """
        # Prepare data and indicators
        self._prepare_data(df, cached_data)

        # Run simulation loop
        return self._run_simulation(df)

    def _run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generic simulation loop.

        Calls strategy methods (should_long, should_short, etc.)
        at appropriate times.

        Can be overridden if strategy needs custom loop logic,
        but default implementation should work for most strategies.
        """
        # Initialize state
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = math.nan
        stop_price = math.nan
        target_price = math.nan
        entry_idx = 0
        position_size = 0.0
        entry_commission = 0.0

        equity = 100.0  # Starting equity (%)
        equity_curve = []
        trades = []

        commission_rate = self.params.get('commission_rate', 0.0)

        # Main loop
        for idx in range(len(df)):
            current_close = self.close[idx]
            current_high = self.high[idx]
            current_low = self.low[idx]

            # ═══════════════════════════════════════════════════
            # EXIT LOGIC (if in position)
            # ═══════════════════════════════════════════════════
            if position != 0:
                position_info = {
                    'direction': position,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'target_price': target_price,
                    'entry_idx': entry_idx,
                    'size': position_size
                }

                should_exit, exit_price, reason = self.should_exit(idx, position_info)

                if should_exit:
                    # Calculate PnL
                    if position > 0:
                        gross_pnl = (exit_price - entry_price) * position_size
                    else:
                        gross_pnl = (entry_price - exit_price) * position_size

                    exit_comm = exit_price * position_size * commission_rate
                    net_pnl = gross_pnl - entry_commission - exit_comm

                    equity += net_pnl

                    # Record trade
                    trades.append({
                        'direction': 'long' if position > 0 else 'short',
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': position_size,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'reason': reason
                    })

                    # Reset position
                    position = 0
                    entry_price = math.nan
                    stop_price = math.nan
                    target_price = math.nan
                    position_size = 0.0
                    entry_commission = 0.0

            # ═══════════════════════════════════════════════════
            # REVERSAL LOGIC (for reversal strategies)
            # ═══════════════════════════════════════════════════
            if self.allows_reversal() and position != 0:
                reverse_to = None

                if position > 0 and self.should_short(idx):
                    reverse_to = 'short'
                elif position < 0 and self.should_long(idx):
                    reverse_to = 'long'

                if reverse_to:
                    # Close current position first
                    exit_price = current_close
                    if position > 0:
                        gross_pnl = (exit_price - entry_price) * position_size
                    else:
                        gross_pnl = (entry_price - exit_price) * position_size

                    exit_comm = exit_price * position_size * commission_rate
                    net_pnl = gross_pnl - entry_commission - exit_comm
                    equity += net_pnl

                    trades.append({
                        'direction': 'long' if position > 0 else 'short',
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': position_size,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'reason': 'reversal'
                    })

                    # Open reverse position immediately
                    new_entry, new_stop, new_target = self.calculate_entry(idx, reverse_to)
                    if not math.isnan(new_entry):
                        new_size = self.calculate_position_size(idx, reverse_to, new_entry, new_stop, equity)
                        if new_size > 0:
                            position = 1 if reverse_to == 'long' else -1
                            entry_price = new_entry
                            stop_price = new_stop
                            target_price = new_target
                            position_size = new_size
                            entry_idx = idx
                            entry_commission = entry_price * position_size * commission_rate
                            equity -= entry_commission

            # ═══════════════════════════════════════════════════
            # ENTRY LOGIC (if flat)
            # ═══════════════════════════════════════════════════
            if position == 0:
                direction = None

                if self.should_long(idx):
                    direction = 'long'
                elif self.should_short(idx):
                    direction = 'short'

                if direction:
                    entry, stop, target = self.calculate_entry(idx, direction)

                    if not math.isnan(entry):
                        size = self.calculate_position_size(idx, direction, entry, stop, equity)

                        if size > 0:
                            position = 1 if direction == 'long' else -1
                            entry_price = entry
                            stop_price = stop
                            target_price = target
                            position_size = size
                            entry_idx = idx
                            entry_commission = entry_price * position_size * commission_rate
                            equity -= entry_commission

            # ═══════════════════════════════════════════════════
            # EQUITY TRACKING
            # ═══════════════════════════════════════════════════
            current_equity = equity
            if position > 0:
                unrealized_pnl = (current_close - entry_price) * position_size
                current_equity += unrealized_pnl
            elif position < 0:
                unrealized_pnl = (entry_price - current_close) * position_size
                current_equity += unrealized_pnl

            equity_curve.append(current_equity)

        # Calculate final metrics
        net_profit_pct = equity - 100.0
        max_dd = self._calculate_max_drawdown(equity_curve)

        winning_trades = sum(1 for t in trades if t['net_pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['net_pnl'] <= 0)

        if losing_trades > 0:
            win_pnl = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
            loss_pnl = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] <= 0))
            profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 0.0
        else:
            profit_factor = 0.0

        sharpe = self._calculate_sharpe(equity_curve)

        return {
            'net_profit_pct': net_profit_pct,
            'max_drawdown_pct': max_dd,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'trades': trades,
            'equity_curve': equity_curve
        }

    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity_curve:
            return 0.0

        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        return abs(drawdown.min())

    @staticmethod
    def _calculate_sharpe(equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0

        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0

        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized

    # ════════════════════════════════════════════════════════════════
    # OPTIONAL METHODS (can be overridden)
    # ════════════════════════════════════════════════════════════════

    @classmethod
    def get_cache_requirements(
        cls,
        param_combinations: List[Dict]
    ) -> Dict:
        """
        Specify what indicators to pre-compute for optimization.

        Args:
            param_combinations: List of parameter dicts that will be tested

        Returns:
            Dict specifying cache requirements:
            {
                'ma_types_and_lengths': [(type, length), ...],
                'needs_atr': True,
                'long_lp_values': [2, 5, 10],
                'short_lp_values': [2, 5, 10]
            }

        Default implementation returns empty dict (no caching).
        Override in subclass if caching needed.
        """
        return {}

    def allows_reversal(self) -> bool:
        """
        Does this strategy allow position reversal?

        Returns:
            True if strategy can reverse (long→short or short→long)
            False if strategy exits to flat before entering opposite direction

        Default: False (most strategies)
        Override in reversal strategies (e.g., S_03)
        """
        return False
```

---

## Step 3: Create strategy_registry.py

Create `/src/strategy_registry.py`:

```python
"""
Strategy Registry

Central registry for all trading strategies.
Provides strategy discovery, instantiation, and metadata.
"""

from typing import Dict, List, Type, Any
from strategies.base_strategy import BaseStrategy


class StrategyRegistry:
    """
    Registry for all available trading strategies.

    Responsibilities:
    - Map strategy IDs to strategy classes
    - Instantiate strategies with parameters
    - Provide strategy metadata for UI
    """

    # Strategy registry (will be populated as strategies are added)
    _strategies: Dict[str, Type[BaseStrategy]] = {
        # Will add: "s01_trailing_ma": S01TrailingMA,
        # Will add: "s03_reversal": S03Reversal,
    }

    @classmethod
    def get_strategy_class(cls, strategy_id: str) -> Type[BaseStrategy]:
        """
        Get strategy class by ID.

        Args:
            strategy_id: Strategy identifier (e.g., "s01_trailing_ma")

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy_id not found
        """
        if strategy_id not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: '{strategy_id}'. "
                f"Available strategies: {available}"
            )
        return cls._strategies[strategy_id]

    @classmethod
    def get_strategy_instance(
        cls,
        strategy_id: str,
        params: Dict[str, Any]
    ) -> BaseStrategy:
        """
        Create strategy instance with parameters.

        Args:
            strategy_id: Strategy identifier
            params: Strategy parameters

        Returns:
            Initialized strategy instance
        """
        strategy_class = cls.get_strategy_class(strategy_id)
        return strategy_class(params)

    @classmethod
    def list_strategies(cls) -> List[Dict[str, Any]]:
        """
        List all available strategies with metadata.

        Returns:
            List of dicts with structure:
            [
                {
                    'id': 's01_trailing_ma',
                    'name': 'S_01 TrailingMA v26',
                    'version': '26',
                    'param_definitions': {...}
                },
                ...
            ]
        """
        strategies = []

        for strategy_id, strategy_class in cls._strategies.items():
            strategies.append({
                'id': strategy_id,
                'name': strategy_class.STRATEGY_NAME,
                'version': strategy_class.VERSION,
                'param_definitions': strategy_class.get_param_definitions()
            })

        return strategies

    @classmethod
    def register_strategy(
        cls,
        strategy_id: str,
        strategy_class: Type[BaseStrategy]
    ) -> None:
        """
        Register a new strategy (for dynamic registration).

        Args:
            strategy_id: Unique strategy identifier
            strategy_class: Strategy class (must inherit from BaseStrategy)

        Raises:
            ValueError: If strategy_id already registered or class doesn't inherit from BaseStrategy
        """
        if strategy_id in cls._strategies:
            raise ValueError(f"Strategy '{strategy_id}' already registered")

        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy class must inherit from BaseStrategy")

        cls._strategies[strategy_id] = strategy_class
```

---

## Testing

### Test 1: Import Test

```bash
cd src
python -c "
from strategies.base_strategy import BaseStrategy
from strategy_registry import StrategyRegistry

print('✅ BaseStrategy imported successfully')
print('✅ StrategyRegistry imported successfully')
"
```

### Test 2: ABC Enforcement Test

```python
# Test that BaseStrategy cannot be instantiated
from strategies.base_strategy import BaseStrategy

try:
    strategy = BaseStrategy({})
    print('❌ ERROR: BaseStrategy should not be instantiable!')
except TypeError as e:
    print(f'✅ ABC enforcement works: {e}')
```

### Test 3: Registry Test

```python
from strategy_registry import StrategyRegistry

# Should return empty list (no strategies registered yet)
strategies = StrategyRegistry.list_strategies()
print(f'Registered strategies: {len(strategies)}')  # Should be 0

# Should raise error for unknown strategy
try:
    StrategyRegistry.get_strategy_class('unknown')
    print('❌ Should have raised ValueError')
except ValueError as e:
    print(f'✅ Error handling works: {e}')
```

### Test 4: Reference Test

**No changes to existing code, so S_01 should still work:**

```bash
cd src
python run_backtest.py --csv ../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
```

Results should match baseline exactly.

---

## Commit

```bash
git add src/strategies/ src/strategy_registry.py
git commit -m "Phase 2: Create base strategy contract

- Create BaseStrategy ABC with all abstract methods
- Implement generic _run_simulation() loop
- Create StrategyRegistry for strategy management
- Add support for reversal strategies (allows_reversal)
- No changes to existing backtest code yet

Files added:
- src/strategies/__init__.py
- src/strategies/base_strategy.py (400+ lines)
- src/strategy_registry.py"
```

---

## Next Steps

Proceed to **Phase 3: Extract S_01 to Strategy Module**
- See `migration_prompt_3.md`
- See `migration_checklist.md` - Phase 3

---

**End of Migration Prompt 2**
