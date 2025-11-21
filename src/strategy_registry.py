"""
Strategy Registry

Central registry for all trading strategies.
Provides strategy discovery, instantiation, and metadata.
"""

from typing import Any, Dict, List, Type

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
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: '{strategy_id}'. Available strategies: {available}"
            )
        return cls._strategies[strategy_id]

    @classmethod
    def get_strategy_instance(
        cls,
        strategy_id: str,
        params: Dict[str, Any],
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
        strategies: List[Dict[str, Any]] = []

        for strategy_id, strategy_class in cls._strategies.items():
            strategies.append(
                {
                    "id": strategy_id,
                    "name": strategy_class.STRATEGY_NAME,
                    "version": strategy_class.VERSION,
                    "param_definitions": strategy_class.get_param_definitions(),
                }
            )

        return strategies

    @classmethod
    def register_strategy(
        cls,
        strategy_id: str,
        strategy_class: Type[BaseStrategy],
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
            raise ValueError("Strategy class must inherit from BaseStrategy")

        cls._strategies[strategy_id] = strategy_class
