"""Trading strategies module"""
from .base_strategy import BaseStrategy
from .s01_trailing_ma import S01TrailingMA

__all__ = ["BaseStrategy", "S01TrailingMA"]
