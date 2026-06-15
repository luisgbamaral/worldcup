"""World Cup analytics — international football data (1872–2026).

Reusable loaders, feature builders and visualizations for the
exploratory analysis and the 2026 World Cup forecasting experiments.
"""
from . import clean, config, data, elo, features, lookups, update, viz

__all__ = ["clean", "config", "data", "elo", "features", "lookups", "update", "viz"]
__version__ = "0.1.0"
