from __future__ import annotations

from typing import Any, Dict

import altair as alt

alt.data_transformers.disable_max_rows()


def to_vega_spec(chart: alt.Chart) -> Dict[str, Any]:
    """Convert an Altair chart into a Vega-Lite spec dict (JSON-serializable)."""
    return chart.to_dict()

