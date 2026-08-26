"""
Regression test: pattern-summary percentages must print on a 0-100 scale.

``PatternInfo.percent_cases`` and ``PatternSummary.complete_cases_percent``
are stored on a 0-100 scale (``analyze_patterns`` multiplies by 100), but
``PatternSummary.__str__`` formatted them with ``:.1%``, which multiplies by
100 again — a 66% complete-case share printed as ``6600.0%``. The rendered
string is user-facing (it is the pattern-inspection output shown in the
package's worked examples), so the format is pinned here.
"""

import numpy as np

from pystatistics.mvnmle import analyze_patterns, pattern_summary


def test_summary_percentages_are_on_percent_scale():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 4))
    data[:34, 1] = np.nan          # 66 complete rows, 34 missing column 1

    rendered = str(pattern_summary(analyze_patterns(data), data.shape))

    assert "Complete cases: 66 (66.0%)" in rendered
    assert "66 cases (66.0%)" in rendered
    assert "6600" not in rendered
