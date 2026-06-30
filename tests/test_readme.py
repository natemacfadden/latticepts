# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

import pathlib
import re


def test_readme_python_snippets_execute():
    """Quickstart drift guard: every ```python block in the README must run,
    execed in sequence in a shared namespace (later blocks reuse earlier vars,
    as a reader copying them in order would)."""
    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.DOTALL)
    assert blocks, "no python blocks found in README"
    ns = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f"<README block {i}>", "exec"), ns)
