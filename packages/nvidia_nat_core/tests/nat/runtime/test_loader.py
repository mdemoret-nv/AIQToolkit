# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_entrypoints


class FakeEntryPoints:

    def __init__(self):
        self.selected_groups: list[str] = []

    def select(self, group: str) -> list[str]:
        self.selected_groups.append(group)
        return [group]


def setup_function():
    discover_entrypoints.cache_clear()


def teardown_function():
    discover_entrypoints.cache_clear()


def test_front_end_discovery_includes_generic_plugin_entrypoints():
    """Front ends can be registered through either the specific or generic plugin entry point group."""
    fake_entry_points = FakeEntryPoints()

    with patch("nat.runtime.loader.importlib.metadata.entry_points", return_value=fake_entry_points):
        discovered = discover_entrypoints(PluginTypes.FRONT_END)

    assert fake_entry_points.selected_groups == ["nat.front_ends", "nat.plugins"]
    assert discovered == ["nat.front_ends", "nat.plugins"]


def test_config_object_discovery_deduplicates_generic_plugin_entrypoints():
    """Combined component and front-end discovery should not load nat.plugins twice."""
    fake_entry_points = FakeEntryPoints()

    with patch("nat.runtime.loader.importlib.metadata.entry_points", return_value=fake_entry_points):
        discover_entrypoints(PluginTypes.CONFIG_OBJECT)

    assert fake_entry_points.selected_groups.count("nat.plugins") == 1
