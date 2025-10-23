# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import importlib.metadata
import logging
import time
from contextlib import asynccontextmanager
from enum import IntFlag
from enum import auto
from functools import lru_cache
from functools import reduce

from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.config import Config
from nat.runtime.session import SessionManager
from nat.utils.data_models.schema_validator import validate_schema
from nat.utils.debugging_utils import is_debugger_attached
from nat.utils.io.yaml_tools import yaml_load
from nat.utils.type_utils import StrPath

logger = logging.getLogger(__name__)


class PluginTypes(IntFlag):
    """
    Flag enumeration for different types of NAT plugins.

    Each flag represents a different category of plugin that can be discovered
    and loaded by the NAT runtime system.
    """

    COMPONENT = auto()
    """A plugin that is a component of the workflow. This includes tools, LLMs, retrievers, etc."""

    FRONT_END = auto()
    """A plugin that provides a front end interface for the workflow. This includes FastAPI, Gradio, etc."""

    EVALUATOR = auto()
    """A plugin that provides evaluation capabilities for the workflow. This includes RAGAS, SWE-bench, etc."""

    AUTHENTICATION = auto()
    """A plugin that provides API authentication for the workflow. This includes OAuth2, API Key, etc."""

    REGISTRY_HANDLER = auto()
    """A plugin that handles registry operations for the workflow."""

    # Convenience flags for groups of plugin types
    CONFIG_OBJECT = COMPONENT | FRONT_END | EVALUATOR | AUTHENTICATION
    """Any plugin that can be specified in the NAT configuration file."""

    ALL = COMPONENT | FRONT_END | EVALUATOR | REGISTRY_HANDLER | AUTHENTICATION
    """All available plugin types combined."""


def load_config(config_file: StrPath | Config) -> Config:
    """
    Load and validate a NAT configuration file.

    This is the primary entry point for loading a NAT configuration file. It ensures that all
    necessary plugins are discovered and registered, then validates the configuration file
    against the Config schema.

    Parameters
    ----------
    config_file : StrPath | Config
        Either the path to a YAML configuration file to load, or an already loaded Config object.
        If a Config object is provided, it will be returned as-is without further processing.

    Returns
    -------
    Config
        The validated Config object loaded from the configuration file or passed through
        if already a Config instance.

    Raises
    ------
    ValidationError
        If the configuration file does not conform to the expected schema.
    FileNotFoundError
        If the specified configuration file path does not exist.
    """

    # Ensure all of the plugins are loaded
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    if isinstance(config_file, Config):
        return config_file

    config_yaml = yaml_load(config_file)

    # Validate configuration adheres to NAT schemas
    validated_nat_config = validate_schema(config_yaml, Config)

    return validated_nat_config


@asynccontextmanager
async def load_workflow(config_file: StrPath | Config, max_concurrency: int = -1):
    """
    Load a NAT workflow configuration and create a session manager for executing workflows.

    This is the primary entry point for running NAT workflows. It loads the configuration,
    builds the workflow, and yields a SessionManager that can be used to execute workflow
    invocations with proper resource management and concurrency control.

    Parameters
    ----------
    config_file : StrPath | Config
        Either the path to a YAML configuration file to load, or an already loaded Config object.
    max_concurrency : int, optional
        The maximum number of parallel workflow invocations to support. Specifying 0 or -1
        will allow unlimited concurrent invocations. Default is -1.

    Yields
    ------
    SessionManager
        A session manager instance that can be used to execute workflow invocations.
        The session manager handles resource allocation, concurrency control, and cleanup.

    Examples
    --------
    >>> async with load_workflow("config.yml", max_concurrency=10) as session:
    ...     result = await session.invoke({"query": "Hello world"})
    Notes
    -----
    This function is an async context manager and must be used with `async with` syntax.
    The workflow resources will be automatically cleaned up when exiting the context.
    """

    # Load the config object
    config = load_config(config_file)

    # Must yield the workflow function otherwise it cleans up
    async with WorkflowBuilder.from_config(config=config) as workflow:

        yield SessionManager(workflow.build(), max_concurrency=max_concurrency)


@lru_cache
def discover_entrypoints(plugin_type: PluginTypes):
    """
    Discover and return entry points for the specified plugin types.

    This function searches for plugins that have been registered via Python entry point
    groups. Entry points allow packages to advertise their plugins so they can be
    automatically discovered by the NAT runtime system.

    The function searches across multiple entry point groups for backward compatibility,
    including both legacy 'aiq.*' groups and current 'nat.*' groups.

    Parameters
    ----------
    plugin_type : PluginTypes
        A flag or combination of flags specifying which plugin types to discover.
        Can be combined using bitwise OR operations (e.g., PluginTypes.COMPONENT | PluginTypes.FRONT_END).

    Returns
    -------
    list[importlib.metadata.EntryPoint]
        A list of entry points matching the specified plugin types. Each entry point
        contains information about the plugin module and name.

    Notes
    -----
    This function is cached using @lru_cache to improve performance on repeated calls
    with the same plugin_type parameter.

    The function searches the following entry point groups based on plugin type:
    - COMPONENT: "aiq.plugins", "aiq.components", "nat.plugins", "nat.components"
    - FRONT_END: "aiq.front_ends", "nat.front_ends"
    - REGISTRY_HANDLER: "aiq.registry_handlers", "nat.registry_handlers"
    - EVALUATOR: "aiq.evaluators", "nat.evaluators"
    - AUTHENTICATION: "aiq.authentication_providers", "nat.authentication_providers"
    """

    entry_points = importlib.metadata.entry_points()

    plugin_groups = []

    # Add the specified plugin type to the list of groups to load
    # The aiq entrypoints are intentionally left in the list to maintain backwards compatibility.
    if (plugin_type & PluginTypes.COMPONENT):
        plugin_groups.extend(["aiq.plugins", "aiq.components", "nat.plugins", "nat.components"])
    if (plugin_type & PluginTypes.FRONT_END):
        plugin_groups.extend(["aiq.front_ends", "nat.front_ends"])
    if (plugin_type & PluginTypes.REGISTRY_HANDLER):
        plugin_groups.extend(["aiq.registry_handlers", "nat.registry_handlers"])
    if (plugin_type & PluginTypes.EVALUATOR):
        plugin_groups.extend(["aiq.evaluators", "nat.evaluators"])
    if (plugin_type & PluginTypes.AUTHENTICATION):
        plugin_groups.extend(["aiq.authentication_providers", "nat.authentication_providers"])

    # Get the entry points for the specified groups
    nat_plugins = reduce(lambda x, y: list(x) + list(y), [entry_points.select(group=y) for y in plugin_groups])

    return nat_plugins


@lru_cache
def get_all_entrypoints_distro_mapping() -> dict[str, str]:
    """
    Create a mapping from module paths to their corresponding distribution package names.

    This function builds a comprehensive mapping that associates Python module paths with
    the names of the distribution packages (wheels/eggs) that contain them. This is useful
    for determining which package provides a particular module or plugin.

    The mapping includes all module prefixes for NAT entry points, allowing for efficient
    lookup of package information given a module path.

    Returns
    -------
    dict[str, str]
        A dictionary mapping module path prefixes to distribution package names.
        Keys are module paths (e.g., "nat.plugins.llama_index") and values are
        package names (e.g., "nvidia-nat-llama-index").
    Examples
    --------
    >>> mapping = get_all_entrypoints_distro_mapping()
    >>> mapping["nat.plugins.llama_index"]
    "nvidia-nat-llama-index"
    Notes
    -----
    This function is cached using @lru_cache to improve performance since the mapping
    is expensive to compute and typically doesn't change during runtime.

    The mapping includes all possible module prefixes, not just the full module paths.
    For example, if an entry point has module "nat.plugins.llama_index.llm", the mapping
    will include entries for "nat", "nat.plugins", "nat.plugins.llama_index", and
    "nat.plugins.llama_index.llm".
    """

    mapping = {}
    nat_entrypoints = discover_entrypoints(PluginTypes.ALL)
    for ep in nat_entrypoints:
        ep_module_parts = ep.module.split(".")
        current_parts = []
        for part in ep_module_parts:
            current_parts.append(part)
            module_prefix = ".".join(current_parts)
            mapping[module_prefix] = ep.dist.name

    return mapping


def discover_and_register_plugins(plugin_type: PluginTypes):
    """
    Discover and register plugins of the specified types into the global type registry.

    This function finds all plugins of the requested types that were registered via Python
    entry point groups, loads their modules, and registers them into the GlobalTypeRegistry.
    This makes the plugins available for use in NAT workflows.

    The function includes performance optimizations such as pausing registry hooks during
    bulk loading, and comprehensive error handling and logging for troubleshooting plugin
    loading issues.

    Parameters
    ----------
    plugin_type : PluginTypes
        A flag or combination of flags specifying which plugin types to discover and register.
        Can be combined using bitwise OR operations (e.g., PluginTypes.COMPONENT | PluginTypes.FRONT_END).
    Notes
    -----
    - Registry change hooks are paused during loading for better performance when loading
      many plugins simultaneously.
    - Import errors are logged as warnings but do not stop the loading process.
    - Other exceptions during plugin loading are logged as errors but do not stop the process.
    - Loading times are monitored and warnings are logged for slow-loading plugins to help
      identify performance issues.
    - The function is tolerant of plugin loading failures and will continue loading other
      plugins even if some fail.
    Raises
    ------
    This function does not raise exceptions for plugin loading failures. All errors are
    logged and the function continues processing remaining plugins.
    """

    # Get the entry points for the specified groups
    nat_plugins = discover_entrypoints(plugin_type)

    count = 0

    # Pause registration hooks for performance. This is useful when loading a large number of plugins.
    with GlobalTypeRegistry.get().pause_registration_changed_hooks():

        for entry_point in nat_plugins:
            try:
                logger.debug("Loading module '%s' from entry point '%s'...", entry_point.module, entry_point.name)

                start_time = time.time()

                entry_point.load()

                elapsed_time = (time.time() - start_time) * 1000

                logger.debug("Loading module '%s' from entry point '%s'...Complete (%f ms)",
                             entry_point.module,
                             entry_point.name,
                             elapsed_time)

                # Log a warning if the plugin took a long time to load. This can be useful for debugging slow imports.
                # The threshold is 300 ms if no plugins have been loaded yet, and 100 ms otherwise. Triple the threshold
                # if a debugger is attached.
                if (elapsed_time > (300.0 if count == 0 else 150.0) * (3 if is_debugger_attached() else 1)):
                    logger.debug(
                        "Loading module '%s' from entry point '%s' took a long time (%f ms). "
                        "Ensure all imports are inside your registered functions.",
                        entry_point.module,
                        entry_point.name,
                        elapsed_time)

            except ImportError:
                logger.warning("Failed to import plugin '%s'", entry_point.name, exc_info=True)
                # Optionally, you can mark the plugin as unavailable or take other actions

            except Exception:
                logger.exception("An error occurred while loading plugin '%s'", entry_point.name)

            finally:
                count += 1


# Compatibility alias
get_all_aiq_entrypoints_distro_mapping = get_all_entrypoints_distro_mapping
