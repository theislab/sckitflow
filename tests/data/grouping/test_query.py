from collections.abc import Collection
from typing import Any

import pytest

from sc_flow.core.data.grouping import QueryFactory

invalid_level_name = "invalid_level"


class TestQueryFactory:
    def _is_valid_level_query_dict(
        self, factory: QueryFactory, level_name: str, query_dict: dict[str, Any], allow_missing: bool
    ) -> None:
        if len(query_dict) != len(factory.registry[level_name]) and not allow_missing:
            return False
        if not all(k in factory.registry[level_name] for k in query_dict.keys()):
            return False
        return True

    @pytest.mark.parametrize("values", [(0.0, 0.0), (0.0,)])
    @pytest.mark.parametrize("level_name", ["level0", invalid_level_name])
    def test_query_tuple_to_dict(
        self,
        registry: dict[str, tuple[str, ...] | None],
        values: tuple[Any],
        level_name: str,
    ):
        factory = QueryFactory(registry)
        if level_name not in factory.registry_keys:
            with pytest.raises(KeyError):
                query_dict = factory.query_tuple_to_dict(registry, values, level_name)
            return None
        if len(values) != len(registry[level_name]):
            with pytest.raises(ValueError):
                query_dict = factory.query_tuple_to_dict(registry, values, level_name)
            return None
        query_dict = factory.query_tuple_to_dict(registry, values, level_name)
        is_valid = self._is_valid_level_query_dict(factory, level_name, query_dict, False)
        assert is_valid

    @pytest.mark.parametrize("level_name", ["level0", "level1", invalid_level_name])
    @pytest.mark.parametrize("reference", [None, ["level0", "level1"]])
    def test_verify_valid_level_names(
        self, registry: dict[str, tuple[str, ...] | None], level_name: str, reference: Collection[str] | None
    ) -> None:
        factory = QueryFactory(registry)
        if level_name not in factory.registry_keys:
            with pytest.raises(KeyError):
                factory.verify_valid_level_name(level_name, reference=reference)
            return None
        factory.verify_valid_level_name(level_name, reference=reference)

    @pytest.mark.parametrize("level_name", ["level0", "level1"])
    @pytest.mark.parametrize(
        "query_dict",
        [
            {"level0-A": 0.0},
            {"level0-B": 0.0},
            {"level0-A": 0.0, "level0-B": 0.0},
            {"level1-A": 0.0},
            {"level1-B": 0.0},
            {"level1-A": 0.0, "level1-B": 0.0},
        ],
    )
    def test_verify_level_query_dict(
        self,
        registry: dict[str, tuple[str, ...] | None],
        level_name: str,
        query_dict: dict[str, Any],
    ) -> None:
        factory = QueryFactory(registry)
        is_valid_query_dict = self._is_valid_level_query_dict(factory, level_name, query_dict, True)
        if not is_valid_query_dict:
            with pytest.raises(ValueError):
                factory.verify_level_query_dict(level_name, query_dict)
            return None
        factory.verify_level_query_dict(level_name, query_dict)

    @pytest.mark.parametrize(
        "query_dict",
        [
            {"level0": {"level0-A": 0.0}},
            {"level0": {"level0-B": 0.0}},
            {"level0": {"level0-A": 0.0, "level0-B": 0.0}},
            {"level1": {"level1-A": 0.0}},
            {"level1": {"level1-B": 0.0}},
            {"level1": {"level1-A": 0.0, "level1-B": 0.0}},
            {invalid_level_name: {"level1-A": 0.0, "level1-B": 0.0}},
        ],
    )
    def test_verify_query_dict(
        self,
        registry: dict[str, tuple[str, ...] | None],
        query_dict: dict[str, dict[str, Any]],
    ) -> None:
        factory = QueryFactory(registry)
        invalid_levels = set(query_dict.keys()) - set(factory.registry)
        if len(invalid_levels) > 0:
            with pytest.raises(ValueError):
                factory.verify_query_dict(query_dict)
            return None
        valid_levels = [
            self._is_valid_level_query_dict(factory, level, qdict, True) for level, qdict in query_dict.items()
        ]
        if not all(valid_levels):
            with pytest.raises(KeyError):
                factory.verify_query_dict(query_dict)
            return None
        factory.verify_query_dict(query_dict)

    @pytest.mark.parametrize(
        "level_name",
        [
            "level0",
        ],
    )
    @pytest.mark.parametrize("query_dict", [{"level0-A": 0.0}, {"level0-A": 0.0, "level0-B": 0.0}])
    def test_prepare_partial_level_query_dict(
        self,
        registry: dict[str, tuple[str, ...] | None],
        level_name: str,
        query_dict: dict[str, Any],
    ) -> dict[str, Any]:
        factory = QueryFactory(registry)
        query_dict = factory.prepare_partial_level_query_dict(level_name, query_dict)
        is_valid_query_dict = self._is_valid_level_query_dict(factory, level_name, query_dict, False)
        assert is_valid_query_dict

    @pytest.mark.parametrize(
        "query_dict",
        [
            {"levelA": {"levelA-0": 0.0}},
            {"levelB": {"levelB-0": 0.0}},
            {"levelA": {"levelA-0": 0.0}, "levelB": {"levelB-0": 0.0}},
        ],
    )
    def test_prepare_partial_query_dict(
        self,
        registry: dict[str, tuple[str, ...] | None],
        query_dict: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        factory = QueryFactory(registry)
        query_dict = factory.prepare_partial_query_dict(query_dict)
        missing_levels = set(factory.registry) - set(query_dict.keys())
        assert len(missing_levels) == 0
        is_valid_query = [
            self._is_valid_level_query_dict(factory, level, ldict, False) for level, ldict in query_dict.items()
        ]
        assert all(is_valid_query)
