import pickle  # nosec B403
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Sequence, Set, TypeAlias, Union

import pandas as pd

from src.utils.setuptools import RequiresSetupABCMeta, requires_setup

Key: TypeAlias = Union[str, int]
Record: TypeAlias = Dict[str, Any]
DataSource: TypeAlias = str


# region abstract.


class DfData(ABC):
    def __getitem__(self, key) -> Record:
        return self._getitem(key)

    def __contains__(self, key: Key) -> bool:
        return self._contains(key)

    def __len__(self) -> int:
        return self._len()

    def __hash__(self) -> int:
        return self._hash()

    @abstractmethod
    def _getitem(self, key) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def _contains(self, key: Key) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _len(self) -> bool:
        raise NotImplementedError

    def _hash(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_keys(self) -> Set[Key]:
        raise NotImplementedError


class DataReader(DfData, ABC, metaclass=RequiresSetupABCMeta):
    def setup(self) -> None:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _getitem(self, key) -> Record:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _contains(self, key: Key) -> bool:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _len(self) -> int:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def _hash(self) -> int:
        raise NotImplementedError

    @requires_setup
    @abstractmethod
    def get_keys(self) -> Set[Key]:
        raise NotImplementedError


class OfflineDataReader(DataReader, ABC):
    _keys: Set[Key] = None
    _computed_hash: int = None

    def __init__(self, data_source: DataSource):
        super().__init__()
        self._data_source = data_source

    def _contains(self, key: Key) -> bool:
        return key in self.get_keys()

    def _len(self) -> int:
        return len(self.get_keys())

    def _hash(self) -> int:
        if self._computed_hash is None:
            import hashlib

            sorted_keys = sorted(list(self.get_keys()))
            sorted_keys_bytes = str(sorted_keys).encode()
            self._computed_hash = int(
                hashlib.sha1(sorted_keys_bytes).hexdigest(), 16  # nosec B324
            )

        return self._computed_hash


class OnlineDataReader(DataReader, ABC):
    def _contains(self, key: Key) -> bool:
        return key in self.get_keys()

    def _len(self) -> int:
        return len(self.get_keys())

    def _hash(self) -> int:
        raise ValueError("Unhashable data reader")


# endregion


class PickleReader(OfflineDataReader):
    _data: Sequence = None

    def setup(self) -> None:
        with open(self._data_source, "rb") as stream_reader:
            self._data = pickle.load(stream_reader)  # nosec B301

        if isinstance(self._data, dict):
            self._keys = set(self._data.keys())
        else:
            self._keys = set(range(len(self._data)))

    def _getitem(self, key) -> Record:
        record = self._data[key]
        assert isinstance(record, dict)
        return record

    def get_keys(self) -> Set[Key]:
        return self._keys


class PandasReader(OfflineDataReader):
    _data: pd.DataFrame = None

    def __init__(self, data_source: DataSource, source_reading_method: Callable):
        super().__init__(data_source)
        self._data_source = data_source
        self._source_reading_method = source_reading_method

    def setup(self) -> None:
        self._data = self._source_reading_method(self._data_source)
        self._keys = set(self._data.index)

    def _getitem(self, key) -> Record:
        record = self._data.loc[key].to_dict()
        assert isinstance(record, dict)
        return record

    def get_keys(self) -> Set[Key]:
        return self._keys


