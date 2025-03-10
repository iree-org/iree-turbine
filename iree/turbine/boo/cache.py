# Abstract Cache-Manager implementation

from abc import ABC, abstractmethod


class CacheManager(ABC):
    @abstractmethod
    def get_hash(self, *args, **kwargs):
        """Should implement some kernel hashing algorithm"""
        ...

    @abstractmethod
    def store_kernel_to_file(self, *args, **kwargs):
        """Should store a kernel to a file"""
        ...

    @abstractmethod
    def load_kernel_from_file(self, *args, **kwargs):
        """Should load a kernel from a file"""
        ...
