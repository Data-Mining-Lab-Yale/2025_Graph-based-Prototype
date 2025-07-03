# text_to_graph/base.py
from abc import ABC, abstractmethod
import networkx as nx

class GraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, text: str) -> nx.DiGraph:
        pass
