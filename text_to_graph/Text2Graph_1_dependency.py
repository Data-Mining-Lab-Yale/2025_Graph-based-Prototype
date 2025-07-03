# text_to_graph/dependency.py
import spacy
import networkx as nx
from text_to_graph.Text2Graph_1_base import GraphBuilder

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class DependencyGraphBuilder(GraphBuilder):
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def build_graph(self, text: str) -> nx.DiGraph:
        doc = self.nlp(text)
        G = nx.DiGraph()

        for token in doc:
            G.add_node(token.i, label=token.text, pos=token.pos_)
            if token.head.i != token.i:
                G.add_edge(token.head.i, token.i, dep=token.dep_)

        return G
