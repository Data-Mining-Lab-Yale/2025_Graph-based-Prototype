# text_to_graph/Text2Graph_2_srl.py
import networkx as nx
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

class SRLGraphBuilder:
    def __init__(self):
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"
        )

    def build_graph(self, text: str) -> nx.DiGraph:
        output = self.predictor.predict(sentence=text)
        words = output["words"]
        verbs = output["verbs"]  # list of dicts with "verb" and "tags"

        G = nx.DiGraph()
        for i, word in enumerate(words):
            G.add_node(i, label=word)

        for verb_frame in verbs:
            tags = verb_frame["tags"]
            predicate_index = None

            for i, tag in enumerate(tags):
                if tag == "B-V":
                    predicate_index = i
                    predicate_node = f"V:{words[i]}"
                    G.add_node(predicate_node, label=words[i])
                    break

            if predicate_index is None:
                continue

            for i, tag in enumerate(tags):
                if tag.startswith("B-ARG") or tag.startswith("B-AM"):
                    arg_label = tag[2:]
                    G.add_edge(f"V:{words[predicate_index]}", i, role=arg_label)

        return G
