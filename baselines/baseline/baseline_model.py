import random
import networkx as nx
from configuration import Configuration


class BaselineModel:
    def __init__(self, G=None):
        self.G = G
        self.yolo_node = "root"

    def make_predictions(self, nodes):
        previous_node = nodes[0]
        hit_count = 0
        for current_node in nodes[1:]:
            predicted_node = self.predict(previous_node)
            print("Current node", current_node)
            print("Predicted node", predicted_node)
            print()

            if self.checkEq(current_node, predicted_node):
                hit_count += 1
            previous_node = current_node
        return hit_count/(len(nodes)-1)

    def checkEq(self,nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1!=nd2) and (splt1 == splt2)

    def yolo_predict(self, node):
        prediction = self.predict(node)
        return prediction

    def predict(self, node):
        out_edges = self.G.out_edges([node])
        if len(out_edges) > 1:
            return self.perform_step(out_edges)
        if len(out_edges) == 0:
            return node
        return list(out_edges)[0][1]

    def perform_step(self, out_edges):
        out_edge_probs = []
        out_edges = list(out_edges)
        for edge in out_edges:
            out_edge_probs.append(self.G.get_edge_data(edge[0], edge[1])['weight'])

        edge_choice = random.choices(out_edges, weights=out_edge_probs, k=1)
        return edge_choice[0][1]


if __name__ == "__main__":
    configuration = [("Cup", (0, 0, 0), (1, 1)),
                     ("Crate", (1, 1, 1), (3, 3)),
                     ("Feeder", (2, 2, 2), (8, 8)),
                     ("Gold", (0, 0, 0), (1, 1))]
    graph = Configuration()
    graph.initGraph(configuration)
    graph.assign_probs()

    baseline_model = BaselineModel(graph.get_graph())
    # print(baseline_model.predict("Cup000_root"))
    print(baseline_model.make_predictions(["root", "Crate111_root", "Gold000_Crate111_root", "Gold000_Crate111_Cup000_root","Gold000_Feeder222_Crate111_Cup000_root"]))
