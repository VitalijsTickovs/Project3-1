import itertools
import networkx as nx
import matplotlib.pyplot as plt



def initGraph(input):
    G = nx.DiGraph()
    G.add_node("root", name=None, pos=None, size=None, prob=None)
    # append root to queue
    allPoss = list(itertools.permutations(input))

    for pos in allPoss: # iterate through possibilities
        prev_ref = "root" # keep track of the previous reference to add edges
        for objTup in pos:
            # get reference for this node
            ref = objTup[0]+str(objTup[1][0])+str(objTup[1][1])+str(objTup[1][2])+"_"+str(prev_ref)
            # foundEqls = False
            # for n in G.nodes:
            #     if checkEq(n, ref):
            #         G.add_edge(prev_ref, n)
            #         foundEqls = True
            #         prev_ref = n
            edges_to_add = []
            foundEqls = False
            for n in G.nodes:
                if checkEq(n, ref):
                    edges_to_add.append((prev_ref, n))
                    foundEqls = True
                    prev_ref = n


            # Add all the new edges after iterating through the nodes
            for edge in edges_to_add:
                G.add_edge(*edge)
                # add node if not already added
            if (not foundEqls and not G.has_node(ref)):
                G.add_node(ref, name=objTup[0], pos=objTup[1], size = objTup[2])
                G.add_edge(prev_ref, ref)
            
            # add edge
                prev_ref = ref
            if(G.has_node(ref)):
                prev_ref = ref
            
    return G

def checkEq(nd1, nd2):
    splt1 = set(nd1.split("_"))
    splt2 = set(nd2.split("_"))

    return (nd1!=nd2) and (splt1 == splt2)

class Tree:
    def __init__(self, input):
        self.root = TreeNode(None, None)
        self.levels = [[]]*len(input+1)

        # append root to queue
        allPoss = list(itertools.permutations(input))
        self.levels[0].append(self.root)

        for pos in allPoss: # iterate through possibilities
            previousNode = self.root
            lvl = 1
            for objTup in pos:
                newNode = TreeNode(objTup, previousNode)
                self.levels[lvl].append(newNode)
                previousNode = newNode


class TreeNode:
    """A basic tree node class."""
    def __init__(self, tuple_in, parent): # for standard node
        if (parent!= None):
            self.name = tuple_in[0]
            self.pos = tuple_in[1] #tuple
            self.size = tuple_in[2] #tuple
        else:
            self.name = None
            self.pos = None
            self.size = None
        self.prob = 0
        self.children = []

        self.parent = parent # if 'None' then root

    def add_child(self, child_node):
        """Adds a child to this node."""
        self.children.append(child_node)

    def remove_child(self, child_node):
        """Removes a child from this node."""
        self.children = [child for child in self.children if child != child_node]

    def traverse(self):
        """Traverses the tree starting from this node."""
        nodes = [self]
        while nodes:
            current_node = nodes.pop()
            print(current_node.value)
            nodes.extend(current_node.children)


if __name__ == "__main__":
    configuration = [("Cup", (0,0,0), (1,1)), 
                     ("Crate", (-3,1,5), (3,3)),
                     ("Feeder", (9,9,9), (8,8))]
    graph = initGraph(configuration)
    nx.draw_planar(graph, with_labels = True, font_size=10)
    plt.savefig("graph_vis.png")
