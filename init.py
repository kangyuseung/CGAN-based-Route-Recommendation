import pandas as pd
import stellargraph as sg
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec

def HIN_graph(data):
    #HIN Modeling
    #node
    route_node=data['ROUTE_ID']
    place_node=data['PLACE_ID']
    user_node=data['USER_ID']

    route_node = route_node.drop_duplicates()
    place_node = place_node.drop_duplicates()
    user_node = user_node.drop_duplicates()

    route_node_ids=pd.DataFrame(route_node)
    place_node_ids=pd.DataFrame(place_node) 
    user_node_ids=pd.DataFrame(user_node) 

    route_node_ids.set_index('ROUTE_ID', inplace=True)
    place_node_ids.set_index('PLACE_ID', inplace=True)
    user_node_ids.set_index('USER_ID', inplace=True)

    #edge
    user_route_edge = data[['USER_ID', 'ROUTE_ID']]
    user_route_edge.columns = ['source', 'target']

    route_place_edge = data[['ROUTE_ID', 'PLACE_ID']]
    route_place_edge.columns = ['source','target']

    start=len(user_route_edge)
    route_place_edge.index=range(start, start+len(route_place_edge))

    g=sg.StellarDiGraph(nodes={'user' : user_node_ids, 'route' : route_node_ids, 'place' : place_node_ids}, 
                        edges={'user_route' : user_route_edge, 'route_place' : route_place_edge})

    print(g.info())
    return g

def HIN_embedding(g):
    #HIN Embedding
    walk_length = 50
    metapaths = [["user", "route", "place", "route", "user"], ["user", "route", "user"]]

    rw = UniformRandomMetaPathWalk(g)

    walks = rw.run(
        nodes=list(g.nodes()),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
        seed=42
    )

    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, vector_size=64, window=5, min_count=0, sg=1, epochs=5)

    return model