from gymnasium.spaces import GraphInstance
import numpy as np

g1 = GraphInstance(
	nodes=np.random.randint(0, 10, size=(10,)),
	edge_links=np.random.randint(0, 10, size=(10, 2)),
	edges=None,
)

g2 = GraphInstance(
	nodes=np.random.randint(0, 5, size=(5,)),
	edge_links=np.random.randint(0, 5, size=(10, 2)),
	edges=None,
)

batch = [g1, g2]
batch = np.array(batch, dtype=object)

print(batch.shape)