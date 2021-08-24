import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
import numpy as np
from pathlib import PurePath

def plot_data(data, title=None):
	fig, axes = plt.subplots(1,1)
	axes.imshow(data.T, aspect='equal', origin='lower',
		interpolation='none', norm=Normalize(-18,2))
	if title is not None:
		plt.title(title)
	plt.show()

	
if __name__ == '__main__':
	path = sys.argv[1]
	name = sys.argv[2]

	path = PurePath(path)

	if path.suffix == '.pt':
		with open(path, "rb") as file:
			spec = torch.load(file)
	elif path.suffix == '.npy':
		spec = np.load(path)
	else:
		raise ValueError('Unknown file extension')

	plot_data(spec, name)