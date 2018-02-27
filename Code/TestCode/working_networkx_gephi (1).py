#!/usr/bin/python3

import numpy as np
import scipy.sparse as spsp
import networkx as nx
import numpy.linalg

# Read .mtx file
def read_file (filename):
		rows, cols= np.loadtxt (filename, skiprows=18, unpack=True)
		data = np.ones (len (rows))
		return ((rows, cols, data))

#Create sparse matrix
def create_sp_matrix (data, rows, cols):
		a = int (max (rows)) + 1
		b = int (max (cols)) + 1
		print (a, b)
		sparse_matrix = spsp.coo_matrix ((data, (rows, cols)), shape=(a, b))
		print (sparse_matrix)
		return (sparse_matrix)

#Create numpy array
def convert_np_arr (sparse_matrix):
		numpy_array = sparse_matrix.toarray ()
		return (numpy_array)


#Create graph using numpy array
def create_graph (numpy_array):
		G = nx.from_numpy_matrix (numpy_array)
		return (G)

#Write to file and view in gephi
def write_file (G):
		nx.write_graphml(G,'new.graphml')

def cal_eig (G):
		L = nx.normalized_laplacian_matrix(G)
		print (L)
		e = numpy.linalg.eigvals(L.A)
		print (e)
		print("Largest eigenvalue:", max(e))
		print("Smallest eigenvalue:", min(e))
		#plt.hist(e,bins=100) # histogram with 100 bins
		#plt.xlim(0,2)  # eigenvalues between 0 and 2
		#plt.show()

(rows, cols, data) = read_file ("wb-cs-stanford.mtx")
sparse_matrix = create_sp_matrix (data, rows, cols)
numpy_arr = convert_np_arr (sparse_matrix)
G = create_graph (numpy_arr)
write_file (G)
cal_eig (G)
