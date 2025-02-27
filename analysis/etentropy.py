# -*- mode: python -*-

import numpy

class etEntropy(object):
	def __init__(self):
		self.epsilon: float = None
		self.tau: int = None
		self.dimension: int = None
		self.num_sample: int = None
		self.tsdata = None
		self.num_data = None
		self.param_names = ('epsilon', 'tau', 'dimension', 'num_sample', 'tsdata', 'num_data')

	def __str__(self):
		return "epsilon={}  tau={}  dim={}  sample={}  num_data={}  tsdata={}".format(self.epsilon, self.tau, self.dimension, self.num_sample, self.num_data, self.tsdata)

	def keep_params(func):
		def wrapper(self, *args, **kwargs):
			param_store = {}
			for pname in self.param_names:
				param_store[pname] = getattr(self, pname)
			result = func(self, *args, **kwargs)
			for pname, pvalue in param_store.items():
				setattr(self, pname, pvalue)
			return result
		return wrapper

	def set_data(self, time_series_data):
		"""
		Input the data to calculator.

		Parameters
		----------
		time_series_data: array.array or list
			one dimensional time series data.
		"""
		data = time_series_data
		if not (isinstance(time_series_data, numpy.ndarray) or isinstance(time_series_data, list)):
			raise TypeError("expected list or array but given {}.".format(type(data)))
		self.tsdata = numpy.array(time_series_data)

	@staticmethod
	def embed_tsdata_to_coord_obsolete(strides, steps, time_series_data):
		"""
		Embed time series data to high dimensional coordinate

		Returns
		-------
		matrix of coordinate
			axis 0: value of each dimensions
			axis 1: time
		"""
		max_offset = (steps-1) * strides
		data = numpy.ndarray(shape=(steps, max_offset+len(time_series_data)), dtype=float)
		for i in range(steps):
			data[i,i*strides:i*strides+len(time_series_data)] = time_series_data
		return data[:, max_offset:-max_offset]

	@staticmethod
	def embed_tsdata_to_coord(strides, steps, time_series_data, max_length=None):
		"""
		Embed time series data to high dimensional coordinate
		Align the head of time series data when make matrix

		Returns
		-------
		matrix of coordinate
			axis 0: value of each dimensions
			axis 1: time
		"""
		max_offset = (steps-1) * strides
		if not max_length: # max_length is auto (use all data)
			max_length = len(time_series_data) - max_offset
		if max_length > len(time_series_data) - max_offset:
			raise RuntimeError("Couldn't configure matrix: max_length is too large")
		if max_length <= 0:
			raise RuntimeError("Couldn't configure matrix: not enough data")
		data = numpy.ndarray(shape=(steps, max_length), dtype=float)
		for i in range(steps):
			data[i] = time_series_data[i*strides:data.shape[1]+i*strides]
		return data

	@staticmethod
	def calc_distances(matrix_coord):
		"""
		Calculate distances of each data points

		Parameters
		----------
		matrix_coord
			axis 0: value of each dimensions
			axis 1: time
		"""
		mc = matrix_coord
		matdist = numpy.linalg.norm(mc[:,numpy.newaxis,:] - mc[:,:,numpy.newaxis], ord=numpy.inf, axis=0)
		# matdist = (mc[:,numpy.newaxis,:] - mc[:,:,numpy.newaxis]).abs().max(axis=0) # same result as above
		assert matdist[1,1] == 0
		assert matdist[0,1] == matdist[1,0]
		return matdist

	@staticmethod
	def calc_prob_nearness(matrix_distance, threshold):
		"""
		Calculate probability that a distance is lower than a threshold

		Returns
		-------
		array_prob
			Probability matrix (1-D)
		"""
		md = matrix_distance
		ts = threshold
		assert md.shape[0] == md.shape[1]
		assert ts > 0
		count = numpy.count_nonzero(md < ts, axis=0)
		assert numpy.any(count>=1)
		num_elements = md.shape[0]
		return count / num_elements

	@staticmethod
	def calc_correlation(array_probability):
		from numpy import log, sum, divide
		ap = array_probability
		r = len(array_probability)
		#return divide(sum(ap), r)
		return -divide(sum(log(ap)), r) # same as return -ap.log().sum().divide(r)

	def docalc_correlation_index(self):
		"""
		Calculate epsilon-tau entropy with set data.
		"""
		# check tsdata is exists
		if self.tsdata.any():
			RuntimeError("Time series data is not inputed")
		m_pgdata = self.embed_tsdata_to_coord(self.tau, self.dimension, self.tsdata, self.num_data)
		m_dist = self.calc_distances(m_pgdata)
		a_prob = self.calc_prob_nearness(m_dist, self.epsilon)
		if self.num_sample:
			numpy.random.seed(0)
			a_prob_selected = numpy.random.choice(a_prob, self.num_sample, replace=False) # sampling WITHOUT replacement
			#a_prob_selected = a_prob[:self.num_sample]
		else:
			a_prob_selected = a_prob
		return self.calc_correlation(a_prob_selected)

	@keep_params
	def docalc_cp_entropy(self):
		"""
		Calculate entropy approximately
		"""
		ci1 = self.docalc_correlation_index()
		self.dimension = self.dimension - 1
		ci2 = self.docalc_correlation_index()
		# Notice that tau is not realtime, its just offset of data to embed data to space.
		return (ci1 - ci2) / self.tau

	@classmethod
	def entropy(cls, data, epsilon, tau, dimension, num_sample):
		inst = cls(epsilon, tau, dimension, num_sample)
		inst.set_data(data)
		return inst.docalc()

if __name__ == '__main__':
	ent = etEntropy.entropy(data=numpy.random.random(10000), epsilon=0.95, tau=1, dimension=5, num_sample=200)
	print(ent)
