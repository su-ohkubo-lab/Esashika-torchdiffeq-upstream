# -*- mode: python -*-

import argparse
import numpy
import torch
import torch.nn as nn
import logging
import examples.odenet_mnist as tdeom

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	cmdargs = parser.parse_args()

class extractor(object):
	def __init__(self, config={}):
		if config.pop('conv_degrade', False):
			self.downsampling_layers = [
				nn.Conv2d(1, 64, 3, 1),
				tdeom.norm(64),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, 4, 4, 1),
			]
		else:
			self.downsampling_layers = [
				nn.Conv2d(1, 64, 3, 1),
				tdeom.norm(64),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, 4, 2, 1),
				tdeom.norm(64),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 64, 4, 2, 1),
			]
	
		self.feature_layers = [
			tdeom.ODEBlock(tdeom.ODEfunc(64, mode=config.pop('odemode', None)))
		]
	
		self.fc_layers = [
			tdeom.norm(64),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1,1)),
			tdeom.Flatten(),
			nn.Linear(64,10)
		]

		if len(config) != 0:
			logging.warn('Unknown config options are detected')

		self.model = nn.Sequential(
			*self.downsampling_layers,
			*self.feature_layers,
			*self.fc_layers
		)

		self.odefunc = self.feature_layers[0].odefunc
		self.tol = 1e-3
		self.odesolver = 'dopri5'

	def load_model(self, filename):
		logging.info('Loading parameters...')
		ldic = torch.load(filename, map_location=torch.device('cpu'))
		logging.info('Saved Arguments:\n{}'.format(ldic['args']))
		self.model.load_state_dict(ldic['state_dict'])
		logging.info('Parameters are restored.')

	def show_params(self):
		for n, v in self.model.named_parameters():
			print('{}({}): {}'.format(n,v.shape,v))

	def integrate(self, init_data, probe_points):
		import torchdiffeq as tde
		with torch.no_grad():
			self.model.eval()
			logging.info('calculating integration...')
			return tde.odeint(
				func=self.odefunc,
				y0=init_data,
				t=probe_points,
				rtol=self.tol,
				atol=self.tol,
				method=self.odesolver,
				options=None
			)

	def plot(self, start, stop, ndiv):
		import matplotlib
		import matplotlib.pyplot as pplt
		#matplotlib.use('gtk3agg')
		tsdata = self.integrate(init_data=torch.ones(1,64,6,6), probe_points=torch.linspace(start,stop,ndiv))
		pplt.plot(tsdata[:,:,:,0,0].reshape(ndiv,64))
		pplt.show()

if __name__ == '__main__':
	ext = extractor()
	ext.load_model(cmdargs.file)
	#ext.show_params()
	#out = ext.integrate(init_data=torch.ones(1,64,6,6), probe_points=torch.linspace(0,2,10000))
	#print(out[:,:,:,0,0].reshape(10000,64))
	ext.plot(0,2,10000)
	#pplt.plot(out[:,:,-1,0,0].flatten())
