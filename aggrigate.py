import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dpath', type=str)
parser.add_argument('--selector', type=str)
args = parser.parse_args()

import matplotlib
matplotlib.use('tkagg')

def get_datafilename():
	import os.path, glob
	spath = os.path.expanduser(args.dpath)
	if not os.path.isdir(spath):
		raise RuntimeError
	spath = os.path.join(spath, '*/logs')
	return glob.glob(spath)

def parse_datafile(path, col='test'):
	result = []
	with open(path, 'r') as fd:
		for l in fd:
			m = re.match(r'Epoch (?P<epoch>\d*) \| .* \| Train Acc (?P<train>[\d.]*) \| Test Acc (?P<test>[\d.]*)', l)
			if not m:
				continue
			result.append(float(m.group(col)))
	return result

import pandas
import re
pda = []
files = map(lambda x: re.search('layer(?P<layer>\d*)-try(?P<try>\d*)', x), get_datafilename())
files = sorted(list(files), key=lambda x: int(x.group('layer')+x.group('try')))
for fpath in files:
	path = fpath.string
	layer = fpath.group('layer')
	tries = fpath.group('try')
	result = parse_datafile(path)
	pda.append(pandas.Series(result, name='L'+layer+'-T'+tries))
pd = pandas.DataFrame(pda).T

if __name__ == '__main__':
	#fpd = pd.filter(regex=args.selector)
	#fpd.plot(colormap='cubehelix')
	#import matplotlib.pyplot as plt
	#r = map(lambda x: pd.loc[159].filter(regex=x), ['L0-', 'L1-', 'L2-', 'L3-', 'L4-', 'L5-', 'L6-', 'L7-', 'L8-', 'L9-'])
	#plt.show()
	import matplotlib.pyplot as plt
	plt.ylim(0.92, 0.94)
	#res = list()
	for layer in range(10):
		fpd = pd.filter(regex='L'+str(layer)+'-').iloc[-1]
		#res.append({'layer':layer, 'mean':fpd.mean(), 'std':fpd.std()})
		#plt.bar(layer, fpd.mean())
		plt.errorbar(layer, fpd.mean(), 1.96*fpd.sem(), ecolor='black', fmt='D', markeredgecolor='black', color='white', markersize=10, capsize=5)
		plt.scatter([layer]*len(fpd), fpd)
	plt.show()
