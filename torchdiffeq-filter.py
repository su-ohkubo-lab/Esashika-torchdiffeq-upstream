import re
import sys

print('Epoch , Train Acc , Test Acc')

for s in sys.stdin:
	m = re.match(r'Epoch (\d*) \| .* \| Train Acc ([\d.]*) \| Test Acc ([\d.]*)', s)
	if not m:
		continue
	print('{} , {} , {}'.format(m.group(1),m.group(2),m.group(3)))
