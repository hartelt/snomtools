import numpy as np

import os

#Locale settings
import locale
locale.setlocale(locale.LC_ALL, "")

from matplotlib import rc, cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt

###########################################
# matplotlib rc settings
###########################################

fontsize_xylabel = 11.
fontsize_title = 13.
fontsize_ticks = 9.

rc('font', **{'family': 'serif', 'weight': 'light',"size":11., "style": 'normal'})
rc('xtick', **{'labelsize': fontsize_ticks})
rc('ytick', **{'labelsize': fontsize_ticks})
#rc('axes', **{'labelweight': 'normal'})
rc('axes', **{'linewidth': 1.5})
rc('axes.formatter', use_locale=True)
# tex using six units and icomma

def usepackage(package, options=None):
	opt = '\[%s\]' % options if options is not None else ''
	return "\\usepackage%s{%s}" % (opt, package)
preamble = "\n".join([
	usepackage('siunitx'),
	usepackage('icomma'),
	usepackage('fixltx2e')	# for using \textsubscript
#	r'\sisetup{detect-all}'
	])

# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('text.latex', unicode=True, preamble=preamble)

###########################################
# matplotlib convenience
###########################################

figures_path = os.path.abspath(os.path.expanduser('~'))

def unit(text, prefix=''):
	return u"%s\\si{%s}" % (prefix, text)

def a2r(angle):
	if isinstance(angle, list):
		angle = np.array(angle)
	return angle/180.*np.pi
def r2a(rad):
	if isinstance(rad, list):
		rad = np.array(rad)
	return rad/np.pi*180.

def savefig(filename=None, suffix='', figures_path=figures_path, ext='pdf', **kwargs):
	import sys
	if '-build' not in sys.argv:
		return False
	if filename is None:
		import inspect
		frm = inspect.stack()[1]
		module = inspect.getmodule(frm[0])
		filename = ''.join([os.path.basename(module.__file__).replace('.py', ''),
							suffix, '.%s' % ext])

	path = '/'.join([figures_path,
					 filename])
	kwargs['bbox_inches'] = 'tight'
	plt.savefig(path, **kwargs)
	print('stored %s' % path)


def figure(size_cm=(3, 3), dpi=600, num=0):
	import sys
	if '-build' not in sys.argv:
		return  plt.figure(num=num)
	in_cm = 1. / 2.514  # 1 inch = 2.514 cm
	size_inch = (size_cm[0] * in_cm, size_cm[1] * in_cm)
	return plt.figure(num, size_inch, dpi=dpi)

def set_xlabel(text, ax=None, fontsize=fontsize_xylabel, **kwargs):
	"""
	applies the x label text
	:param ax: if None gca is used
	:return: axes_label
	"""
	if ax is None:
		ax = plt.gca()
	return ax.set_xlabel(text, fontsize=fontsize, **kwargs)

def set_ylabel(text, ax=None, fontsize=fontsize_xylabel, **kwargs):
	"""
	applies the y label text
	:param ax: if None gca is used
	:return: axes_label
	"""
	if ax is None:
		ax = plt.gca()
	return ax.set_ylabel(text, fontsize=fontsize, **kwargs)

def set_title(text, ax=None, fontsize=fontsize_title, **kwargs):
	"""
	applies the title text
	:param ax: if None gca is used
	:return: label
	"""
	if ax is None:
		ax = plt.gca()
	return ax.set_title(text, fontsize=fontsize, **kwargs)

def set_label(text, ax, fontsize=fontsize_xylabel, **kwargs):
	"""
	applies the label text
	:return: label
	"""
	return ax.set_label(text, fontsize=fontsize, **kwargs)

def legend(axis=None, handles=None, labels=None, loc=None, fontsize=fontsize_xylabel, *args, **kwargs):
	if axis is None:
		axis = plt.gca()
	frameon = kwargs.get('frameon', False)
	kwargs['frameon'] = frameon
	kwargs['fontsize'] = fontsize
	kwargs['loc'] = loc
	if handles is not None and labels is not None:
		leg = axis.legend(handles, labels, **kwargs)
	else:
		leg = axis.legend(*args, **kwargs)

	if not frameon:
		rect = leg.get_frame()
		#rect.set_facecolor('#')
		rect.set_linewidth(0.0)
	return leg

###########################################
# matplotlib colormap
###########################################

BuOrRd = {'red':   ((0.0,  8./256., 8./256.),
				   (0.5,  253./256., 253./256.),
				   (1.0,  179./256, 179./256)),

		 'green': ((0.0,  29./256., 29./256.),
				   (0.5, 141./256, 141./256),
				   (1.0,  0.0, 0.0)),

		 'blue':  ((0.0,  88./256., 88./256.),
				   (0.5,  60./256., 60./256.),
				   (1.0,  0.0, 0.0))}

cmap_BuOrRd = LinearSegmentedColormap('BuOrRd', BuOrRd)
cmap_spectral = cm.get_cmap('Spectral')

def invert(color):
	pcolor = [(1.-m, start, stop) for m, start, stop in color[::-1]]
	return pcolor

cmap_spectral_inv = LinearSegmentedColormap('ISpectral',
	dict(((color, invert(values)) for color, values in cmap_spectral._segmentdata.iteritems())))
cm.cmap_d['ISpectral'] = cmap_spectral_inv

#cmap_evolution = ListedColormap((np.array([(179, 179, 179, 256),  # unknown
#                                 (20, 20, 200, 256),  # Individual meaning child
#                                 (200, 60.0, 60.0, 256.0),  # Reference
#                                 (137, 230, 124, 256),  # Guess
#                                 (0, 0, 0, 256),  # Survivor aka Bauer
#                                 ], dtype=float)/256.).tolist(), name='individual_typed')
cmap_evolution = ListedColormap([
	'#c0c0c0',
	cmap_BuOrRd(120),
	(220./256., 60./256., 60./256.),
	(137./256., 230./256., 124./256.),
	cmap_BuOrRd(0)
])

def get_cmap(N=3, Offset=.00):
	cmap = cmap_BuOrRd
	n = float(N + abs(Offset)) - 1
	Offset = max(Offset, 0)
	def map(value):
		index = int(np.clip(value+Offset, Offset, n) * cmap.N / n)
		return cmap(index)
	return map
