import os
import sys
import numpy as np
import argparse
import time
        
import SimpleITK as sitk
from scipy.io import loadmat
from scipy.optimize import least_squares, curve_fit, leastsq, root
from scipy.special import huber
from numpy.fft import fftn, ifftn
from tqdm import trange, tqdm
from functools import partial
from multiprocessing import Pool, Lock, cpu_count, current_process
from multiprocessing.shared_memory import SharedMemory #V3.8+ only

import warnings
warnings.filterwarnings("ignore")
from itertools import combinations

app_name = 'pDate: proton density and T1/T2 Estimater'
version = '0.9.0'
release_date = '2022-11-25'

def print_header():
	print('\n\n')
	print(app_name, '- version: v', version, release_date, '\n')
	print('Computational Radiology Lab (CRL)\nBoston Children\'s Hospital, and Harvard Medical School\nhttp://crl.med.harvard.edu\nAuthor: Yao Sui')
	print('\n')


print_header()

def np_to_img(x, ref):
	img = sitk.GetImageFromArray(x)
	img.SetOrigin(ref.GetOrigin())
	img.SetSpacing(ref.GetSpacing())
	img.SetDirection(ref.GetDirection())
	return img
    
def aic(k, y_est, y_ref):
	return 2 * k + len(y_ref) * np.log(np.sum((y_est-y_ref)**2)/len(y_ref))

def aicc(k, y_est, y_ref):
	#return aic(k, y_est, y_ref) + 2 * k * (k+1) / (len(y_ref) - k - 1)
	n = len(y_ref)
	return np.log(np.sum((y_est-y_ref)**2)/n) + (n + k) / (n - k - 2)

def t1_model(ti, pd, t1, fa):
	return pd * (1 - (1 - np.cos(fa)) * np.exp(-ti / t1))

def cost_t1(x, ti, fa, y):
	t1 = np.clip(x[1], np.finfo(np.float32).eps, None)
	return x[0] * (1 - (1 - np.cos(fa)) * np.exp(-ti / t1)) - y

def t2_model(te, pd, t2):
	return pd * np.exp(-te/t2)

def cost_t2(x, te, y):
	t2 = np.clip(x[1], np.finfo(np.float32).eps, None)
	return x[0] * np.exp(-te/t2) - y

def cost_t1_t2(x, ti, fa, te, y1, y2):
	t1 = np.clip(x[1], np.finfo(np.float32).eps, None)
	cost_t1 = x[0] * (1 - (1 - np.cos(fa)) * np.exp(-ti / t1)) - y1
	t2 = np.clip(x[2], np.finfo(np.float32).eps, None)
	cost_t2 = x[0] * np.exp(-te/t2) - y2
	return np.concatenate([cost_t1, cost_t2], axis=-1)

def load_imgs(ii):
	shm_x = SharedMemory(name='x')
	x = np.ndarray([d,n,m,n_imgs], dtype=np.float32, buffer=shm_x.buf)

	img = sitk.ReadImage(flist[ii], sitk.sitkFloat32)
	print('loaded %d - %s ...' % (ii, flist[ii]))
	x[...,ii] = sitk.GetArrayFromImage(img)

	shm_x.close()
	print('Loaded image:', ii, flist[ii])

def init(l):
	global lock
	lock = l
    
def init_t1(y):
    pd0 = max(500, (y[-1]))
    if pd0 == 0:
        t10 = 0
    else:
        ti = TI[:-1]
        #t10 = TI[:-1] / -np.log(1/2 - 1/2*y[:-1]/pd0)
        
        tmp = 1/2 - 1/2*y[:-1]/pd0
        L = tmp > 0
        if np.all(L):
            t10 = np.mean(ti[L] / -np.log(tmp[L]))
            if t10 < 100:
                t10 = 500
        elif np.any(L):
            t10 = np.mean(ti[L] / -np.log(tmp[L]))
            if t10 < 100:
                t10 = 500
        else:
            t10 = 500
        
    if pd0<30:
        pd0=600
    
    t10 = max(500, min(t10, 5000))
    return [pd0, t10]
    
def init_t2(y):
    pd0 = max(0, y[0])
    if pd0 == 0:
        t20 = 0, 0
    else:
        t20 = TE[1:]/-np.log(y[1:]/pd0)
        if np.any(t20 > -np.inf):
            t20 = np.median(t20[t20>-np.inf])
        else:
            t20 = 0
    
    t20 = min(max(t20, 0), 4000)
    return [pd0, t20]

def compute_model(vidx, n_cores):
	# declare shared memory for numpy arrays
	shm_pd = SharedMemory(name='pd')
	pd = np.ndarray([d,n,m], dtype=np.float32, buffer=shm_pd.buf)
	shm_tm = SharedMemory(name='tm')
	tm = np.ndarray([d,n,m], dtype=np.float32, buffer=shm_tm.buf)
	shm_x = SharedMemory(name='x')
	x = np.ndarray([d,n,m,n_imgs], dtype=np.float32, buffer=shm_x.buf)

	#print(current_process().name, 'shared memory all declared', np.sum(pd), np.sum(x))

	#t0 = time.time()
	n_voxels = d * n * m
	job_size = len(range(vidx, n_voxels, n_cores))
	pd_local = np.empty(job_size)
	tm_local = np.empty(job_size)

	if vidx == 0:
		the_range = tqdm(range(vidx, n_voxels, n_cores))
	else:
		the_range = range(vidx, n_voxels, n_cores)

	kk = 0
	for vv in the_range:
		zz = int(np.floor(float(vv) / (d * n)))
		yy = int(vv % (d * n) % n)
		xx = int(np.floor(float(vv % (d * n)) / n))

		yii = np.squeeze(x[xx,yy,zz,:])
		if t1_mapping:
			r0 = init_t1(yii)
			res = least_squares(cost_t1, r0,
				args=(TI, fa, yii), loss='linear',
				bounds=([0, 0], [np.inf, 5000]))
		else:
			r0 = init_t2(yii)
			res = least_squares(cost_t2, r0,
				args=(TE, yii), loss='linear',
				bounds=([0, 0], [np.inf, 4000]))

		r = res.x

		pd_local[kk] = r[0]
		tm_local[kk] = r[1]

		kk += 1

	#lock.acquire()
	kk = 0
	for vv in range(vidx, n_voxels, n_cores):
		zz = int(np.floor(float(vv) / (d * n)))
		yy = int(vv % (d * n) % n)
		xx = int(np.floor(float(vv % (d * n)) / n))

		pd[xx,yy,zz] = pd_local[kk]
		tm[xx,yy,zz] = tm_local[kk]

		kk += 1
	#lock.release()
	#t = (time.time() - t0) * 1000. / job_size
	#print(current_process().name, 'generated', np.sum(np.isnan(pd)), t, 'ms per fit')


	# clean up for the shared memory
	shm_pd.close()
	shm_tm.close()
	shm_x.close()


parser = argparse.ArgumentParser()
parser.add_argument('-V', '--version', action='version',
		version='%s version : v %s %s' % (app_name, version, release_date),
		help='show version')
#group = parser.add_mutually_exclusive_group()
parser.add_argument('-m', '--mapping',
        help='specify [T1] or [T2] mapping will be estimated',
        choices=['T1', 'T2'], required=True)
parser.add_argument('--TI', type=ascii,
		help='specify inverse time (TI) values, separated by comma(,)')
parser.add_argument('--TE', type=ascii,
		help='specify echo time (TE) values, separated by comma(,)')
parser.add_argument('-fa', '--flip-angle', type=ascii,
		help='specify flip angle values in DEGREES, separated by comma(,)')
args = parser.parse_args()


data_path = 'data/'
out_path = 'recons/'

if not os.path.isdir(out_path):
	os.mkdir(out_path)

file_names = sorted([os.path.join(data_path, f)
		for f in os.listdir(data_path) if not os.path.isdir(f)])

# Generate unique groups of files of different sizes
generated_combinations = set()

group=[[0,2,4,6,8],[1,3,5,7,9],[0,1,2,3,4],[5,6,7,8,9],[0,2,6,7,9], [0,3,6,7,9],[0,4,7,8,9],[0,3,6,8,9],[0,1,2,4,7,8,9],[0,1,2,3,4,6,7,9],[0,1,2,3,4,5,6,7,8,9]]
# group=[[0,2,4,6,8],[1,3,5,7,9]]
TIs=[130,390,650,910,1170,1430,1690,1950,2380,2600]
data_path1 = 'data/'
flist1 = sorted([os.path.join(data_path1, f)
for f in os.listdir(data_path1) if not os.path.isdir(f)])
n_imgs1 = len(flist1)

for gg in range(,len(group)):
	data_path = 'data/'
	out_path = 'recons/'

	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	file_names = sorted([os.path.join(data_path, f)
			for f in os.listdir(data_path) if not os.path.isdir(f)])

	# Generate unique groups of files of different sizes
	generated_combinations = set()

	group=[[0,2,4,6,8],[1,3,5,7,9],[0,1,2,3,4],[5,6,7,8,9],[0,2,6,7,9], [0,3,6,7,9],[0,4,7,8,9],[0,3,6,8,9],[0,1,2,4,7,8,9],[0,1,2,3,4,6,7,9],[0,1,2,3,4,5,6,7,8,9]]
	# group=[[0,2,4,6,8],[1,3,5,7,9]]
	TIs=[130,390,650,910,1170,1430,1690,1950,2380,2600]
	data_path1 = 'data/'
	flist1 = sorted([os.path.join(data_path1, f)
	for f in os.listdir(data_path1) if not os.path.isdir(f)])
	n_imgs1 = len(flist1)

	data_path='data'+str(gg)+'/'

	print('found %d images:\n' % n_imgs1)
	print('\n'.join(flist1))
	os.system('mkdir data'+str(gg))
	print(group[gg])
	files = [flist1[i] for i in group[gg]]
	flist = sorted([os.path.join(data_path, f)
	for f in os.listdir(data_path) if not os.path.isdir(f)])
	print(files)
	n_imgs = len(flist)

	for temp in range(len(files)):
		os.system('cp -r '+files[temp]+' data'+str(gg)+'/')
	out_path='recons'+str(gg)+'/'
	if not os.path.isdir(out_path):
		os.mkdir(out_path)


	t1_mapping = True

	if args.mapping.lower() == 't1':
		if args.TI:
			# TI = np.array([float(x) for x in args.TI[1:-1].split(',')])
			TI = np.asarray([TIs[i] for i in group[gg]])
			print(TI)
		else:
			print('inverse time (TI) for each image is required')
			exit()
		if args.flip_angle:
			fa = np.array([float(x) / 158. * np.pi for x in args.flip_angle[1:-1].split(',')])
			if fa.shape[-1] == 1:
				fa = np.repeat(fa, TI.shape[-1])
		else:
			print('flip angle for each image is required')
			exit()
			
		if not TI.shape[-1] == fa.shape[-1]:
			print('TIs and flip angles should be the same length')
			exit()
		n_inversions = TI.shape[-1]
		# if not n_imgs == n_inversions:
		# 	print('the number of images should be equal to the number of inversions')
		# 	exit()
		
	elif args.mapping.lower() == 't2':
		if args.TE:
			np.array(TE = [float(x) for x in args.TE[1:-1].split(',')])
		else:
			print('echo time (TE) for each image is required')
			exit()
		t1_mapping = False
		n_echoes = TE.shape[-1]
		if not n_imgs == n_echoes:
			print('the number of images should be equal to the number of echoes')
			exit()
	else:
		print('please specify T1 or T2 that will be estimated')
	print(flist)

	# load the first image and use it as the reference image
	img0 = sitk.ReadImage(flist[0], sitk.sitkFloat32)

	m, n, d = img0.GetSize()
	print(len(flist), TI, fa, m, n, d, n_imgs)
	print('creating shared memory between processes')
	# create shared memory for x
	size_x = d * n * m * n_imgs * np.dtype(np.float32).itemsize
	shm_x = SharedMemory(name='x', create=True, size=size_x)
	x = np.ndarray([d,n,m,n_imgs], np.float32, buffer=shm_x.buf)
	x[:] = 0

	print('loading images...')
	with Pool(processes=min(cpu_count(), x.shape[-1])) as p:
		p.map(load_imgs, range(0, x.shape[-1]))
	print(x.shape[-1], 'images have been loaded')

	# create shared memory for variables
	size_pd = d * n * m * np.dtype(np.float32).itemsize
	shm_pd = SharedMemory(name='pd', create=True, size=size_pd)
	pd = np.ndarray([d,n,m], dtype=np.float32, buffer=shm_pd.buf)
	pd[:] = 0
	shm_tm = SharedMemory(name='tm', create=True, size=size_pd)
	tm = np.ndarray([d,n,m], dtype=np.float32, buffer=shm_tm.buf)
	tm[:] = 0

	l = Lock()
	n_cores = cpu_count()
	print('start estimating on', n_cores, 'cores ...')

	pool = Pool(processes=n_cores, initializer=init, initargs=(l,))
	pool.map(partial(compute_model, n_cores=n_cores), range(0, n_cores))

	sitk.WriteImage(sitk.Cast(np_to_img(pd, img0), sitk.sitkFloat32),
			out_path + 'pd'+str(gg)+'.nrrd')
	sitk.WriteImage(sitk.Cast(np_to_img(tm, img0), sitk.sitkFloat32),
			out_path + 'tm'+str(gg)+'.nrrd')

	# clean up for all shared memory
	shm_pd.close()
	shm_pd.unlink()
	shm_tm.close()
	shm_tm.unlink()
	shm_x.close()
	shm_x.unlink()

	pool.close()
	pool.join()


	print('========== ALL DONE ============')
