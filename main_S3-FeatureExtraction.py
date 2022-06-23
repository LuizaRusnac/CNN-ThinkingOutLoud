import numpy as np
from FileUtils import load_data
import preprocessing
import featureExtr

# Establishing parameters
window = 250
domain = 'frequency'
band_samples = 0

print("Loading data...")
names = ['xtrain_rawdata_%ds_50tr-50tst_kfold.npy'%(window),
    'ytrain_rawdata_%ds_50tr-50tst_kfold.npy'%(window),
	'xval_rawdata_%ds_50tr-50tst_kfold.npy'%(window),
	'yval_rawdata_%ds_50tr-50tst_kfold.npy'%(window),
    'xtest_rawdata_%ds_50tr-50tst_kfold.npy'%(window),
    'ytest_rawdata_%ds_50tr-50tst_kfold.npy'%(window)]

xtrain, ytrain = load_data(names[0], names[1])
xval, yval = load_data(names[2], names[3])
xtest, ytest = load_data(names[4], names[5])
print("Loaded complete!")
print("Data train shape: ", xtrain.shape)
print("Data val shape: ", xval.shape)
print("Data test shape: ", xtest.shape)

xtr = np.zeros((xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],int(xtrain.shape[3]/2)))
xv = np.zeros((xval.shape[0],xval.shape[1],xval.shape[2],int(xval.shape[3]/2)))
xtst = np.zeros((xtest.shape[0],xtest.shape[1],xtest.shape[2],int(xtest.shape[3]/2)))

print("Computing features...")
if domain == 'frequency':
	for i in range(len(xtrain)):
		xtr[i] = featureExtr.spectrumChn(xtrain[i])
	for i in range(len(xval)):
		xv[i] = featureExtr.spectrumChn(xval[i])
	for i in range(len(xtest)):
		xtst[i] = featureExtr.spectrumChn(xtest[i])

	if band_samples != 0:
		for i in range(len(xtrain)):
			xtr[i] = featureExtr.freqBandMean(xtrain[i], band_samples)
		for i in range(len(xval)):
			xv[i] = featureExtr.freqBandMean(xval[i], band_samples)
		for i in range(len(xtest)):
			xtst[i] = featureExtr.freqBandMean(xtest[i], band_samples)
else:
	xtr = xtrain
	xv = xval
	xtst = xtest

xtrain = np.zeros((xtr.shape[0], xtr.shape[1], xtr.shape[2], xtr.shape[2]))
xval = np.zeros((xv.shape[0], xv.shape[1], xv.shape[2], xv.shape[2]))
xtest = np.zeros((xtst.shape[0], xtst.shape[1], xtst.shape[2], xtst.shape[2]))

for i in range(len(xtrain)):
	xtrain[i] = featureExtr.chConv(xtr[i])
for i in range(len(xval)):
	xval[i] = featureExtr.chConv(xv[i])
for i in range(len(xtest)):
	xtest[i] = featureExtr.chConv(xtst[i])
print("Done!")
print("Feature train shape: ", xtrain.shape)
print("Feature val shape: ", xval.shape)
print("Feature test shape: ", xtest.shape)

print("Normalizing features...")
for i in range(len(xtrain)):
	xtrain[i], minim, maxim = preprocessing.featureStd(xtrain[i], flag = 1)
for i in range(len(xval)):
	xval[i] = preprocessing.featureStd(xval[i], minim, maxim)
for i in range(len(xtest)):
	xtest[i] = preprocessing.featureStd(xtest[i], minim, maxim)
print("Done!")
print("Normalized xtrain features:")
print(xtrain)
print("Normalized xval features:")
print(xval)
print("Normalized xtest features:")
print(xtest)

print("Saving features...")
names = ['xtrain_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples),
    'ytrain_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples),
	'xval_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples),
	'yval_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples),
    'xtest_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples),
    'ytest_%ds_50tr-50tst_cov-%s_B%d_mean_kfold'%(window, domain, band_samples)]
np.save(names[0], xtrain)
np.save(names[1], ytrain)
np.save(names[2], xval)
np.save(names[3], yval)
np.save(names[4], xtest)
np.save(names[5], ytest)
print("Saved data!")
