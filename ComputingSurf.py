import cv2
import numpy as np
import math


ImagePaths = [line.rstrip('\n') for line in open('Path/to/images.txt')]

# ------> Computing dictionary

dictionarySize = 5
sift = cv2.SIFT()

BOW = cv2.BOWKMeansTrainer(dictionarySize)

for i in range(len(ImagePaths)):
    gray = cv2.cvtColor(cv2.imread(ImagePaths), cv2.COLOR_BGR2GRAY)
    kp=[]
    kp, dsc= sift.detectAndCompute(cv2.resize(gray, (89, 40)), None)  # extracting keypoints and their descriptors
    if len(kp) > 0:  BOW.add(dsc)

voc = BOW.cluster()


#  -------> Computing final features

NofSample=len(ImagePaths)
SIFT=np.zeros([NofSample,dictionarySize])
NoDes=[]
for i in range(NofSample):
    gray = cv2.cvtColor(cv2.imread(ImagePaths), cv2.COLOR_BGR2GRAY)
    kp = []
    kp, dsc = sift.detectAndCompute(cv2.resize(gray, (89, 40)), None)
    if len(kp) > 0:
        print dsc.shape
        hist=np.zeros([1,dictionarySize])
        for k in range(dsc.shape[0]):

            d=[]

            for j in range(dictionarySize):
                x=voc[j,:]-dsc[k,:]
                sum=0
                for ii in range(x.shape[0]):
                    sum+=x[ii]*x[ii]
                d.append(math.sqrt(sum))

            val, idx = min((val, idx) for (idx, val) in enumerate(d))
            SIFT[i,idx]+=1
        # HIST[i,:]=HIST[i,:]/np.sum(HIST[i,:])

    else:
        # sometimes we cannot detect any keypoints. Let's keep track of it
        NoDes.append(i)



if len(NoDes)>0:
    SIFT=np.delete(SIFT,tuple(NoDes),0)


print SIFT