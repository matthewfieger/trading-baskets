#!/usr/bin/env python
# MMSC basket optimization algo
# On 20120516 by MLdP <lopezdeprado@lbl.gov>
import numpy as np
from scipy import delete
from itertools import combinations
from math import log
#-------------------------------------------
#-------------------------------------------
class MMSC:
	def __init__(self,covar,maxSubsetSize):
		# Class constructor
		self.covar=covar
		if maxSubsetSize>=covar.shape[0]:maxSubsetSize=covar.shape[0]-1
		self.subsets=self.get_Subsets(range(covar.shape[0]),maxSubsetSize)
		self.weights,self.tErr,self.iter=0,0,0
		self.scorrel=0,0
#-------------------------------------------
	def solve(self,precision,seed,hedge=True):
		# Compute the MMSC holdings and stats
		w=np.zeros(shape=(seed.shape[0],seed.shape[1]))
		weights=np.zeros(shape=(seed.shape[0],seed.shape[1]))
		w[:]=seed[:]
		iterTotal,iterW,aim,msc,n,N,grad=0,0,0,1,len(w)-1,len(self.subsets)-1,range(3)
		if hedge==False:msc=0 # For a trading basket
		loop=True
		#1) Iterations
		while loop==True:
			iterTotal+=1
			iterW+=1
			#2) Aim size
			if iterW==1:
				aim=max(1,aim/(1+log(n)))
			else:
				aim=min(10**6,aim*(1+log(n)))
				w[:]=weights[:]
			#3) Iterate subsets
			for i in range(N):
				#4) Normalize holdings
				w=w/w[0]
				#5) Compute subset correl matrix
				scovar=self.get_SubsetCovar(w)
				scorrel=get_Correl(scovar)
				#6) Determine which leg to change
				avg=(sum(scorrel[:,-1])-1)/N
				err=[avg-scorrel[j,-1]for j in range(N)]
				iErr,tErr=0,0
				for j in range(N):
					tErr+=err[j]**2
					if hedge==True and scorrel[j,-1]>scorrel[iErr,-1]:iErr=j
					if hedge==False and scorrel[j,-1]<scorrel[iErr,-1]:iErr=j
				#7) Store new optimum
				if (hedge==True and scorrel[iErr,-1]<msc) or \
					(hedge==False and scorrel[iErr,-1]>msc):
					weights[:]=w[:]
					iterW=0
					msc=scorrel[iErr,-1]
				#8) CtB's Taylor expansion
				grad[0]=err[iErr]/aim
				grad[1]=-(scovar[i,i]/scovar[-1,-1])**.5*(scorrel[i,iErr]-scorrel[iErr,-1]* \
					scorrel[i,-1])
				grad[2]=.5*scovar[i,i]/scovar[-1,-1]*(scorrel[iErr,-1]+scorrel[i,-1]* \
					(2*scorrel[i,iErr]-3*scorrel[iErr,-1]*scorrel[i,-1]))
				#9) Step size
				if grad[2]!=0:
					delta=(grad[1]**2-4*grad[2]*grad[0])**.5
					if grad[1]>=0:
						delta=(-grad[1]+delta)/(2.*grad[2])
					else:
						delta=(-grad[1]-delta)/(2.*grad[2])
				elif grad[1]!=0:
					delta=-grad[0]/grad[1]
				#10) Backpropagate subset step back to legs
				w=self.get_Backpropagate(w,delta,i)
				#11) Exit conditions
				tErr=(tErr/N)**.5 # Root mean squared deviation
				#if tErr<10**-precision:loop=False # -If exit by convergence
				if iterTotal>10**precision:loop=False
		scovar=self.get_SubsetCovar(weights)
		scorrel=get_Correl(scovar)
		self.weights,self.tErr,self.iter,self.scorrel=weights,tErr,iterTotal,scorrel
#-------------------------------------------
	def get_Seed(self,col,hedge):
		# Seed for basket
			if hedge==True:
				# Seeds for hedging basket, based on alternating OLS holdings
				a=np.zeros(shape=(self.covar.shape[0],self.covar.shape[1]))
				a[:]=self.covar[:]
				a=delete(a,col,0)
				a=delete(a,col,1)
				a=np.linalg.inv(a)
				b=self.covar[:,col]
				b=delete(b,col,0)
				c=a*b
				c=np.insert(c,col,-1,axis=0)
				return -c
			else:
				# Seeds for trading basket, based on the leg's betas to the basket
				a=np.zeros(shape=(self.covar.shape[0],1))
				for b in range(self.covar.shape[0]):
					a[b,0]=self.covar[b,b]/self.covar[0,0]
				return a
#-------------------------------------------
	def get_Backpropagate(self,w,delta,subset):
		# Backpropagates subset delta to the legs
		for leg in self.subsets[subset]:
			w[leg]=w[leg]*(1+delta)
		return w
#-------------------------------------------
	def get_Subsets(self,iterable,maxSubsetSize):
		# Generate all subsets up to maxSubsetSize, but including the full set
		subsets=[item for item in combinations(iterable,1)]
		for subsetSize in range(2,maxSubsetSize+1):
			for item in combinations(iterable,subsetSize):
				subsets.append(item)
		for item in combinations(iterable,self.covar.shape[0]):
			subsets.append(item)
		return subsets
#-------------------------------------------
	def get_SubsetCovar(self,w):
		# Computes covariances among all subsets
		subsetCovar=np.zeros(shape=(len(self.subsets),len(self.subsets)))
		for i in range(len(self.subsets)):
			for j in range(i,len(self.subsets)):
				for k in self.subsets[i]:
					for l in self.subsets[j]:
						subsetCovar[i,j]+=w[k]*w[l]*self.covar[k,l]
				subsetCovar[j,i]=subsetCovar[i,j] 
		return subsetCovar
#-------------------------------------------
#-------------------------------------------
def get_Sign(number):
	# Sign of the holding, needed to compute CtB
	if number==0:return 0
	if number>=0:return 1
	if number<=0:return -1
#-------------------------------------------
def get_Correl(covar):
	# Correl matrix from covar
	correl=np.zeros(shape=(covar.shape[0],covar.shape[1]))
	for i in range(covar.shape[0]):
		for j in range(i,covar.shape[1]):
			correl[i,j]=covar[i,j]/(covar[i,i]*covar[j,j])**.5
			correl[j,i]=correl[i,j]
	return correl
#-------------------------------------------
def main():
	#1) Input parameters --- to be changed by the user
	precision=3
	covar=np.matrix('846960.805351971,515812.899769821,403177.059835136; \
	515812.899769821,351407.396443653,280150.614979364; \
	403177.059835136,280150.614979364,232934.832710412')
	maxSubsetSize=covar.shape[0]-1 # Set maxSubsetSize=1 for MDR
	hedge=True # True for a hedging basket, False for trading basket
	#2) Instantiate class
	mmsc=MMSC(covar,maxSubsetSize)
	#3) Compute and report solution for alternative seeds
	for i in range(covar.shape[0]):
		seed=mmsc.get_Seed(i,hedge) #--- Seed for baskets mmsc.solve(precision,seed,hedge)
		# Report results
		print '##### SOLUTION '+str(i+1)+' #####'
		print '##### Holdings #####'
		print mmsc.weights
		print '##### Subset Correl #####'
		print mmsc.scorrel
		print '##### Stats #####'
		print 'RMSD='+str(mmsc.tErr)
		print '#Iter='+str(mmsc.iter)
		if hedge==False:return
#-------------------------------------------
# Boilerplate
if __name__=='__main__': main()