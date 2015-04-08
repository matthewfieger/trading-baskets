#!/usr/bin/env python
# ERC basket optimization algo
# On 20120516 by MLdP <lopezdeprado@lbl.gov>

import numpy as np
import scipy as sp
#-------------------------------------------
#-------------------------------------------
class ERC:
	def __init__(self,covar):
		# Class constructor
		self.covar=covar
		self.weights,self.ctr,self.ctb,self.tErr,self.iter=0,0,0,0,0
#-------------------------------------------
	def solve(self,precision,seed):
		# Compute the ERC holdings and stats
		w=np.zeros(shape=(seed.shape[0],seed.shape[1]))
		w[:]=seed[:]
		iter,n,grad=0,len(w),range(3)
		while True:
			#1) Normalize holdings
			w=w/w[0]
			#2) Compute CtB, CtR
			iter+=1
			risk=float((w.transpose()*self.covar*w)[0,0])**0.5
			tErr,mErr,iErr=0,0,0
			ctb,ctr,err=[0 for i in range(n)],[0 for i in range(n)],[0 for i in range(n)]
			for i in range(n):
				for j in range(n):
					ctb[i]+=w[j,0]*self.covar[i,j]/float(self.covar[i,i])**.5
				ctb[i]=ctb[i]/risk*get_Sign(w[i,0])
				ctr[i]=abs(w[i,0])*self.covar[i,i]**.5/risk*ctb[i]
				err[i]=1./n-ctr[i]
				tErr+=err[i]**2
				# Determine which leg to change
				if abs(err[i])>abs(mErr):
					mErr=err[i]
					iErr=i
			#3) Exit conditions
			tErr=(tErr/n)**.5 # Root mean squared deviation
			if tErr<10**-precision:break
			if iter>10**precision:break
			#4) CtR's Taylor expansion
			grad[0]=-err[iErr]
			grad[1]=w[iErr,0]*self.covar[iErr,iErr]/risk**2*(1-2*ctb[iErr]**2)+ \
				self.covar[iErr,iErr]**.5*ctb[iErr]/risk
			grad[2]=self.covar[iErr,iErr]/risk**2*(1-2*ctb[iErr]**2)- \
				ctb[iErr]*w[iErr,0]*(self.covar[iErr,iErr]**.5/risk)**3*(2-3*ctb[iErr]**2)
			#5) Step size
			if grad[2]!=0:
				delta=(grad[1]**2-4*grad[2]*grad[0])**.5
				if grad[1]>=0:
					delta=(-grad[1]+delta)/(2.*grad[2])
				else:
					delta=(-grad[1]-delta)/(2.*grad[2])
			else:
				delta=-grad[0]/grad[1]
			w[iErr,0]+=delta
		self.weights,self.ctr,self.ctb,self.tErr,self.iter=w,ctr,ctb,tErr,iter
#-------------------------------------------
	def get_Seed(self,col):
		# Seeds for hedging basket, based on alternating OLS holdings
		a=np.zeros(shape=(self.covar.shape[0],self.covar.shape[1]))
		a[:]=self.covar[:]
		a=sp.delete(a,col,0)
		a=sp.delete(a,col,1)
		a=np.linalg.inv(a)
		b=self.covar[:,col]
		b=sp.delete(b,col,0)
		c=a*b
		c=np.insert(c,col,-1,axis=0)
		return -c
#-------------------------------------------
#-------------------------------------------
def get_Sign(number):
	# Sign of the holding, needed to compute CtB
	if number==0:return 0 
	if number>=0:return 1
	if number<=0:return -1
#-------------------------------------------
def main():
	#1) Inputs (covariance, optional seed) --- to be changed by the user
	precision=5
	covar=np.matrix('846960.805351971,515812.899769821,403177.059835136; \
					515812.899769821,351407.396443653,280150.614979364; \
					403177.059835136,280150.614979364,232934.832710412')
	#seed=np.mat(np.ones((covar.shape[0],1))) #--- If a vector of ones is used as seed
	#2) Instantiate class
	erc=ERC(covar)
	#3) Compute and report solution for alternative seeds
	for i in range(covar.shape[0]):
		seed=erc.get_Seed(i) #--- Seed for hedging baskets
		erc.solve(precision,seed)
		# Report results
		print '##### SOLUTION '+str(i+1)+' #####'
		print '##### Holdings #####'
		print erc.weights
		print '##### CtR #####'
		print erc.ctr
		print '##### CtB #####'
		print erc.ctb
		print '##### Stats #####'
		print 'RMSD='+str(erc.tErr)
		print '#Iter='+str(erc.iter)
#-------------------------------------------
# Boilerplate
if __name__=='__main__': main()