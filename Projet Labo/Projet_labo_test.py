from __future__ import print_function

import math
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import scipy as sp 
from skimage import data
from skimage import transform as tf
import pandas as pd

repere = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
nouveau_repere = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

mesures_X = randint(0,300,50)
mesures_Y = randint(0,50,50)



Mesures = np.zeros((mesures_X.shape[0],2))
for i in range(Mesures.shape[0]):
	Mesures[i,0] = mesures_X[i] 
	Mesures[i,1] = mesures_Y[i]

class Transformation:
	def __init__(self,Input,Output):
		self.input_ = Input
		self.output_ = Output
		self.Input_ = pd.read_csv(Input)
		self.Output_ = pd.read_csv(Output)
		self.Repere_init_ = self.Input_['Type'] == 'Repere'
		self.Repere_init_array_ = np.zeros([self.Input_[self.Repere_init_].shape[0],2])
		self.Repere_init_array_[:,0] = self.Input_[self.Repere_init_]['X']
		self.Repere_init_array_[:,1] = self.Input_[self.Repere_init_]['Y']
		self.Repere_final_ = self.Output_['Type'] == 'Repere'
		self.Repere_final_array_ = np.zeros([self.Output_[self.Repere_final_].shape[0],2])
		self.Repere_final_array_[:,0] = self.Output_[self.Repere_final_]['X']
		self.Repere_final_array_[:,1] = self.Output_[self.Repere_final_]['Y']
		self.Mesures_init_ = self.Input_['Type'] == 'Mesure'
		self.Mesures_init_array_ = np.zeros([self.Input_[self.Mesures_init_].shape[0],2])
		self.Mesures_init_array_[:,0] = self.Input_[self.Mesures_init_]['X']
		self.Mesures_init_array_[:,1] = self.Input_[self.Mesures_init_]['Y']


	def get_repere_init(self):
		return self.Input_[self.Repere_init_]

	def get_repere_final(self):
		return self.Output_[self.Repere_final_]

	def get_mesures_init(self):
		return self.Input_[self.Mesures_init_]

	def transform(self,type=None):
		if type is None :
			self.transformation_ = tf.AffineTransform()
			self.transformation_.estimate(self.Repere_init_array_, self.Repere_final_array_)
			self.Mesures_final_array_ = self.transformation_(self.Mesures_init_array_)

		else:
			self.transformation_ = tf.ProjectiveTransform()
			self.transformation_.estimate(self.Repere_init_array_, self.Repere_final_array_)
			self.Mesures_final_array_ = self.transformation_(self.Mesures_init_array_)

	def get_transform_matrix(self):
		return self.transformation_.params

	def get_transform_repere(self):
		return self.transformation_(self.Repere_init_array_)


	def extract_mesures_final(self):
		fd = open(self.output_,"a")
		for i in range(self.Mesures_final_array_.shape[0]):
	 		fd.write('{},{},{}\n'.format('Mesures',self.Mesures_final_array_[i,0],self.Mesures_final_array_[i,1]))
		fd.close()



###### Read Text files containing data #######
# Input = pd.read_csv('ProjetLaboIn.txt')
# Output = pd.read_csv('ProjetLaboOut.txt')
# Repere_init = Input['Type'] == 'Repere'
# Repere_init_array = np.zeros([Input[Repere_init].shape[0],2])
# Repere_init_array[:,0] = Input[Repere_init]['X']
# Repere_init_array[:,1] = Input[Repere_init]['Y']
# Repere_final = Output['Type'] == 'Repere'
# Repere_final_array = np.zeros([Output[Repere_final].shape[0],2])
# Repere_final_array[:,0] = Output[Repere_final]['X']
# Repere_final_array[:,1] = Output[Repere_final]['Y']
# Mesures_init = Input['Type'] == 'Mesure'
# Mesures_init_array = np.zeros([Input[Mesures_init].shape[0],2])
# Mesures_init_array[:,0] = Input[Mesures_init]['X']
# Mesures_init_array[:,1] = Input[Mesures_init]['Y']


# transformation = tf.ProjectiveTransform()
# transformation.estimate(Repere_init_array,Repere_final_array)
# Mesures_final_array = transformation(Mesures_init_array)
# fd = open('ProjetLaboOut.txt',"a")
# for i in range(Mesures_final_array.shape[0]):
# 	fd.write('{},{},{}\n'.format('Mesures',Mesures_final_array[i,0],Mesures_final_array[i,1]))
# fd.close()

########## Using Scikit Image - Projective Transform###############
projection = tf.ProjectiveTransform()
projection.estimate(repere, nouveau_repere)
output = projection(Mesures)
############################################
 
########## Using Scikit Image - Affine Transform #################
affine = tf.AffineTransform()
affine.estimate(repere, nouveau_repere)
Output = affine(Mesures)
############################################


################ Plotting #################

fig, ax = plt.subplots(nrows=3, figsize=(12, 7))

ax[0].scatter(repere[:,0],repere[:,1],c='red')
ax[0].scatter(Mesures[:,0],Mesures[:,1],c='black')
ax[0].grid()
ax[0].set_title("Repère d'origine")
ax[1].scatter(nouveau_repere[:,0],nouveau_repere[:,1],c='red')
ax[1].scatter(output[:,0],output[:,1],c='black')
ax[1].grid()
ax[1].set_title("Nouveau repère utilisant ProjectiveTransform")
ax[2].scatter(affine(repere)[:,0],affine(repere)[:,1],c='red')
ax[2].scatter(Output[:,0],Output[:,1],c='black')
ax[2].grid()
ax[2].set_title("Nouveau repère utilisant AffineTransform")

plt.tight_layout()
plt.axis()
plt.show()