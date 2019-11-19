######################################################
import numpy as np
import pandas as pd
from skimage import data
from skimage import transform as tf
######################################################




class Transformation:
	""" Class that allows to compute the transformation of coordinates given some landmarks between two systems."""

	def __init__(self,Input,Output):
		""" Input & Output are .txt files
			Input contains Landmarks (named Repere) and points to tranfsorm (named Mesures)
			Output only contains Landmarks in the new system """
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
		""" Return a dataframe of the input landmarks"""
		return self.Input_[self.Repere_init_]

	def get_repere_final(self):
		""" Return a dataframe of the output landmarks """
		return self.Output_[self.Repere_final_]

	def get_mesures_init(self):
		""" Return a dataframe of the points to transform from the input """
		return self.Input_[self.Mesures_init_]

	def transform(self,type=None):
		""" In our working cases the transformation should only be affine (Rotation,Translation and Shear).
			But in general cases, the transformation could also have deformation and the transformation is 
			a projection. 
			See Mathematical definition of affine transformation and Projection (Homography) for more details. """
		if type is None :
			self.transformation_ = tf.AffineTransform()
			self.transformation_.estimate(self.Repere_init_array_, self.Repere_final_array_)
			self.Mesures_final_array_ = self.transformation_(self.Mesures_init_array_)

		else:
			self.transformation_ = tf.ProjectiveTransform()
			self.transformation_.estimate(self.Repere_init_array_, self.Repere_final_array_)
			self.Mesures_final_array_ = self.transformation_(self.Mesures_init_array_)

	def get_transform_matrix(self):
		""" Return the array of the transformation matrix """
		return self.transformation_.params

	def get_transform_repere(self):
		""" Return the transformation of the Landmarks done by the matrix.
			Usefull to compare the the true and the calculated landmarks """
		return self.transformation_(self.Repere_init_array_)


	def extract_mesures_final(self):
		""" Write the calculated coordinates into the output textfile """
		fd = open(self.output_,"a")
		for i in range(self.Mesures_final_array_.shape[0]):
	 		fd.write('{},{},{}\n'.format('Mesures',self.Mesures_final_array_[i,0],self.Mesures_final_array_[i,1]))
		fd.close()




################################################################################################################################
#--------------------------- Running the code - Achieving the transformation --------------------------
TF = Transformation('ProjetLaboIn.txt','ProjetLaboOut.txt')
TF.transform(type='Proj')
TF.extract_mesures_final()

