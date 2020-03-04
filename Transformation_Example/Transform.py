######################################################
import numpy as np
import pandas as pd
from skimage import transform as tf
import matplotlib.pyplot as plt
######################################################


class Transformation:
	""" Class that allows to compute the transformation of coordinates given some landmarks between two systems."""

	def __init__(self,Input,Output):
		""" Input & Output are .txt files
			Input contains Landmarks (named Repere) and points to tranfsorm (named Mesures)
			Output only contains Landmarks in the new system """
		#Load the textfile into the class
		self.input_ = Input
		self.output_ = Output

		#Create Dataframes from the textfiles
		try :
			self.Input_ = pd.read_csv(Input)
			self.Output_ = pd.read_csv(Output)
		except:
			self.Input_ = pd.read_excel(Input,header=1)
			self.Output_ = pd.read_excel(Output,header=1)
		#Extract Landmarks from initial system
		self.Repere_init_ = self.Input_[self.Input_['Type'].str.contains('Repere')]
		#Create array of initial landmark values for transform calculations
		self.Repere_init_array_ = np.zeros([self.Repere_init_.shape[0],2])
		#Fill the array with the values
		self.Repere_init_array_[:,0] = self.Repere_init_['X']
		self.Repere_init_array_[:,1] = self.Repere_init_['Y']
		#Extract Landmarks from final system
		self.Repere_final_ = self.Output_[self.Output_['Type'].str.contains('Repere')]
		#Create array of final landmark values for transform calculations
		self.Repere_final_array_ = np.zeros([self.Repere_final_.shape[0],2])
		#Fill the array with the values
		self.Repere_final_array_[:,0] = self.Repere_final_['X']
		self.Repere_final_array_[:,1] = self.Repere_final_['Y']
		#Extract measurements values from initial system
		self.Mesures_init_ = self.Input_[self.Input_['Type'].str.contains('Mesure')]
		#Create array of initial measurements for transform calculations
		self.Mesures_init_array_ = np.zeros([self.Mesures_init_.shape[0],2])
		#Fill the array with the values
		self.Mesures_init_array_[:,0] = self.Mesures_init_['X']
		self.Mesures_init_array_[:,1] = self.Mesures_init_['Y']


	def get_repere_init(self):
		""" Return a dataframe of the input landmarks"""
		return self.Input_[self.Input_['Type'].str.contains('Repere')]

	def get_repere_final(self):
		""" Return a dataframe of the output landmarks """
		return self.Output_[self.Output_['Type'].str.contains('Repere')]

	def get_mesures_init(self):
		""" Return a dataframe of the points to transform from the input """
		return self.Input_[self.Input_['Type'].str.contains('Mesure')]

	def transform(self,type='Proj'):
		""" In our working cases the transformation should only be affine (Rotation,Translation and Shear).
			But in general cases, the transformation could also have deformation and the transformation is 
			a projection. 
			See Mathematical definition of affine transformation and Projection (Homography) for more details. """
		if type == 'Affine' :
			self.transformation_ = tf.AffineTransform()
			self.transformation_.estimate(self.Repere_init_array_, self.Repere_final_array_)
			self.Mesures_final_array_ = self.transformation_(self.Mesures_init_array_)

		if type == 'Proj':
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
		fd.write('\n')
		for i in range(self.Mesures_final_array_.shape[0]):
	 		fd.write('{},{},{}\n'.format('Mesure'+str(i+1),round(self.Mesures_final_array_[i,0],2),round(self.Mesures_final_array_[i,1],2)))
		fd.close()


################################################################################################################################
#--------------------------- Running the code - Achieving the transformation --------------------------
TF = Transformation('Input.txt','Output.txt')
Tf.transform()
TF.extract_mesures_final()


