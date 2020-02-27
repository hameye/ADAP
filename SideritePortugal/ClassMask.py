######################################################
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import hyperspy.api as hs 
######################################################

plt.rcParams['image.cmap'] = 'cividis'


class Mask :
	def __init__(self,prefix,suffix,table, Echelle = True, Type = 'images', Normalisation = True):
		self.type_ = Type
		self.prefix_ = prefix
		self.suffix_ = suffix
		self.echelle_ = Echelle
		self.Normalisation_ = Normalisation
		try:
			self.table_ = pd.read_csv(table)
		except:
			self.table_ = pd.read_excel(table,header=1)

	def datacube_creation(self):

		if self.type_ == 'images' : 
			self.Elements_ = {}
			test = np.linalg.norm(imread(self.prefix_+self.table_.iloc[0][0]+self.suffix_),axis=2)
			self.datacube_ = np.zeros((test.shape[0],test.shape[1],self.table_.shape[0]))
			test[:,:]=0

			for element in range(self.table_.shape[0]):
				self.Elements_[element]=self.table_.iloc[element]['Element']

				if '/' not in self.table_.iloc[element]['Element']:
					self.datacube_[:,:,element] = np.linalg.norm(imread(self.prefix_+self.table_.iloc[element][0]+self.suffix_),axis=2)#/np.max(np.linalg.norm(imread(self.prefix_+self.table_.iloc[element][0]+self.suffix_),axis=2))*100
					if self.echelle_ == True :
						test +=self.datacube_[:,:,element]
				else : 
					image_over = imread(self.prefix_+self.table_['Element'][element].split('/')[0]+self.suffix_)
					image_under = imread(self.prefix_+self.table_['Element'][element].split('/')[1]+self.suffix_)
					image_over_grey = np.linalg.norm(image_over,axis=2)
					image_under_grey = np.linalg.norm(image_under,axis=2)
					image_under_grey[image_under_grey==0.]=0.001
					self.datacube_[:,:,element] = image_over_grey/image_under_grey
			        #if echelle == True :
			            #test +=self.datacube_[:,:,element]



		else : 
			cube = hs.load(self.prefix_[:-1]+".rpl", signal_type="EDS_SEM",lazy=True)
			cube.axes_manager[-1].name = 'E'
			cube.axes_manager['E'].units = 'keV'
			cube.axes_manager['E'].scale = 0.01
			cube.axes_manager['E'].offset = -0.97
			self.Elements_ = {}
			test = np.linalg.norm(imread(self.prefix_+self.table_.iloc[0][0]+self.suffix_),axis=2)
			self.datacube_ = np.zeros((test.shape[0],test.shape[1],self.table_.shape[0]))
			test[:,:]=0

			for element in range(self.table_.shape[0]):
				self.Elements_[element]=self.table_.iloc[element]['Element']

				if '/' not in self.table_.iloc[element]['Element']:
					cube.set_elements([self.table_.iloc[element]['Element']])
					array = cube.get_lines_intensity()
					#self.datacube_[:,:,element] = np.linalg.norm(imread(self.prefix_+self.table_.iloc[element][0]+self.suffix_),axis=2)
					self.datacube_[:,:,element] = np.asarray(array[0])

					if self.echelle_ == True :
						test +=self.datacube_[:,:,element]
				
				else : 
					image_over = imread(self.prefix_+self.table_['Element'][element].split('/')[0]+self.suffix_)
					image_under = imread(self.prefix_+self.table_['Element'][element].split('/')[1]+self.suffix_)
					image_over_grey = np.linalg.norm(image_over,axis=2)
					image_under_grey = np.linalg.norm(image_under,axis=2)
					image_under_grey[image_under_grey==0.]=0.001
					self.datacube_[:,:,element] = image_over_grey/image_under_grey
			        #if echelle == True :
			            #test +=self.datacube_[:,:,element]

		if self.echelle_ == True :
			for i in range(len(self.Elements_)): 
				self.datacube_[:,:,i][test>3000]=np.nan

		if self.Normalisation_ == True :
			for i in range(len(self.Elements_)): 
				self.datacube_[:,:,i] = self.datacube_[:,:,i]/np.nanmax(self.datacube_[:,:,i])*100




	def workingcube_creation(self):
		self.Minerals_ = {}
		self.mineralcube_ = np.zeros((self.datacube_.shape[0],self.datacube_.shape[1],self.table_.shape[1]-1))

		for mask in range(1,self.table_.shape[1]):
			name = self.table_.columns[mask]
			self.Minerals_[mask-1] = name
			str_table = self.table_[name].astype('str',copy=True)
			index_str = np.where(self.table_[name].notnull())[0]
			mask_i_str = np.zeros((self.datacube_.shape[0],self.datacube_.shape[1],index_str.shape[0]))

			for k in range(index_str.shape[0]):
				mask_i_str[:,:,k] = self.datacube_[:,:,index_str[k]]

				if len(str_table[index_str[k]].split('-')) == 1 : 
					threshold_min = float(str_table[index_str[k]].split('-')[0])
					threshold_max = 1.

				else:
					threshold_min = float(str_table[index_str[k]].split('-')[0])
					threshold_max = float(str_table[index_str[k]].split('-')[1])

				mask_i_str[:,:,k][mask_i_str[:,:,k]>threshold_max*np.nanmax(mask_i_str[:,:,k])]=np.nan
				mask_i_str[:,:,k][mask_i_str[:,:,k]<threshold_min*np.nanmax(mask_i_str[:,:,k])]=np.nan
				mask_i_str[np.isfinite(mask_i_str)]=1

			mask_i_str = np.nansum(mask_i_str,axis=2)
			mask_i_str[mask_i_str<np.max(mask_i_str)]=np.nan
			self.mineralcube_[:,:,mask-1] = mask_i_str/np.nanmax(mask_i_str)

	def get_mask(self,indice):
		plt.imshow(self.mineralcube_[:,:,indice])
		plt.title(self.Minerals_[indice])
		plt.grid()
		plt.show()

	def save_mask(self,indice):
		plt.imshow(self.mineralcube_[:,:,indice])
		plt.title(self.Minerals_[indice])
		plt.savefig('Mask_'+self.Minerals_[indice]+'.tif')
		plt.close()
		#np.save('Mask'+self.Minerals_[indice],self.mineralcube_[:,:,indice])

	def get_hist(self,indice):
		fig, axes = plt.subplots(1, 2, figsize=(10, 5))
		ax = axes.ravel()
		Anan=test.datacube_[:,:,indice][~np.isnan(test.datacube_[:,:,indice])]
		im = ax[0].imshow(self.datacube_[:,:,indice])
		ax[0].grid()
		ax[0].set_title("Carte élémentaire : "+self.Elements_[indice])
		fig.colorbar(im,ax=ax[0])
		plt.ylim(0,np.max(Anan))
		sns.distplot(Anan, kde=False,ax=axes[1],hist_kws={'range': (0.0, np.max(Anan))},vertical=True)
		ax[1].set_xscale('log')
		ax[1].set_title("Histograme d'intensité : "+self.Elements_[indice])
		fig.tight_layout()
		plt.show()
	
	def plot_mineral_mask(self):
		proportion={}
		array = np.zeros((self.datacube_.shape[0],self.datacube_.shape[1]))
		array[np.isfinite(array)]=np.nan
		for indice in range(len(self.Minerals_)):
			array[np.isfinite(self.mineralcube_[:,:,indice])] = indice
			proportion[indice] = np.sum(np.isfinite(self.mineralcube_[:,:,indice])) / np.sum(np.isfinite(self.mineralcube_))*100#(array.shape[0]*array.shape[1])*100
		im =plt.imshow(array,cmap = 'Paired')
		array =array[np.isfinite(array)]
		values = np.unique(array.ravel())
		colors = [im.cmap(im.norm(value)) for value in values]
		# create a patch (proxy artist) for every color 
		patches = [mpatches.Patch(color=colors[i], label="{} : {} %".format(self.Minerals_[i],str(round(proportion[i],2)))) for i in range(len(values)) ]
		# put those patched as legend-handles into the legend
		plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
		plt.grid(True)
		plt.title("Classificiation minéralogique - "+ self.prefix_ )
		plt.tight_layout()
		plt.show()

test = Mask('thin_section_1_','.bmp','Mask.csv',Type='images',Normalisation = True)
test.datacube_creation()
test.workingcube_creation()
test.plot_mineral_mask()








