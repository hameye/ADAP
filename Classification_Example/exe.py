
from ClassMask import *

#crée l'instance de mask
cm = Mask('lead_ore_','.bmp','Mask.xlsx')
#charge le tableur associés
cm.load_table()
#crée le cube des cartes chimiques
cm.datacube_creation()
#crée le cube minéralogique
cm.mineralcube_creation()
#et on sort la carte minéralogique
cm.plot_mineral_mask()