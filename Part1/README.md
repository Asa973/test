# part 1. Classification model?

 Approche 1 : EfficientNetB0 (transfer learning)
	 Résultat : précision test ~33%
 
 Approche 2 : CNN personnalisé
     	3 couches convolutives + pooling

    	Flatten + Dense + Dropout + Softmax

    	Data augmentation légère

    	Apprentissage depuis zéro

	Résultat : précision test ~60%.
	
## Bibliothèques à installer :

`pip install tensorflow matplotlib pillow`

EfficientNetB0		~33%
CNN personnalisé	~60%

## notion

CNN (Convolutional Neural Network)

Couches : blocs fondamentaux d’un réseau (Conv2D, Dense...)

Epochs : 1 epoch = 1 passe sur tout l'ensemble d'entraînement

Data augmentation : techniques pour enrichir les données disponibles (flip, zoom, rotation, etc.)

![CNN](images/explication1.jpg)
source : https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

![CNN](images/Figure_1.png)
source : moi

![CNN](images/Figure2.png)
source : moi

## links
https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide

https://datascientest.com/convolutional-neural-network

https://github.com/Bengal1/Simple-CNN-Guide

https://stackoverflow.com/questions/78719585/keras-model-input-shapes-are-incompatible-with-the-layer-despite-being-a-compat

https://fr.mathworks.com/discovery/convolutional-neural-network.html

https://www.ibm.com/fr-fr/think/topics/convolutional-neural-networks
