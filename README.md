# A trafficking model for K<sub>v</sub>11.1 channels 

This model is part of the Journal of Physiology (2023) paper: 'In-silico analysis of the dynamic regulation of cardiac electrophysiology by K<sub>v</sub>11.1 ion-channel trafficking' by Stefan Meier, Adaïa Grundland, Dobromir Dobrev, Paul G.A. Volders, and Jordi Heijman. 
Doi: [10.1113/JP283976](https://physoc.onlinelibrary.wiley.com/doi/10.1113/JP283976)

:file_folder: The [MMT](https://github.com/HeijmanLab/IonChannelTraffickingModel/tree/main/MMT) folder contains the adapted O'Hara Rudy human ventricular cardiomyocyte model (ORd), wherein Clancy and Rudy's I<sub>Kr</sub> Markov model was implemented and adapted to model the acute effects of temperature on I<sub>Kr</sub>  gating. Moreover, the trafficking model with its modulators (i.e., temperature, drugs, and extracellular K<sup>+</sup> was also added to the ORd model. 

:file_folder: The [Q10](https://github.com/HeijmanLab/IonChannelTraffickingModel/tree/main/Q10) folder contains the Q<sub>10</sub> and V<sub>1/2</sub> optimization scripts. 

:file_folder: The [Data](https://github.com/HeijmanLab/IonChannelTraffickingModel/tree/main/Data) folder contains all the experimental data needed for the simulations (generally obtained with MyoKit's graph data extractor). 

:file_folder: The [Figures](https://github.com/HeijmanLab/IonChannelTraffickingModel/tree/main/Figures) folder is a results folders where some of the figures will be stored. 

:computer: :snake: The Python script to create the simulations and figures used in the paper can be found in [TraffickingModelFinal](https://github.com/HeijmanLab/IonChannelTraffickingModel/blob/main/TraffickingModelFinal.py).

:computer: :snake: The functions used for the above-mentioned simulations can be found in [TraffickingModelFunctions](https://github.com/HeijmanLab/IonChannelTraffickingModel/blob/main/TraffickingModelFunctions.py).


## Virtual environment (Instructions for Anaconda):

Follow the below mentioned steps to re-create te virtual environment with all the correct package versions for this project.

:exclamation: **Before creating a virtual environment please make sure you fully installed myokit (v. 1.33.0)[^1] already. Please follow these steps carefully: http://myokit.org/install.** :exclamation:
[^1]: Note, you can downgrade MyoKit's version in the Anaconda prompt by writing: `pip install myokit==1.33.0`

***1. Clone the repo:***

`git clone git@github.com:HeijmanLab/IonChannelTraffickingModel.git`

***2. Create virtual environment:***

This re-creates a virtual environment with all the correct packages that were used to create the model and run the simulations. 

- Set the directory:

`cd IonChannelTraffickingModel`

- Create the corresponding environment:

`conda env create -f TraffickingModel.yml`

- Activate the environment:

`conda activate traffickingmodel`

***3. Setup and launch spyder (or any other IDE of your liking) from your anaconda prompt:***

`spyder`

