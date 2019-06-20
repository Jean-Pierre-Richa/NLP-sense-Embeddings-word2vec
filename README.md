# Sense Embeddings

#### This project was done to learn sense embeddings instead of word embeddings, a full description can be found in the report

# Instructions
- Download EuroSense from http://lcl.uniroma1.it/eurosense/ .
- Download WordSimilarity-353 from http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/ .

# Running the code
- python train.py will execute the preprocessing to build the dictionary, train the network and test the correlation between the gold dataset and the learned embeddings from the model output.

- Check python train.py -h for more info about the arguments that can be used

- python similarity.py can be executed independently to test the correlation and draw the senses vectors on a 2d plane using PCA for dimensionality reduction.

- Check python similarity.py -h for more info about the arguments that can be used (e.g., python similarity.py --draw False)
