# Code for paper "Efficient Graph Similarity Search by Learning with Landmarks"

## requirements
See "requirements.txt". 
Please following the https://pytorch-geometric.readthedocs.io/en/1.7.1/notes/installation.html to install the torch_geometric==1.7.1.

## Run our code
To train our SGim : 
`
python train.py --datasets AIDS700nef --gpu 0 --model SGim --name full --mode full

python train.py --datasets LINUX --gpu 0 --model SGim --name full --mode full

python train.py --datasets IMDBMulti --gpu 0 --model SGim --name full --mode full
`
To train our SLL after training the SGim:
`
python landmark_xgboost.py --datasets AIDS700nef

python landmark_xgboost.py --datasets LINUX

python landmark_xgboost.py --datasets IMDBMulti
`