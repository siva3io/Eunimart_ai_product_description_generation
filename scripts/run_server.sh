#!/bin/bash

# cd /home/ubuntu/vdezi_ai_product_description_generation    #replace service_repowith  your repo name 
#for gcp 
cd /home/azureuser/vdezi_ai_product_description_generation
git stash

git pull origin master

source ./venv/bin/activate

pip3 install -r requirements.txt

python3 serve.py



