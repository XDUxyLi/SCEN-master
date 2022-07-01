# SCEN
Code For CVPR2022 paper "Siamese Contrastive Embedding Network for Compositional Zero-Shot Learningy"

# Usage
## **Requirements**<br>
environment.yml

## **Data Preparation**
bash ./utils/download_data.sh DATA_ROOT<br>
  python general_main.py --data  cifar100 --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 1000 --dl_weight 4.0mkdir logs
