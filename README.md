# SCEN
Code For CVPR2022 paper "Siamese Contrastive Embedding Network for Compositional Zero-Shot Learningy"

# Usage
## **Requirements**<br>
environment.yml

## **Data Preparation**
```bash ./utils/download_data.sh DATA_ROOT```

# Training
## **MIT-States**
```python train.py --config configs/scen/mit/scen_cw.yml```<br>

## **Ut-Zappos**
```python train.py --config configs/scen/utzppos/scen_cw.yml```<br>

## **C-GQA**
```python train.py --config configs/scen/cgqa/scen_cw.yml```<br>
