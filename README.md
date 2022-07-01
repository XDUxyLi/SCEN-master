# SCEN
Code For CVPR2022 paper "Siamese Contrastive Embedding Network for Compositional Zero-Shot Learningy"

# Usage
## **Requirements**<br>
environment.yml

## **Data Preparation**
**Mit-States**: Download from http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip<br>

# Training
## **MIT-States**
```python train.py --config configs/scen/mit/scen_cw.yml```<br>

## **Ut-Zappos**
```python train.py --config configs/scen/utzppos/scen_cw.yml```<br>

## **C-GQA**
```python train.py --config configs/scen/cgqa/scen_cw.yml```<br>
