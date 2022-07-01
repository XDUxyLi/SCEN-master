# SCEN
Code For CVPR2022 paper "Siamese Contrastive Embedding Network for Compositional Zero-Shot Learning"

## Usage
### **Requirements**<br>
environment.yml

### **Data Preparation**
**Mit-States**: Download from http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip<br>
**Utzappos**: Download from http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip<br>
**C-GQA**: Download from https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa-updated.zip<br>

## Training 
### **MIT-States**
```
python train.py --config configs/scen/mit/scen_cw.yml
```

### **Ut-Zappos**
```
python train.py --config configs/scen/utzppos/scen_cw.yml
```

### **C-GQA**
```
python train.py --config configs/scen/cgqa/scen_cw.yml
```

## References
If you use this code, please cite
```
@inproceedings{li2022siamese,
  title={Siamese Contrastive Embedding Network for Compositional Zero-Shot Learning},
  author={Li, Xiangyu and Yang, Xu and Wei, Kun and Deng, Cheng and Yang, Muli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9326--9335},
  year={2022}
}
```
