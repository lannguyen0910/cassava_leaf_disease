# **🌿 Cassvana Leaf Classification with Pytorch Custom Template for Image Classification**
Kaggle Competition: https://www.kaggle.com/c/cassava-leaf-disease-classification


## **Dataset**
**Train set:** ~26,000 images (21367 images of the 2020 contest was merged with 500 images from the 2019 contest).
**Test set:** ~15,000 images.
**Public test:** 31% of the test set.
**Private test:** 69% of the test set.
**The dataset is imbalanced with 5 labels**

## **Requirements**

Python >= 3.8. Run this command to install all the dependencies:
```
pip install -r requirements.txt
```


## **Directories Structures**

```
  this repo
  └───  train_images                        
  │     └───  ***.png                    # Dataset folder   
  └───  test_images                        
  │     └───  ***.png              
  |
  └───  configs                 # Config folder                                          
  │     └─── train.yaml
  │     └─── test.yaml
  │     └─── config.py
  |              
  └─── csv                   # labels folder               
  │     └─── folds
  │         └─── fold_train.csv
  │         └─── fold_val.csv
  │                     
  └─── loggers                    # experiments folder               
  │     └─── runs
  │         └─── loss_fold
  |         └─── acc_fold        
  └─── weights                    # experiments folder               
  │     └─── model_name.pth    
  |     
  |            
  train.py
  test.py
```


## **Edit YAML**
**Full explanation on each YAML file**


## **Training**

Run this command and fine-tune on parameters for fully train observation (Require change)
```
python train.py --config=config_name   --resume=weight_path   --print_per_iters=100  --gradcam_visualization
```


## **Inference**

Run this command to generate predictions and submission file (Require fine-tune inside)
```
python test.py --config=test
```


## **To-do list:**

- [x] Multi-GPU support (nn.DataParallel)
- [x] GradCAM vizualization
- [x] Gradient Accumulation
- [x] Mixed precision
- [x] Stratified KFold splitting 
- [x] Inference with Ensemble Model and TTA
- [x] Metrics: Accuracy, Balanced Accuracy, F1-Score
- [x] Losses: Focal Loss, SmoothCrossEntropy Loss
- [x] Optimizer: AdamW, SGD, SAM (not debug yet)
- [x] Scheduler: ReduceLROnPlateau, CosineAnnealingWarmRestarts
- [x] Usable Models: Vit, EfficientNet, Resnext, Densenet
- [x] Early Stopping on training


## **Reference:**
- timm models from https://github.com/rwightman/pytorch-image-models
