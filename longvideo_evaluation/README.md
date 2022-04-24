
# Evaluation Codes
This script is designed to compute the J and F scores where the annotations are sampled. In other words, the groundtruths are not provided for every frame.

## Usages
```python
python3 score_longvideo.py \
        --path /path/to/longvideo_dataset \
        --results_path ../output/AFB-URR_LongVideo \
        --method AFB-URR \
        --update
````
### Our results based on the pretrained model
Global results for AFB-URR

|  J&F-Mean |  J-Mean | J-Recall |  J-Decay  | F-Mean|F-Recall  |F-Decay|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.832571 | 0.827035 | 0.916667 | 0.115116  |0.838107   |0.916667 | 0.138554|

Per sequence results for AFB-URR

| Sequence  |  J-Mean  | F-Mean|
| ---- | ---- | ---- |
| rat_1 | 0.745197 | 0.763893|
|dressage_1 | 0.834371 | 0.874584|
|blueboy_1 | 0.901536 | 0.875845|



## Copyright
These codes are modified from DAVIS VOS Evaluation. 
If you use this code in your work, please cite both DAVIS dataset and our paper.