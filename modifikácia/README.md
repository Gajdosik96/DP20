# Modifikácia
---
Príprava prostredia, inštalácia hlavných knižníc:

```python
pip install -r requirements.txt
```
## Dataset
  Odporučanie využiť dataset s variabilným súborom farebných forografii napr. *ImageNet* [[Link](http://www.image-net.org/)], *COCO* [[Link](http://cocodataset.org/)] a pod.

## Trénovanie
  Proces trénovania je implementovaný pomocou interaktívneho Jupyter prostredia [trainColorModel.ipynb].
  
  **Hardvér**
  
  Veľkosť trénovacích batchov je priamoúmerne spojená so záťažou pämate GPU/RAM. Náše trénvoanie prebehlo na vysoko-výkonných server grafike/procesore. 

## Model
  Výstupný model generátora/diskriminátora je dostupný na stiahnutie [[Generátor](https://drive.google.com/file/d/1gq3-xgmZWm6hDBZEdW-FOqwrfnpnyfjj/view?usp=sharing)], [[Diskriminátor](https://drive.google.com/file/d/1xcs9QckDG95t8k03pYtUiTaPFZy1CwVx/view?usp=sharing)]. 
  
  Použitie generátora a výstupná transformácia sa nachádza v [runColorization.ipynb]

## Zdroj
[1] https://www.fast.ai/

[2] https://github.com/opencv/cvat

[3] https://github.com/jantic/DeOldify
