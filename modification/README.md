# Modifikácia
---
Príprava prostredia, inštalácia hlavných knižníc:

```python
pip install -r requirements.txt
```
## Dataset
  Vytvorenie trénovacích/validačných/testovacích dát prebehlo pomocou CVAT nástroja. 
  Odporučanie využiť dataset s variabilným súborom farebných forografii napr. *ImageNet* [Link](http://www.image-net.org/), *COCO* [Link](http://cocodataset.org/) a pod.

## Trénovanie
  Proces trénovania je implementovaný pomocou interaktívneho Jupyter prostredia [trainColorModel.ipynb].
  
  **Hardvér**
  Veľkosť trénovacích batchov je priamoúmerne spojená so záťažou pämate GPU/RAM. Náše trénvoanie prebehlo na vysoko-výkonných server grafike/procesore. 

## Model
  Výstupný model generátora/diskriminátora je dostupný na stiahnutie [Generátor](https://drive.google.com/file/d/12TbZ-Ztr8QPpl5uk8_tytKHscwFkPOFS/view?usp=sharing), [Diskriminátor](https://drive.google.com/file/d/1LABuD-3iXwWnaFwWp5CRYhN7CXprOL3D/view?usp=sharing). 
  
  Použitie generátora a výstupná transformácia sa nachádza v [runColorization.ipynb]

## Zdroj
[1] https://www.fast.ai/

[2] https://github.com/opencv/cvat

[3] https://github.com/jantic/DeOldify
