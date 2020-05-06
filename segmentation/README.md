# Segmentácia

Príprava prostredia, inštalácia hlavných knižníc:

```python
pip install -r requirements.txt
```
## Dataset
  Vytvorenie trénovacích/validačných/testovacích dát prebehlo pomocou CVAT nástroja. Použitý súkromný dataset.

## Trénovanie
  Proces trénovanie je implementovaný pomocou interaktívneho Jupyter prostredia [UNET-dp.ipynb].

## Model
  Natrénované modely pre veľkosť vstupu (128x128) [Model1](https://drive.google.com/file/d/12TbZ-Ztr8QPpl5uk8_tytKHscwFkPOFS/view?usp=sharing) a (512x512) [Model2](https://drive.google.com/file/d/1LABuD-3iXwWnaFwWp5CRYhN7CXprOL3D/view?usp=sharing). 
  
  Na výstup bol aplikovaný postprocessing pomocou morfologických operácii.

## Zdroj
[1] https://www.fast.ai/

[2] https://github.com/opencv/cvat
