# Detekcia tváre
---
Príprava prostredia, inštalácia hlavných knižníc:

```python
pip install -r requirements.txt
```
## Model
  Dlib model dostupný v knižnici po inštalácii:

  ```python
  pip install dlib
  ```
  
  MTCNN model dostupný v knižnici po inštalácii:

  ```python
  pip install mtcnn
  ```

## Dataset
  Anotácia pravdivej zložky prebehla pomocou CVAT nástroja. Bolo označených 250 forografii, na ktorých bolo 1903 tvári. 

## Párovanie
  Párovanie pravdivej a výstupnej zložky detektora prebehlo pomocou naimplementovaných funkcii. Na párovanie sme vytvorili korelačnú maticu. 
  
## Vizualizácia
  
  Na vizualizáciu výsledkov sme využili knižnicu *Plotly* [5](https://plotly.com/python/).  

## Zdroj
[1] https://github.com/opencv/cvat

[2] https://github.com/deepinsight/insightface

[3] http://dlib.net/

[4] https://github.com/ipazc/mtcnn

[5] https://plotly.com/python/
