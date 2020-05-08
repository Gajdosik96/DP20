# Detekcia tváre
---
Príprava prostredia, inštalácia hlavných knižníc:

```python
pip install -r requirements.txt
```
Stiahnutie repozitára na detekciu tváre pomocou konvolúcie:

```python
git clone https://github.com/deepinsight/insightface
```

## Dataset
  Anotácia pravdivej zložky prebehla pomocou CVAT nástroja. Bolo označených 250 forografii, na ktorých bolo 1903 tvári. 

## Párovanie
  Párovanie pravdivej a výstupnej zložky detektora prebehlo pomocou naimplementovaných funkcii. Na párovanie sme vytvorili korelačnú maticu. 
  
  **Vizualizácia**
  
  Na vizualizáciu výsledkov sme využili knižnicu *Plotly* [4](https://plotly.com/python/).  

## Zdroj
[1] https://github.com/opencv/cvat

[2] https://github.com/deepinsight/insightface

[2] http://dlib.net/
