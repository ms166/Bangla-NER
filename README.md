## Usage
* Use python version <= 3.8

```
git clone https://github.com/ms166/Bangla-NER
cd Bangla-NER
```
* Install dependencies
```
cat requirements.txt | xargs -n 1 pip install
```
* Download dataset
```
python download-dataset.py
```
* Start training
```
python run.py
```

Training time is around 3 hours on Kaggle.
