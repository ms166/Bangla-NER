import gdown
import os

url = 'https://drive.google.com/uc?id=1k6ZB3tGgzeSXtc4BC2kN4H1UZDumu19I'
os.system('mkdir data')

output = 'data/Bangla-NER-Splitted-Dataset.json'

gdown.download(url, output, quiet=False)
