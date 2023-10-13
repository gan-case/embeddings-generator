import logging
import threading
from concurrent.futures import ProcessPoolExecutor
import gdown
from deepface import DeepFace
from tqdm import tqdm
import json
import os
import glob
import tarfile

def gen_embedding_json(i):
  try:
    logging.info("Started generation for %s", i)
    embedding_json = {}
    embedding_json['image_name'] = i
    image_embeddings = open("embeddings/" + i[:-4] + ".json", "w+")
    embedding_objs = DeepFace.represent(img_path = i)
    embedding_json.update(embedding_objs[0])
    image_embeddings.write(json.dumps(embedding_json))
    image_embeddings.close()
    logging.info("Finished generation for %s", i)
  except:
    logging.error("error at %s", i)

dataset_id = "1JWTqMEiUZ2yNUJJl_5Ctq8SuskVocn51"
output = "CACD2000_refined.tar"

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

logging.info("Downloading Dataset")
gdown.download(id=dataset_id, output=output, quiet=False)

logging.info("Extracting Dataset")
dataset = tarfile.open(output)
dataset.extractall()
dataset.close


os.chdir('CACD2000')
images = glob.glob('*.jpg')
os.makedirs("embeddings", exist_ok=True)


logging.info("Now Processing %d images", len(images))

pool = ProcessPoolExecutor(max_workers=5)

results = pool.map(gen_embedding_json, images)
pool.shutdown(wait=True)



