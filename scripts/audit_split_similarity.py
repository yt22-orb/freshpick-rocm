"""Screen random-split visual similarity; dHash is a heuristic, not item identity."""
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

root=Path('prepared_dataset');out=Path('reports/evaluation')
rows=pd.read_csv('runs/mobilenetv3-finetuned/splits.csv')

def dhash(path):
 with Image.open(root/path) as source:
  pixels=np.asarray(source.convert('L').resize((9,8),Image.Resampling.LANCZOS))
 return (pixels[:,1:]>pixels[:,:-1]).ravel()

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  hashes=np.stack(list(pool.map(dhash,rows.image_path)))
 test=np.where(rows.split.eq('test'))[0];train=np.where(rows.split.eq('train'))[0]
 results=[]
 for block in np.array_split(test,16):
  distances=(hashes[block,None,:]!=hashes[train][None,:,:]).sum(2)
  for idx,distance in zip(block,distances):
   nearest=train[distance.argmin()]
   results.append({'test_sample_id':rows.iloc[idx].sample_id,'test_image_path':rows.iloc[idx].image_path,
    'test_class':rows.iloc[idx].class_name,'nearest_train_sample_id':rows.iloc[nearest].sample_id,
    'nearest_train_image_path':rows.iloc[nearest].image_path,'nearest_train_class':rows.iloc[nearest].class_name,
    'dhash_hamming_distance':int(distance.min())})
 result=pd.DataFrame(results).sort_values('dhash_hamming_distance');result.to_csv(out/'test-train-nearest-dhash.csv',index=False)
 summary={'method':'64-bit horizontal dHash on prepared RGB image converted to grayscale, 9x8 resize',
  'exact_source_pixel_hash_overlap':len(set(rows[rows.split.eq('test')].pixel_sha256)&set(rows[rows.split.eq('train')].pixel_sha256)),
  'test_n':len(test),'test_with_nearest_train_dhash_le_3':int((result.dhash_hamming_distance<=3).sum()),
  'test_with_nearest_train_dhash_le_5':int((result.dhash_hamming_distance<=5).sum()),
  'warning':'Visual-similarity screening only; hash collisions can be unrelated items and no verified item IDs exist.'}
 (out/'split-similarity.json').write_text(json.dumps(summary,indent=2));print(summary)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(8,2,figsize=(8,20))
 for axrow,(_,row) in zip(axes,result.head(8).iterrows()):
  for ax,column in zip(axrow,['test_image_path','nearest_train_image_path']):
   with Image.open(root/row[column]) as image:ax.imshow(image)
   ax.set_title(('Test' if column=='test_image_path' else 'Train')+f' · dHash distance {row.dhash_hamming_distance}',fontsize=10);ax.axis('off')
 fig.tight_layout();fig.savefig('runs/evaluation-web/nearest-train-test.jpg',dpi=110)
