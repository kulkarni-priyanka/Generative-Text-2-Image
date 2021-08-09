# dependencies
import os
import json
import numpy as np
import pandas as pd
import re
import time
import nltk
import re
import string
#from tensorlayer import * as tl
import tensorlayer as tl
#import .utils
import imageio

cwd = "D:/My_Stuff/JHU_Stuff/625.742/coco/annotations/"
caption_dir = cwd #os.path.join(, 'annotations_trainval2014/annotations/')
img_dir =  "D:/My_Stuff/JHU_Stuff/625.742/coco/annotations/train2014/train2014/"
VOC_FIR = cwd + '/vocab.txt'

with open(caption_dir + 'instances_train2014.json') as json_data:
    inst = json.load(json_data)

# annotations
anns = pd.DataFrame.from_dict(inst['annotations'])

# categories
cats = pd.DataFrame.from_dict(inst['categories'])

with open(caption_dir + 'captions_train2014.json') as json_data:
    caps = json.load(json_data)

imagerefs = pd.DataFrame.from_dict(caps['images'])
captions = pd.DataFrame.from_dict(caps['annotations'])

# combine
textdf = (anns.loc[:, ['image_id', 'category_id']]
          .merge(cats.rename(columns={'id': 'category_id', 'name': 'category'}), how='inner', on='category_id')
          .merge(imagerefs.loc[:, ['file_name', 'height', 'width', 'id']].rename(columns={'id': 'image_id'}),
                 how='inner', on='image_id')
          .merge(captions.rename(columns={'id': 'caption_id'}), how='inner', on='image_id')
          .drop_duplicates()
          .reset_index(drop=True)
          )

# subset for easier training
imgIDs = textdf.loc[textdf.category == 'bus', 'image_id'].unique() #textdf['image_id'].unique()
textdf = textdf.loc[textdf.image_id.isin(imgIDs), :].reset_index(drop=True)
capIDs = textdf['image_id'].unique()


## format training captions
captions_dict = {}
processed_capts = []
for t in capIDs:
    caps = []
    caplist = textdf.loc[textdf.image_id==t, 'caption'].tolist()
    for cap in caplist:
        cap = re.sub('[%s]' % re.escape(string.punctuation), ' ', cap.rstrip())
        cap = cap.replace('-', ' ')
        caps.append(cap)
        processed_capts.append(tl.nlp.process_sentence(cap, start_word='<S>', end_word='</S>'))
    assert(len(caps) == 10, "Every image have 10 captions")
    captions_dict[t] = caps
print(" * %d x %d captions found " % (len(captions_dict), len(caps)))


## build vocab
# if not os.path.isfile('vocab.txt'):
print("Hello")
print(VOC_FIR)
_ =  tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1) #tl.nlp.
# else:
#     print("WARNING: vocab.txt already exists")
vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")


## store all captions ids in list
captions_ids = []
try: # python3
    tmp = captions_dict.items()
except: # python3
    tmp = captions_dict.iteritems()
for key, value in tmp:
    for v in value:
        captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
        # print(v)              # prominent purple stigma,petals are white inc olor
        # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
        # exit()
captions_ids = np.asarray(captions_ids)
print(" * tokenized %d captions" % len(captions_ids))

## load training images
imgs_title_list = textdf['file_name'].unique().tolist()

print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
s = time.time()

images = []
images_256 = []
for name in imgs_title_list:
    # print(name)
    img_raw = imageio.imread( os.path.join(img_dir, name) )
    img = tl.prepro.transform.resize(img_raw, output_shape=[64, 64])    # (64, 64, 3)
    img = img.astype(np.float32)
    images.append(img)
#     if need_256:
#         img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
#         img = img.astype(np.float32)

    images_256.append(img)
# images = np.array(images)
# images_256 = np.array(images_256)
print(" * loading and resizing took %ss" % (time.time()-s))


n_images = len(captions_dict)
n_captions = len(captions_ids)
n_captions_per_image = len(caps) # 10
print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

train_count = int(0.80*n_images)
test_count = n_images - train_count

captions_ids_train, captions_ids_test = captions_ids[: train_count*n_captions_per_image], captions_ids[train_count*n_captions_per_image :]
images_train, images_test = images[:train_count], images[train_count:]

print(len(images_test)) #to check the lenght of the list with elements having different shape

for idx,item in enumerate(images_test):
  if item.shape!=(64,64,3):
      print(str(item.shape))
      images_test.remove(item)
      captions_ids_test = np.delete(captions_ids_test,np.s_[idx:idx+n_captions_per_image],axis=0) #captions_ids_test[idx:idx+n_captions_per_image]

for idx,item in enumerate(images_train):
  if item.shape!=(64,64,3):
      print(str(item.shape))
      images_train.pop(idx)
      captions_ids_train = np.delete(captions_ids_train, np.s_[idx:idx + n_captions_per_image],axis=0)  # captions_ids_test[idx:idx+n_captions_per_image]


n_images_train = len(images_train)
n_images_test = len(images_test)
n_captions_train = len(captions_ids_train)
n_captions_test = len(captions_ids_test)

print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))


import pickle
def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

m = textdf['caption'].unique()
print(*m, sep = "\n")

save_all(vocab, '_vocab.pickle')
save_all(images_train, '_image_train.pickle')
save_all(images_test, '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')


