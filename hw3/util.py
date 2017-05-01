import os
import numpy as np
import skimage
import skipthoughts

def parse_raw_tag_dict(tag_file):
    '''
    raw_tag_dict: a dict of all image dicts
        - one image has a dict to collect its all tags.
    '''
    f = open(tag_file,"r")
    raw_tag_dict = {}
    for line in f:
        this_dict = {}
        # separate id and tag
        id, context = line.split(sep=",")
        # parse tag
        detail = context.split(sep="\t")
        for des in detail:
            if(des != "\n"):
                key, num = des.split(sep=":")
                this_dict[key.strip()] = float(num)
        # add into tag dictionary
        raw_tag_dict[int(id)] = this_dict
    return raw_tag_dict

def get_tag_dict(raw_tag_dict,wanted_tag=["hair","eye"]):
    '''
    tag_dict_in_use: a dict in use
        (key,value) = (id, a list of description containing wanted_tag)
    '''
    get_desc = lambda tag, keys: [key for idx,key in enumerate(keys) if tag in key]
    tag_dict_in_use = {}
    for i in raw_tag_dict.keys():
        this_desc = ["none"] * len(wanted_tag)
        keys = list(raw_tag_dict[i].keys())
        for j in range(len(wanted_tag)):
            this_desc[j] = " ".join(get_desc(wanted_tag[j],keys)).lower()
        tag_dict_in_use[i] = " and ".join(this_desc)
    return tag_dict_in_use

def get_image_tag_pair(tag_dict_in_use,img_path="data/faces/"):
    print("start loading skipthoughts model")
    # get text vector
    model = skipthoughts.load_model()
    list_text = skipthoughts.encode(model,list(tag_dict_in_use.values()))
    
    # get image
    list_image = []
    for key,item in tag_dict_in_use.items():
        img = skimage.io.imread(os.path.join(img_path,str(key)+".jpg"))
        img = skimage.transform.resize(img,(64,64))
        list_image.append(img)
    return list_image, list_text
