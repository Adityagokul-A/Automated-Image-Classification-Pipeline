import asyncio
import requests
import os
from subprocess import call
from PIL import Image
from duckduckgo_search import AsyncDDGS
import yaml

with open('config.yml','r') as conf:
    config_info = yaml.load(conf, Loader=yaml.SafeLoader)

#print(config_info)

MAX_RESULTS_PER_CLASS = config_info['max_images_per_class']
DATASET_PATH = config_info['dataset_path']
#Create dataset folder(dir)
#find . -depth -name "*" -exec sh -c 'f="{}"; mv -- "$f" "${f%}.jpeg"' \; ->Rename files


if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

extra_descr = config_info['extra_search_descr']
classes = config_info['classes']
classes.sort()


headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
}

#todo : figure out a way to get corrupted downloads out of the way
async def save_image(i : int, class_name : str, url : str):
    #print(url)
    class_path = f"{DATASET_PATH}/{class_name}"
    img_filename = f"{class_path}/{class_name}{i}.jpeg"
    try:
        img_data = requests.get(url['image'], allow_redirects=True, headers=headers, stream=True).content
        with open(img_filename, 'wb') as handler:
            handler.write(img_data)
        img = Image.open(img_filename)
        img.verify()
        #if(img.format != 'RGBA'):       #handle palette images
        #    img.convert('RGBA').save(img_filename)
    except Exception as e:
        #print(e)
        if(os.path.isfile(img_filename)):
            os.remove(img_filename)

async def get_images(class_name: str, extra_descr: str):
    print(f"Data collection started for {class_name}")
    class_path = f"{DATASET_PATH}/{class_name}"
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    results = await AsyncDDGS().aimages(
        keywords=class_name + extra_descr +"image",
        region="wt-wt",
        safesearch="moderate",
        size=None,
        color=None,
        type_image="photo",
        layout=None,
        license_image=None,
        max_results=MAX_RESULTS_PER_CLASS,
    )
    saved_imgs = [save_image(i,class_name,result) for i,result in enumerate(results)]
    await asyncio.gather(*saved_imgs)
    print(f"Data collection done for {class_name} ...")

async def main():
    imgs = [get_images(c,extra_descr) for c in classes]
    await asyncio.gather(*imgs)
    print("Completed")

asyncio.run(main())

