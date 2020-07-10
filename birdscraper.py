# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:17:43 2020

@author: Pat
"""

import selenium 
from selenium import webdriver
import os
import io
import requests
import hashlib
import time
from PIL import Image


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                time.sleep(0.05)
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        pass
        #print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB').resize((448,448), 3)
        width, height = 448, 448
        new_width, new_height = 224, 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        image = image.crop((left, top, right, bottom))
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        pass
        #print(f"ERROR - Could not save {url} - {e}")
        
def search_and_download(search_term:str,driver_path:str,training_path=None, test_path= None,number_images=5):
    training_folder = os.path.join(training_path,'_'.join(search_term.lower().split(' ')))
    test_folder = os.path.join(test_path,'_'.join(search_term.lower().split(' ')))
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term+' close up', number_images, wd=wd, sleep_between_interactions=0.5)
        
    for n, elem in enumerate(res):
        if n < 3*len(res)/4:
            persist_image(training_folder,elem)
        else:
            persist_image(test_folder,elem)
        
        
if __name__ == "__main__":
    driverpath = 'C:\\Users\\Pat\\.spyder-py3\\chromedriver.exe' 
    classes = ['House sparrow', 'Great tit', 'Eurasian blue tit', 'Eurasian magpie', 'Eurasian jay']
    training_path = 'ImageData\\training_set\\'
    test_path = 'ImageData\\test_set\\'
    wd = webdriver.Chrome(executable_path=driverpath)
    for bird in classes:
        search_and_download(search_term = bird, driver_path = driverpath, training_path = training_path, test_path = test_path, number_images=500)
        
        