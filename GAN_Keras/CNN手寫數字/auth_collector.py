import requests
from PIL import Image
import os
import time

url = "https://nportal.ntut.edu.tw/authImage.do"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:65.0) Gecko/20100101 Firefox/65.0'}

img_original_dir = 'img_original/'
img_gray_dir = 'img_gray/'
img_two_dir = 'img_two/'


def init_dir():
    ''' Init the directory for saving images '''
    if not os.path.exists(img_original_dir):
        os.makedirs(img_original_dir)

    if not os.path.exists(img_gray_dir):
        os.makedirs(img_gray_dir)

    if not os.path.exists(img_two_dir):
        os.makedirs(img_two_dir)


def download_auth_img():
    ''' Download auth image and save it '''
    for index in range(0, 1000):
        response = requests.get(url, headers, stream=True)
        response.raw.decode_content = True
        save_img(index, response.raw)


def save_img(index, img_binary):
    ''' Save binary as different type of image.'''
    img = Image.open(img_binary)
    img.save("{}{}.png".format(img_original_dir, index))

    img_gray = img.convert('L')
    img_gray.save("{}{}.png".format(img_gray_dir, index))

    img_two = img_gray.point(lambda x: 255 if x > 140 else 0)
    img_two.save("{}{}.png".format(img_two_dir, index))

    print("正在下載第 {} 張驗證碼".format(index))
    time.sleep(1)


if __name__ == "__main__":
    init_dir()
    download_auth_img()
