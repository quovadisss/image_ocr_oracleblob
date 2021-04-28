# -*- coding: utf-8 -*-
# Crawling
import urllib
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
# Oracle
import cx_Oracle
# text detect
from imutils.object_detection import non_max_suppression
import cv2
# etc
import random
import time
import csv                                              # csv 파일 관련
import os
import re
# file I/O 관련
from io import BytesIO
from tqdm import tqdm                   # 진행바 (그래픽관련)
import shutil                                          # 디렉토리 삭제
from PIL import Image
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def crawling():
    # Initial Set
    folder = ".image/"   # 저장될 폴더 이름
    baseUrl = "https://www.google.com/search" # Base url (Google)
    webDriver = "./chromedriver.exe"  
    searchItem = input('검색어를 입력하세요:') 
    strsize = input('Page Down 횟수를 입력하세요:') 
    size = int(strsize)

    params = {
       "q": searchItem,
        "tbm": "isch",
        "sa": "1",
        "source": "lnms&tbm=isch"
    }
    url = baseUrl+"?"+urllib.parse.urlencode(params)

    # # Option: 브라우저 숨기기(백그라운드 실행)
    # options = webdriver.ChromeOptions()
    # options.add_argument('headless')
    # options.add_argument('disable-gpu')

    # 웹사이트 접속
    # browser = webdriver.Chrome(webDriver, options=options)
    browser = webdriver.Chrome(webDriver)
    browser.implicitly_wait(1)
    browser.get(url)                 
    html = browser.page_source 
    browser.implicitly_wait(1)

    # 검색된 페이지에 대한 이미지 갯수 가져오기
    soup_temp = BeautifulSoup(html, 'html.parser')
    # 한번의 parsing 에 이미지가 얼마나 있는지 이미지 갯수 파악
    img4page = len(soup_temp.findAll("img"))            
    print("Img Num: ", img4page)

    # 페이지 다운 (이미지를 계속 찾기 위함)
    element = browser.find_element_by_tag_name("body")

    imgCnt =0
    while imgCnt < size * img4page:
        element.send_keys(Keys.PAGE_DOWN)   # Page down key event
        rnd = random.random()
        print(imgCnt)
        imgCnt+=img4page
        try:
            browser.find_element_by_class_name("mye4qd").click()
        except:
            time.sleep(rnd)

    # 스크롤 다 내린 후 html 재파싱
    html = browser.page_source
    img_soup = BeautifulSoup(html, 'html.parser')
    small_img = img_soup.find_all('img', attrs={'class': 'rg_i Q4LuWd'})

    small_img_urls = []  # small image urls
    for img in small_img:
        try:
            small_img_urls.append(img['src'])
        except:
            small_img_urls.append(img['data-src'])

    browser.close()

    # # 저장 폴더 생성 및 이미지 저장
    saveDir = folder + searchItem   # 이미지 저장 디렉토리
    try:
        if not(os.path.isdir(saveDir)):
            os.makedirs(os.path.join(saveDir))
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # 이미지 저장
    print("\r\n")
    print(" ==================Progress (Small image save) ========================")

    for i, src in zip(tqdm(range(len(small_img_urls))), small_img_urls):
        try:
            urllib.request.urlretrieve(src, saveDir + "/" + str(i) + ".jpg")
        except:
            continue

    print('Job is done')

    return img_soup, url, searchItem, len(small_img_urls)


# ========================== Text detection ==========================
# Get all image path
def img_path(search_chr):
    # Load_images
    image_name = os.listdir(r'C:\PycharmProjects\DB_teamproject\.image\{}'.format(search_chr))
    abs_path = 'C:/PycharmProjects/DB_teamproject/.image/{}/'.format(search_chr)
    east_path = 'C:/PycharmProjects/DB_teamproject/opencv-text-detection'

    image_path = []
    for i in image_name:
        image_path.append(abs_path + i)

    return image_path, east_path


# GET text box using openCV
def text_detect(image_path, east_path):
    # load the input image and grab the image dimensions
    image = cv2.imread(image_path)
    orig = image.copy()

    # opening(removing noise) : erosion followed by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # morph gradient
    # image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_path + '/frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    roi_all = []
    for index, (startX, startY, endX, endY) in enumerate(boxes):
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        roi = orig[startY:endY, startX:endX]

        roi_all.append(roi)

    return roi_all


# get images that contain text
def img_iter(search_chr):
    image_path, east_path = img_path(search_chr)
    check_text = []
    os.makedirs(os.path.join('opencv-text-detection/box/' + search_chr))
    for i, path in zip(tqdm(range(len(image_path))), image_path):
        roi_all = text_detect(path, east_path)
        file_name = path.split('/')[5].split('.')[0]
        if len(roi_all) > 0:
            check_text.append(1)
            for ind, roi in enumerate(roi_all):
                try:
                    cv2.imwrite('opencv-text-detection/box/' + search_chr + '/{0}_{1}.jpg'.format(file_name, ind), roi)
                except cv2.error:
                    continue
        else:
            check_text.append(0)

    return check_text


# ========================== Crawl big image ==========================
def get_final_img_property(html_source, url, search_chr):
    folder = '.bigimage/'
    url = url
    each_image_parse = html_source.find_all('div', attrs={'class': 'isv-r PNCib MSM1fd BUooTd'})
    webDriver = "./chromedriver.exe"
    browser = webdriver.Chrome(webDriver)
    browser.get(url)

    tags = []
    image_url = []
    image_width = []
    image_height = []
    image_title = []
    image_source = []

    for count, i in zip(tqdm(range(len(each_image_parse))), each_image_parse):
        new_url = url + '#imgrc=' + i['data-id']
        browser.get(new_url)
        time.sleep(3)  # image loading 시간 필요
        html = browser.page_source
        img_soup = BeautifulSoup(html, 'html.parser')
        tag = img_soup.find('div', attrs={'class': 'A8mJGd'})
        tags.append(tag)

    for j in tags:
        tag2 = j.find_all('img', attrs={'class': 'n3VNCb'})
        tag3 = j.find_all('span', attrs={'class': 'VSIspc'})
        if len(tag2) == 3:
            image_url.append(tag2[1]['src'])
            image_size_text = tag3[1].text
        else:
            image_url.append(tag2[0]['src'])
            image_size_text = tag3[0].text
        image_size = image_size_text.split(' ')
        image_width.append(int(image_size[0]))
        image_height.append(int(image_size[2]))

        tag4 = j.find('a', attrs={'class': 'Beeb4e'})
        image_title.append(tag4.text)
        image_source.append(tag4['href'])

    image_title = [re.sub('[^a-zA-Z0-9]', ' ', i).strip() for i in image_title]

    browser.close()


    # # 저장 폴더 생성 및 이미지 저장
    saveDir = folder + search_chr     # 이미지 저장 디렉토리
    try:
        if not(os.path.isdir(saveDir)):
            os.makedirs(os.path.join(saveDir))
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # 이미지 저장
    print("\r\n")
    print(" ==================Progress (Save) ========================")

    image_index = []
    for i, src in zip(tqdm(range(len(image_url))), image_url):
        try:
            urllib.request.urlretrieve(src, saveDir+"/" + str(i) + ".jpg")
            image_index.append(i)
        except:
            continue

    # csv 파일 저장 (생략 가능)
    # csv_file = open(saveDir + "/" + searchItem + '.csv', 'w', encoding='utf-8')
    # wr = csv.writer(csv_file)
    # for i, src, width, height in zip(range(fileNum), src_ImgURL, img_width, img_height):
    #     wr.writerow([i, src, width, height])

    print("크롤링이 끝났습니다.")

    return folder, search_chr, image_width, image_height, image_title, image_source, image_index


def database_upload(folder, searchItem, image_width, image_height, image_title, image_source, img_txt_contain):
    db_searchItem = searchItem
    db_width = image_width
    db_height = image_height
    db_title = image_title
    db_source = image_source
    db_text = img_txt_contain

    # 오라클 접속
    # connect("사용자 ID", "Password", "server")
    connect = cx_Oracle.connect("system", "oraclepractice", "localhost/orcl")
    cursor = connect.cursor()

    # 버전 확인 쿼리
    query = "select * from PRODUCT_COMPONENT_VERSION"
    cursor.execute(query)
    print("Version CHECK ============================")
    version_result = cursor.fetchall()                                  # fetchall = read 결과 확인
    print(version_result)
    print("=======================================")

    # 테이블 존재 확인
    query = "select count(*) from all_tables where table_name = 'DB_IMAGE'"
    cursor.execute(query)
    for row in cursor:
        table_existence_result = (int(row[0]))


    # 테이블 존재 확인 후 없으면 테이블 생성
    if table_existence_result == 0:
        query_table = "create table DB_IMAGE(\
                            IMG_SEARCHITEM varchar2(50), \
                            IMG_WIDTH number, \
                            IMG_HEIGHT number, \
                            IMG_TITLE varchar2(500), \
                            IMG_SOURCE clob, \
                            IMG_TEXT number, \
                            IMG_PICTURE blob)"
        cursor.execute(query_table)

    # # 저장 폴더에서 img 가져온 후 데이터 인코딩(바이너리)
    img_file_list = os.listdir(folder + "{}/".format(db_searchItem))
    paths = []
    for h in img_file_list:
        paths.append(folder + "{}/".format(db_searchItem) + h)
    for k, image in enumerate(paths):
        im = Image.open(image)
        buffer = BytesIO()
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(buffer, format='jpeg')
        img_str = base64.b64encode(buffer.getvalue())
        # SQL
        query_insert = "INSERT INTO DB_IMAGE(IMG_SEARCHITEM, IMG_WIDTH, IMG_HEIGHT, IMG_TITLE, IMG_SOURCE, \
                                    IMG_TEXT, IMG_PICTURE) \
                        VALUES (:item, :width, :height, :title, :clob, :text, :blob)"
        print(db_searchItem, db_width[k], db_height[k], db_title[k], db_source[k], db_text[k])
        cursor.execute(query_insert, (db_searchItem, db_width[k], db_height[k], db_title[k], db_source[k], db_text[k], img_str))
        connect.commit()

    # 이미지 디렉토리 삭제 (생략 가능)
    # folder = ".image/"
    # shutil.rmtree(folder)
    #
    # cursor.close()
    # connect.commit()
    # connect.close()


# ===================================================================================
# Subject 테이블 만들기
def Create_subject_table():
    connect = cx_Oracle.connect("system", "oraclepractice", "localhost/orcl")
    cursor = connect.cursor()

    # 테이블 존재 확인
    query = "select count(*) from all_tables where table_name = 'DB_IMAGE_SUBJECTS'"
    cursor.execute(query)
    for row in cursor:
        table_existence_result = (int(row[0]))

    # DB_IMAGE_SEARCH 테이블 만들기
    if table_existence_result == 0:
        query = "create table DB_IMAGE_SUBJECTS(\
                                   SUBJECT varchar2(50),\
                                   IMG_SEARCHITEM varchar2(50) PRIMARY KEY)"
        cursor.execute(query)

        # DB_IMAGE_SERACH 값
        subjects = ['Nature', 'Animal', 'Flower']
        search_chr = [['tree', 'ocean', 'sky', 'rain', 'sunset'],
                      ['sheep', 'lion', 'elephant', 'bird', 'giraffe'],
                      ['rose', 'forsythia', 'rose of sharon', 'tulip', 'daisy']]

        # DB_IMAGE_SERACH 값 넣기
        for subj, search_chr in zip(subjects, search_chr):
            for i in search_chr:
                query2 = "insert into DB_IMAGE_SUBJECTS(SUBJECT, IMG_SEARCHITEM) values (:subject,:search)"
                subject = subj
                search = i
                cursor.execute(query2, (subject, search))
                connect.commit()
    else:
        pass
# ===================================================================================

# ===================================================================================
# 테이블 foreign key 생성
def alter_table():
    connect = cx_Oracle.connect("system", "oraclepractice", "localhost/orcl")
    cursor = connect.cursor()
    query = "ALTER TABLE DB_IMAGE \
             ADD CONSTRAINTS fk_img FOREIGN KEY(IMG_SEARCHITEM) \
             REFERENCES DB_IMAGE_SUBJECTS(IMG_SEARCHITEM) ON DELETE CASCADE"
    cursor.execute(query)

# ===================================================================================


def database_load():
    db_serchItem = input("Database에서 검색할 검색어를 입력하세요                : ")
    # 오라클 접속
    # connect("사용자 ID", "Password", "server")
    connect = cx_Oracle.connect("system", "oraclepractice", "localhost/orcl")
    cursor = connect.cursor()

    cursor.execute("select IMG_SEARCHITEM, IMG_WIDTH, IMG_HEIGHT, IMG_TITLE, IMG_PICTURE from DB_IMAGE where IMG_TEXT = 0 and IMG_SEARCHITEM = '{}'".format(db_serchItem))
    result = cursor.fetchall()

    datas = []
    for i in range(len(result)):
        datas.append((result[i][0], result[i][1], result[i][2], result[i][3], result[i][4].read()))

    re_datas = list(set(datas))         # 중복 제거

    # 이미지를 출력하기 위한 subplot 행렬 크기 정의
    cols = 10
    row = math.ceil(len(re_datas) / cols)

    # 이미지 출력
    fig = plt.figure(num="검색 : '{}' ".format(db_serchItem), figsize=(15, 10))


    for i in range(len(re_datas)):
        img = base64.decodebytes(re_datas[i][4])
        image = Image.open(BytesIO(img))
        subplot = fig.add_subplot(row, cols, i+1)
        subplot.imshow(image)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_xlabel("Num: {} ({}×{})".format(i, re_datas[i][1], re_datas[i][2]), fontsize=6)

    plt.tight_layout()
    plt.show()


def get_property_index(property):
    new_property = []
    for ind in image_index:
        new_property.append(property[ind])
    return new_property

if __name__ == "__main__":
    operation = input("Crawling = 1, Look up = 2 중 원하는 동작을 선택하세요                : ")
    Create_subject_table()

    if operation == '1':
        html_source, url, search_chr, len_images = crawling()
        img_txt_contain = img_iter(search_chr)
        folder, searchItem, image_width, image_height, image_title, image_source, image_index = get_final_img_property(html_source, url, search_chr)
        image_width = get_property_index(image_width)
        image_height = get_property_index(image_height)
        image_title = get_property_index(image_title)
        image_source = get_property_index(image_source)
        img_txt_contain = get_property_index(img_txt_contain)
        print(len(image_width), len(image_height), len(image_title), len(image_source), len(image_index))
        database_upload(folder, searchItem, image_width, image_height, image_title, image_source, img_txt_contain)

    if operation == '2':
        database_load()

