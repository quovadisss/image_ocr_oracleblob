# Image database management using Oracle Blob and OCR
This repository implements creating image database management system using Oracle Blob and OCR. This project helps people to save image files on Oracle without watermarks or useless text in the image. To classify an image has text or not, Open CV2 were used. Ocracle Blob library was used to save image files on Oracle. This project was done for team project in the Database class, Seoultech.

## Requirements
* python 3.x
* cx_Oracle
* cv2
* imutils

## How to run
* run `python main.py`
* If you type '1', you can crawl images from google and save data to Oracle.
* If you type '2', you can load images from Oracle.