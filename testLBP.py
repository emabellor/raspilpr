import cv2
import numpy as np


def main():
    print('Initializing main function')
    image_name = '/root/plate.jpg'
    image = cv2.imread(image_name)
    
    if image is None:
        print('The image was not loaded')
        exit(0)

    # Initialize cascade classifier
    plate_cascade = cv2.CascadeClassifier()
    plate_cascade_name = '/root/us.xml'
  
    if not plate_cascade.load(plate_cascade_name):
        print('Error loading face cascade')
        exit(0)
  
    results = plate_cascade.detectMultiScale(image, 1.1)
    print(results)

    print('Done')



if __name__ == '__main__':
    main()
