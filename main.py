import time
import picamera
import picamera.array
import cv2
import numpy as np
import io
from threading import Condition

# Global constants
output_width = 1280
output_height = 960


class StreamingOutput(object):
    def __init__(self, plate_cascade):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.plate_cascade = plate_cascade

    def write(self, buf):
        # Here the frame is written in the program
        # We must perform some kind of processing
        # To check if a plate is detected
        # Here we will reshape our buffer
        np_image = np.frombuffer(buf, dtype=np.uint8) \
                    .reshape(output_height, output_width, 3)
       

        # Before applying haar cascades
        # We will resize the image to improve performance
        # In this case, we preserve aspect ratio
        # To avoid distorting the image
        new_dim = (640, 480)
        np_resized = cv2.resize(np_image, new_dim, interpolation= cv2.INTER_NEAREST)

        # Now, we will  perform the multiscale operation
        start = time.time()
        
        results = self.plate_cascade.detectMultiScale(np_resized, 1.1)

        if len(results) > 0:
            print('Plate detected!')

        end = time.time()
        print('Elapsed: ', end - start)

        return self.buffer.write(buf)


def main():
    print('Initializing main function')

    # Initialize cascade classifier
    plate_cascade = cv2.CascadeClassifier()
    plate_cascade_name = '/root/us.xml'

    if not plate_cascade.load(plate_cascade_name):
        print('Error loading face cascade')
        exit(0) 

    print('Setting capture function')
    # Capture an opencv compatible array
    with picamera.PiCamera() as camera:
        # We must capture the frames as a recording
        # To avoid losing FPS
        camera.framerate = 5
        camera.resolution = (output_width, output_height)
        output = StreamingOutput(plate_cascade)
        camera.start_recording(output, format='bgr')

        input('Press enter to stop the program...')        
        # Waiting for enter input


if __name__ == '__main__':
    main()



