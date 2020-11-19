import datetime
import time
import picamera
import picamera.array
import cv2
import numpy as np
import io
from gpiozero import LED
from threading import Condition
from http import server
import socketserver
import math
import json
from openalpr import Alpr
import sqlite3
import logging


# Global constants
output_width = 1024
output_height = 768

IR_led1 = LED(20)
IR_led2 = LED(21)
np_image = None

# This will tell us where the margin of the plate should be put 
# To improve the detection
plate_margin = 100

def crop_image(np_image, x, y, width, height):
    cropped = np_image[y:y+height, x:x+width]
    return cropped


class StreamingOutput(object):
    def __init__(self, plate_cascade, alpr_instance):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.plate_cascade = plate_cascade
        self.alpr_instance = alpr_instance

    def write(self, buf):
        global np_image
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
        scale_factor = output_width / new_dim[0]  # Assume we have the same scale factor
        np_resized = cv2.resize(np_image, new_dim, interpolation= cv2.INTER_NEAREST)

        # Now, we will  perform the multiscale operation
        start = time.time()
        
        results = self.plate_cascade.detectMultiScale(np_resized, 1.1)

        if len(results) > 0:
            print('Plate detected!')

            print(results)
            # Now that we have the results - we need to crop the result 
            
            for result in results:
                x1 = result[0]
                y1 = result[1]
                x2 = result[0] + result[2]
                y2 = result[1] + result[3]
               
                # Here - We must add some margin to improve the alpr results
                x1 -= plate_margin
                if x1 < 0:
                    x1 = 0
                
                y1 -= plate_margin
                if y1 < 0:
                    y1 = 0

                x2 += plate_margin
                if x2 > new_dim[0]:
                    x2 = new_dim[0]

                y2 += plate_margin
                if y2 > new_dim[1]:
                    y2 = new_dim[1]

                x = math.floor(x1 * scale_factor)
                y = math.floor(y1 * scale_factor)
                width = math.floor((x2 - x1) * scale_factor)
                height = math.floor((y2 - y1) * scale_factor)

                print('Scale factor: ', scale_factor)
                print(x, y, width, height)
                
                # Now, we will generate the detection using openalpr
                cropped = crop_image(np_image, x, y, width, height)
              
                # Generate the detection using the cropped image
                output = self.alpr_instance.recognize_ndarray(cropped)
                
                for result in output['results']:
                    print('Candidate Detected') 
                    print('Plate', result['plate'])
                    print('Confidence', result['confidence'])
                    print('Hour: ', datetime.datetime.now())
   
        end = time.time()
        # print('Elapsed: ', end - start)

        return self.buffer.write(buf)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/test':
            content = bytes('Hello world', 'utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/snapshot':
            if np_image is None:
                print('Numpy image is none - Sending 500 error')
                self.send_error(500)
                self.end_headers()
            else:
                _, jpeg_np = cv2.imencode('.JPEG', np_image)
                jpeg_frame = jpeg_np.tobytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(jpeg_frame))
                self.end_headers()
                print(len(jpeg_frame))
                self.wfile.write(jpeg_frame)
        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        # Processing post request
        if self.path == '/start':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'OK',
                'session': 1234
            }

            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif self.path == '/stop':
            # TODO
            # Generate the logic to handle the JSONs
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'OK',
                'session': 1234
            }

            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404)
            self.end_headers()
  

def main():
    print('Initializing main function')

    print('Configuring logger')
    logging.basicConfig(filename='/root/plates.log', level=logging.DEBUG)
    logging.debug('Testing Log')

    print('Turning on the led sensor')
    IR_led1.on()
    IR_led2.on()

    # Initialize cascade classifier
    plate_cascade = cv2.CascadeClassifier()
    plate_cascade_name = '/root/us.xml'

    if not plate_cascade.load(plate_cascade_name):
        print('Error loading face cascade')
        exit(0) 

    # Initializing alpr instance
    alpr_instance = Alpr("us", "/usr/local/share/openalpr/config/openalpr.defaults.conf", \
                        "/usr/local/share/openalpr/runtime_data")

    if not alpr_instance.is_loaded():
        print('Error loading alpr')
        sys.exit(1)

    # Connecting to database instance
    sql_conn = sqlite3.connect('/root/raspilpr/plates.db')
    print('Database was opened successfully')

    print('Creating database tables')
    query = 'create table if not exists Tb_Plates (Id int primary key, Plate text not null, integer real not null, Date text not null)'

    sql_conn.execute(query)
    print('Query executed')

    print('Setting capture function')
    # Capture an opencv compatible array
    with picamera.PiCamera() as camera:
        # We must capture the frames as a recording
        # To avoid losing FPS
        camera.framerate = 5
        camera.resolution = (output_width, output_height)
        output = StreamingOutput(plate_cascade, alpr_instance)

        # Retrieving exposure compensation
        camera.exposure_compensation = -10

        camera.start_recording(output, format='bgr')
    
        print('Starting http server')
        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()


if __name__ == '__main__':
    main()



