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


# Global constants
output_width = 1280
output_height = 960

IR_led = LED(20)
np_image = None


class StreamingOutput(object):
    def __init__(self, plate_cascade):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.plate_cascade = plate_cascade

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
        np_resized = cv2.resize(np_image, new_dim, interpolation= cv2.INTER_NEAREST)

        # Now, we will  perform the multiscale operation
        start = time.time()
        
        results = self.plate_cascade.detectMultiScale(np_resized, 1.1)

        if len(results) > 0:
            print('Plate detected!')

        end = time.time()
        print('Elapsed: ', end - start)

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

def main():
    print('Initializing main function')

    print('Turning on the led sensor')
    # IR_led.on()

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
    
        print('Starting http server')
        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()


if __name__ == '__main__':
    main()



