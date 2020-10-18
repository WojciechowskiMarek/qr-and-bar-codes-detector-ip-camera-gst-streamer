import numpy as np
import cv2
import jetson.utils
#import argparse
import gi
from pyzbar import pyzbar
gi.require_version('Gst', '1.0')

# parse the command line
#parser = argparse.ArgumentParser()
#parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
#parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
#parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")
#opt = parser.parse_args()
#print(opt)

# create display window
display = jetson.utils.glDisplay()

# create camera device
#uridecodebin uri=rtsp://192.168.0.12:8080/video/h264
#camera = jetson.utils.gstCamera(640, 480, "uridecodebin uri=rtsp://192.168.0.12:8080/video/h264, width=640, height=480 ")
camera = jetson.utils.gstCamera(1024, 768, "rtspsrc location=rtsp://192.168.0.12:8080/video/h264 latency=0 ! rtph264depay !  h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx, width=1024, height=768 ")
#camera = jetson.utils.gstCamera(640,480,"rtspsrc location=rtsp://192.168.0.12:8080/video/h264 latency=0 ! queue ! rtph264depay ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink")

# open the camera for streaming
camera.Open()

# capture frames until user exits
while display.IsOpen():
	#image, width, height = camera.CaptureRGBA()
    image, width, height = camera.CaptureRGBA (zeroCopy = True)
    jetson.utils.cudaDeviceSynchronize ()
        # create a numpy ndarray that references the CUDA memory
        # it won't be copied, but uses the same memory underneath
    aimg = jetson.utils.cudaToNumpy (image, width, height, 4)
        #print ("img shape {}".format (aimg1.shape))
    aimg1 = cv2.cvtColor (aimg.astype (np.uint8), cv2.COLOR_RGBA2BGR)
    barcodes = pyzbar.decode(aimg1)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(aimg1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(aimg1, text, (x, y - 20),
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.imshow('Bar and qr codes detector', aimg1)
	#display.RenderOnce(aimg1, width, height)
	#display.SetTitle("{:s} | {:d}x{:d} | {:.0f} FPS".format("Camera Viewer", width, height, display.GetFPS()))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
# close the camera
camera.Close()
cv2.destroyAllWindows()
