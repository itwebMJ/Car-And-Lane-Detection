import pickle, socket, struct
import matplotlib.pyplot as plt
import cv2, base64, warnings
import numpy as np
import threading, math, os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML





class CSock :
    HOST = 'HOST Address'
    PORT = 5656
    def __init__ (self) :
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print('소켓 생성')
        self.conn = None
        self.client_addr = None
        
        
    def listen(self) :
        self.socket.bind((CSock.HOST, CSock.PORT))
        self.socket.listen(10)

        self.conn, self.client_addr = self.socket.accept()
        print("ai connected")
        
        th1 = threading.Thread(target = self.image_decoding)
        th2 = threading.Thread(target = self.image_encoding1)
        th1.start()
        th2.start()
    
        
    def image_decoding(self) :
        data = b''
        payload_size = struct.calcsize("L")
        
        
        while True :
        
            while len(data) < payload_size:
                data += self.conn.recv(4096)
          
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            # 메시지 사이즈 기준으로 데이터 구성
            while len(data) < msg_size:
                data += self.conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 프레임 로드
            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(np.fromstring(base64.b64decode(frame.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

            frame = cv2.flip(frame, 1)
            
            ### 비디오 재생할땐 밑에 주석 처리
            #self.image_encoding2(frame)
        
            
            
    def image_encoding1(self) :
        cap = cv2.VideoCapture("test2.mp4")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] 
        
        # 직렬화(serialization) : 효율적으로 저장하거나 스트림으로 전송할 때 객체의 데이터를 줄로 세워 저장하는 것
        # binary file : 컴퓨터 저장과 처리 목적을 위해 이진 형식으로 인코딩된 데이터를 포함
        while True :
            
            ret, frame = cap.read()
            if not ret :
                break
                
            frame = image_process(frame)
            result, frame = cv2.imencode('.jpg', frame, encode_param) # 프레임 인코딩
            pickle_data = pickle.dumps(frame, 0) # 프레임을 직렬화화하여 binary file로 변환
            size = len(pickle_data)
            self.conn.sendall(struct.pack(">L", size) + pickle_data)
        self.socket.close()


    def image_encoding2(self, frame) :
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] 
        
        # 직렬화(serialization) : 효율적으로 저장하거나 스트림으로 전송할 때 객체의 데이터를 줄로 세워 저장하는 것
        # binary file : 컴퓨터 저장과 처리 목적을 위해 이진 형식으로 인코딩된 데이터를 포함
        frame = image_process(frame)    
        result, frame = cv2.imencode('.jpg', frame, encode_param) # 프레임 인코딩
        pickle_data = pickle.dumps(frame, 0) # 프레임을 직렬화화하여 binary file로 변환
        size = len(pickle_data)
        self.conn.sendall(struct.pack(">L", size) + pickle_data)
            

    def image_resend(self, data) :
        self.conn.sendall(data)
        
    def close(self) :
        self.socket.close()
        
    
        
        
      
    
    
    
    



def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.array([0,0,0])
    upper = np.array([255,255,255])
    yellower = np.array([0,0,0])
    yelupper = np.array([80,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])
    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)
    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img):
    return cv2.Canny(grayscale(img), 50, 120)


def draw_lines(img, lines, thickness=5):
    rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
    rightColor=[255,0,0]   
    leftColor=[255,0,0]

    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y1-y2)/(x1-x2)
                if slope > 0.3:
                    if x1 > 500 :
                        yintercept = y2 - (slope*x2)
                        rightSlope.append(slope)
                        rightIntercept.append(yintercept)
                    else: None
                elif slope < -0.3:
                    if x1 < 600:
                        yintercept = y2 - (slope*x2)
                        leftSlope.append(slope)
                        leftIntercept.append(yintercept)
        #We use slicing operators and np.mean() to find the averages of the 30 previous frames
        #This makes the lines more stable, and less likely to shift rapidly
        leftavgSlope = np.mean(leftSlope[-30:])
        leftavgIntercept = np.mean(leftIntercept[-30:])
        rightavgSlope = np.mean(rightSlope[-30:])
        rightavgIntercept = np.mean(rightIntercept[-30:])
        #Here we plot the lines and the shape of the lane using the average slope and intercepts
        try:
            left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
            left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
            right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
            right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
            pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2+250, int(img.shape[0])],
                            [right_line_x2-250, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img,[pts],(0,255,0))
            cv2.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2+250, int(img.shape[0])), leftColor, 10)
            cv2.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2-250, int(img.shape[0])), rightColor, 10)
            
            
            mid_y, mid_x = int(img.shape[1]/2), int(img.shape[0])
            
            ########## 차선 변경시 화면 빨갛게 표시 ##########
            if (mid_y <= left_line_x2+250) or (mid_y >= right_line_x2-250):
                #if i % 2 == 0: # 프레임 2배수일 때 표시
                img[:] = (0,0,200) #BGR 
           
        except ValueError:
        #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
            pass
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def linedetect(img):
    return hough_lines(img, 1, np.pi/180, 10, 20, 100)

def weightSum(input_set):
    return cv2.addWeighted(image, 1, input_set, 0.8, 0)



def processImage(image):
    interest = roi(image)
    filterimg = color_filter(interest)
    canny = cv2.Canny(grayscale(filterimg), 50, 120)
    myline = hough_lines(canny, 1, np.pi/180, 10, 20, 5)
    weighted_img = cv2.addWeighted(myline, 1, image, 0.8, 0)
    return weighted_img

def processImageWithDetect(image):
    interest = roi(image)
    filterimg = color_filter(interest)
    canny = cv2.Canny(grayscale(filterimg), 50, 120)
    myline = hough_lines(canny, 1, np.pi/180, 10, 20, 5)
    weighted_img = cv2.addWeighted(myline, 1, image, 0.8, 0)
    return weighted_img



def model_init() :
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    classLabels = []
    file_name = 'Labels.txt'
    with open(file_name, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
        
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)   # 255/2 = 127.5
    model.setInputMean((127.5, 127.5, 127.5))   # mobilenet => [-1, 1]
    model.setInputSwapRB(True)
    return model, classLabels

tracking = []

def image_process(frame) :
    model, classLabels = model_init()
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
   
#     lane_img = np.copy(frame)
#     ######### car detect ######## 
#     ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

#     frame_y, frame_x, _ = frame.shape  #frame size

#     if len(ClassIndex) != 0:
#         for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
#             x, y, w, h= boxes

#             if (w > 60 ) & (h >60) :
#                 if x == frame_y/2 : # 객체가 정 가운데 있을때
#                     target = round((frame_y-y + h /2 ) * 1.8 / w, 2)
#                 else:
#                     target = round(math.sqrt(((frame_y-y)+ h /2 ) **2 +((x+w/2)-frame_x/2)**2 ) * 1.8 / w, 1)

#                 #               car              bus              truck
#                 if ClassInd == 3 or ClassInd == 6 or ClassInd == 8:
#                     cv2.rectangle(frame, boxes, (30, 100, 255), 5)
#                     cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]-20), font, fontScale=font_scale, 
#                                 color=(255, 255, 0), thickness=3)

#                     # 15m 보다 가깝고, 관심영역(region of interest) 안에 있을 때 m 표시
#                     if target <= 15 :
#                         tracking.append(target) # m 변화량 감지를 위해 추가

#                         if (x >= frame_x * 0.35) & (x + w <= frame_x * 0.65) & (y >= frame_y * 0.35) & (y+ h<= frame_y * 0.65):
#                             cv2.putText(frame, str(target)+'m', (boxes[0]+120, boxes[1]+40), font, fontScale=font_scale, 
#                                         color=(255, 255, 0), thickness=3)

#                             if len(tracking) >= 10: # 15m 보다 가깝고, 프레임수 10개 이상일 때 
#                                 # 15m 보다 가깝고, 관심영역 안에 있고, 변화량이 5m보다 크면 화면 빨갛게 표시 
#                                 # if 5 <= abs(sum(tracking[-5:]) - sum(tracking[-6:-1])) :
#                                 if 5 <= (tracking[-5] - tracking[-1]) : #앞차랑 가까워질 때
#                                     print('Object is too close!',(tracking[-5],  tracking[-1]))

#                                     if i % 2 == 0: # 프레임 2배수일 때 표시
#                                         frame[:] = (0,0,200)
#                                         # red_img = np.copy(frame)
#                                         # red_img[:] = (0,0,200) #BGR 
#                                         #투명도 때문에 추가
#                                         # cv2.addWeighted(frame, 1, red_img, 0.8, 0) 

#                     else: 
#                         tracking.clear()
    
    lane_img = np.copy(frame)
    ######### car detect ######## 
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

    frame_x, frame_y, _ = frame.shape  #frame size
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            x, y, w, h= boxes

            if (w > 60 ) & (h >60) :
                if x == frame_y/2 : # 객체가 정 가운데 있을때
                    target = round((frame_y-y + h /2 ) * 1.8 / w, 2)
                else:
                    target = round(math.sqrt(((frame_y-y)+ h /2 ) **2 +((x+w/2)-frame_x/2)**2 ) * 1.8 / w, 1)

                #               car              bus              truck
                if ClassInd == 3 or ClassInd == 6 or ClassInd == 8:
                    cv2.rectangle(frame, boxes, (30, 100, 255), 5)
                    cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]-20), font, fontScale=font_scale, color=(255, 255, 0), 
                                thickness=3)
                    cv2.putText(frame, str(target)+'m', (boxes[0]+120, boxes[1]-20), font, fontScale=font_scale, color=(255, 255, 0),
                                thickness=3)
    
    interest = roi(lane_img)
    filterimg = color_filter(interest)
    canny = cv2.Canny(grayscale(filterimg), 50, 120)
    myline = hough_lines(canny, 1, np.pi/180, 10, 20, 0)
    weighted_img = cv2.addWeighted(myline, 1, frame, 0.8, 0)
    return weighted_img
    
    
    #threshold, min_line_len, max_line_gap


