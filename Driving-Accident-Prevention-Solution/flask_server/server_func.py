from simple_websocket_server import WebSocketServer, WebSocket
import socket, struct, threading, pickle, numpy, cv2, base64

def init_my_webServer() :
    my_webSocket = WebSocketServer('172.31.38.121', 5757, CMyWebSocket)
    my_webServer_th = threading.Thread(target = my_webServer_forever, args = (my_webSocket, ))
    my_webServer_th.start()

def my_webServer_forever(my_webSocket) :
    my_webSocket.serve_forever()


class CMyWebSocket(WebSocket):   
    my_ai_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    flag = False
    flag2 = False
    
    def handle(self):
        data = pickle.dumps(self.data)
        
        # 메시지 길이 측정
        message_size = struct.pack("L", len(data))
        
        global my_ai_socket
        # 데이터 전송
        CMyWebSocket.my_ai_socket.sendall(message_size + data)
        if not CMyWebSocket.flag :
            CMyWebSocket.flag = True
            th = threading.Thread(target = self.recvAndsend)
            th.start()
  
    def connected(self):  
        if not CMyWebSocket.flag2 :
            CMyWebSocket.flag2 = True
            CMyWebSocket.my_ai_socket.connect(('3.35.64.10', 5656))
        print("webServer :", self.address, 'connected')
        

    def handle_close(self):
        print("webServer :", self.address, 'closed')
    
    def recvAndsend(self) :
        data = b'' # 수신한 데이터를 넣을 변수
        payload_size = struct.calcsize(">L")
        
        while True :
            # 프레임 수신
            while len(data) < payload_size:
                data += CMyWebSocket.my_ai_socket.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(data) < msg_size:
                data += CMyWebSocket.my_ai_socket.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            # Convert captured image to JPG
            retval, buffer = cv2.imencode('.jpg', frame)
            
            # Convert to base64 encoding 
            #jpg_as_text = base64.b64encode(buffer)
            
            encoded = base64.b64encode(buffer).decode('utf-8')
            
            self.send_message("data:image/{};base64,{}".format('jpg', encoded))
            

                
   
