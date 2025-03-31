import socket
import threading
class SocketAudioInput:
    def __init__(self, host, port, chunk_size, stop_event, send_queue, recv_queue):
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.stop_event = stop_event
        self.send_queue = send_queue
        self.recv_queue = recv_queue 
        self.sock = None
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start the socket listener in a separate thread"""
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        """Main listener loop with feedback prevention"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Listening for audio input on {self.host}:{self.port}...")
        
        conn, addr = self.sock.accept()
        print(f"Accepted audio input connection from {addr}")
        
        try:
            while not self.stop_event.is_set():
                # Always receive the data to prevent socket buffer overflow
                data = b''
                while len(data) < self.chunk_size:
                    packet = conn.recv(self.chunk_size - len(data))
                    if not packet:
                        break
                    data += packet
                
                if data:
                    with self.lock:
                        # Only send if we're not currently playing audio
                        if self.recv_queue.empty():
                            self.send_queue.put(data)
        except ConnectionResetError:
            print("Client disconnected")
        finally:
            conn.close()
            self.sock.close()