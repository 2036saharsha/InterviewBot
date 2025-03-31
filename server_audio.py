import socket
import sounddevice as sd
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class SenderArguments:
    host: str = field(
        default="localhost",
        metadata={"help": "Listener's hostname/IP. Default: localhost"}
    )
    input_port: int = field(
        default=12347,
        metadata={"help": "Listener's input port. Default: 12347"}
    )
    send_rate: int = field(
        default=16000,
        metadata={"help": "Sample rate (Hz). Default: 16000"}
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "Chunk size in bytes. Default: 1024"}
    )

def send_audio():
    parser = HfArgumentParser(SenderArguments)
    args = parser.parse_args_into_dataclasses()[0]  

    # Create TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.input_port))
    
    dtype = "int16"
    channels = 1
    blocksize = args.chunk_size // 2  # 16-bit = 2 bytes per sample

    def callback(indata, frames, time, status):
        """Directly send raw audio chunks over socket"""
        try:
            sock.sendall(bytes(indata))
        except Exception as e:
            print(f"Send error: {e}")
            raise sd.CallbackAbort

    try:
        with sd.RawInputStream(
            samplerate=args.send_rate,
            blocksize=blocksize,
            dtype=dtype,
            channels=channels,
            callback=callback
        ) as stream:
            print(f"Streaming to {args.host}:{args.input_port}")
            print("Start speaking... (Ctrl+C to stop)")
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        sock.close()
        print("Sender closed")

if __name__ == "__main__":
    send_audio()