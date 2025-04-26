import serial
import struct
import time

class AGGripper:
    PORT = "/dev/ttyUSB0"

    def __init__(self, port: str = PORT, baudrate: int = 115200, slave_id: int = 1, timeout: float = 0.5):
        self.port = serial.Serial(port, baudrate=baudrate, bytesize=8, parity='N', stopbits=1, timeout=timeout)
        self.slave_id = slave_id

    def _crc16(self, data: bytes) -> bytes:
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return struct.pack('<H', crc)

    def _send_command(self, function_code: int, register_address: int, value: int) -> bytes:
        frame = struct.pack('>B B H H', self.slave_id, function_code, register_address, value)
        crc = self._crc16(frame)
        self.port.write(frame + crc)
        return self.port.read(8)

    def initialize(self, full: bool = False) -> bytes:
        value = 0xA5 if full else 0x01
        return self._send_command(0x06, 0x0100, value)

    def set_force(self, force_percent: int) -> bytes:
        assert 20 <= force_percent <= 100
        return self._send_command(0x06, 0x0101, force_percent)

    def set_position(self, position_percent: int) -> bytes:
        assert 0 <= position_percent <= 100
        position_per_mille = int(position_percent) * 10  # Convert percentage to per mille (0-1000)
        return self._send_command(0x06, 0x0103, position_per_mille)

    def move_and_wait(self, target_position: int, timeout: float = 5) -> bool:
        self.set_position(target_position)
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_gripper_state()
            if state in (1, 2, 3):
                return True
            time.sleep(0.1)
        raise TimeoutError("Gripper did not reach target in time")

    def save_parameters(self) -> bytes:
        return self._send_command(0x06, 0x0300, 0x01)

    def read_register(self, register_address: int) -> int:
        frame = struct.pack('>B B H H', self.slave_id, 0x03, register_address, 1)
        crc = self._crc16(frame)
        self.port.write(frame + crc)
        response = self.port.read(7)
        if len(response) == 7:
            return struct.unpack('>H', response[3:5])[0]
        else:
            raise IOError("No response or invalid response")

    def get_initialization_state(self) -> int:
        return self.read_register(0x0200)

    def get_gripper_state(self) -> int:
        return self.read_register(0x0201)

    def get_current_position(self) -> int:
        return self.read_register(0x0202) / 10

    def close(self):
        self.port.close()
