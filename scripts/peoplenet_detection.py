#!/usr/bin/env python3
"""
PeopleNet Detection Pipeline for DeepStream on Jetson Orin Nano

Standalone script - no Docker, no external dependencies beyond system packages.
Uses pure DeepStream + TensorRT (no Triton) for minimal VRAM usage.
Need to store on-the-fly compiled .onnx to TensorRT engine file on persistent storage, namely /var/cache
Detects: person, bag, face

Usage:
    peoplenet-detection -i <video_file_or_uri>
    peoplenet-detection -i /opt/nvidia/deepstream/samples/sample.mp4
    peoplenet-detection -i file:///path/to/video.mp4
    peoplenet-detection -i rtsp://camera_ip/stream
    peoplenet-detection --tcp-port 5001          # Accept live JPEG stream from ESP32

Requirements (installed via Yocto):
    - DeepStream 7.1
    - TensorRT
    - GStreamer + NVIDIA plugins
    - Python3 + PyGObject + pyds
    
Sample Command:
    python3 scripts/peoplenet_detection.py   -i   sample_video/20250828132109/D01_20250828132109.mp4   sample_video/20250828132109/D02_20250828132109.mp4   sample_video/20250828132109/D03_20250828132109.mp4   sample_video/20250828132109/D04_20250828132109.mp4
"""

import sys
import os
import argparse
import math
import time
import socket
import struct
import subprocess
import threading
import re
from collections import defaultdict

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds #python deepstream

# =============================================================================
# Configuration
# =============================================================================

# Model paths (Yocto deployment locations)
# PEOPLENET_CONFIG = "/opt/nvidia/deepstream/models/peoplenet/nvinfer_config.txt"
# DEFAULT_SAMPLE = "/opt/nvidia/deepstream/samples/peoplenet/sample.mp4"

# # TensorRT engine build settings
# ENGINE_FILE = "/var/cache/deepstream/peoplenet/resnet34_peoplenet_fp16.engine"
# ONNX_FILE = "/opt/nvidia/deepstream/models/peoplenet/resnet34_peoplenet_int8.onnx"

# Model paths (docker locations)
MODEL_DIR = "/workspace/models/peoplenet_vpruned_quantized_decrypted_v2.3.4"
PEOPLENET_CONFIG = f"{MODEL_DIR}/nvinfer_config.txt"
TRACKER_CONFIG = f"{MODEL_DIR}/../tracker_config.txt"

# TensorRT engine build settings
ENGINE_FILE = f"{MODEL_DIR}/resnet34_peoplenet_int8.onnx_b1_gpu0_fp16.engine"
ONNX_FILE = f"{MODEL_DIR}/resnet34_peoplenet_int8.onnx"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# TCP video stream port (ESP32 sends JPEG frames here, separate from control :5000)
TCP_VIDEO_PORT = 5001

# Max JPEG frame size sanity check (2 MB)
TCP_MAX_FRAME_BYTES = 2 * 1024 * 1024

# Expanded global dictionary to track {object_id: first_seen_time}
object_timers = {}


# =============================================================================
# TensorRT Engine Build
# =============================================================================

def ensure_engine_exists():
    """Build TensorRT engine using trtexec if it doesn't exist"""
    import subprocess

    if os.path.exists(ENGINE_FILE):
        print(f"[INFO] TensorRT engine found: {ENGINE_FILE}")
        return True

    print("=" * 70)
    print("TensorRT Engine Build Required")
    print("=" * 70)

    if not os.path.exists(ONNX_FILE):
        sys.stderr.write(f"[ERROR] ONNX model not found: {ONNX_FILE}\n")
        return False

    if not os.path.exists(TRTEXEC):
        sys.stderr.write(f"[ERROR] trtexec not found: {TRTEXEC}\n")
        return False

    # Create cache directory if needed
    engine_dir = os.path.dirname(ENGINE_FILE)
    os.makedirs(engine_dir, mode=0o777, exist_ok=True)

    print(f"[INFO] Building TensorRT engine from ONNX model...")
    print(f"[INFO] ONNX: {ONNX_FILE}")
    print(f"[INFO] Engine: {ENGINE_FILE}")
    print("[INFO] This may take several minutes on first run...")
    print("-" * 70)

    cmd = [
        TRTEXEC,
        f"--onnx={ONNX_FILE}",
        f"--saveEngine={ENGINE_FILE}",
        "--fp16"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        if os.path.exists(ENGINE_FILE):
            os.chmod(ENGINE_FILE, 0o644)
            print("-" * 70)
            print(f"[INFO] Engine built successfully: {ENGINE_FILE}")
            return True
        else:
            sys.stderr.write("[ERROR] Engine file not created\n")
            return False
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[ERROR] trtexec failed with code {e.returncode}\n")
        return False
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to build engine: {e}\n")
        return False


# PeopleNet class labels
PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_BAG = 1
PGIE_CLASS_ID_FACE = 2
PEOPLENET_LABELS = ["person", "bag", "face"]

# Pipeline settings
# Use model's native input resolution - VIC in streammux handles downscaling
MUXER_OUTPUT_WIDTH = 960
MUXER_OUTPUT_HEIGHT = 544
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 960
TILED_OUTPUT_HEIGHT = 544


# =============================================================================
# Power Monitoring (tegrastats)
# =============================================================================

class PowerMonitor:
    """Monitor power consumption using tegrastats"""

    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self.running = False
        self.thread = None
        self.process = None
        self.power_readings = []  # List of (timestamp, watts)
        self.lock = threading.Lock()
        self.last_print_time = 0
        self.print_interval = 10.0  # Print every 10 seconds

    def start(self):
        """Start tegrastats monitoring in background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("[POWER] Started tegrastats power monitoring")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        """Background thread that runs tegrastats and parses output"""
        try:
            # Run tegrastats with specified interval
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )

            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break

                power_mw = self._parse_power(line)
                if power_mw is not None:
                    with self.lock:
                        self.power_readings.append((time.time(), power_mw / 1000.0))
                        # Keep only last 60 seconds of readings
                        cutoff = time.time() - 60
                        self.power_readings = [(t, p) for t, p in self.power_readings if t > cutoff]

        except FileNotFoundError:
            print("[POWER] tegrastats not found - power monitoring disabled")
        except Exception as e:
            print(f"[POWER] Error: {e}")

    def _parse_power(self, line):
        """Parse VDD_IN power from tegrastats output (in mW)"""
        # Format: VDD_IN 4568mW/4568mW (current/average)
        match = re.search(r'VDD_IN\s+(\d+)mW', line)
        if match:
            return int(match.group(1))
        return None

    def get_stats(self):
        """Get power statistics for the print interval"""
        with self.lock:
            if not self.power_readings:
                return None

            cutoff = time.time() - self.print_interval
            recent = [p for t, p in self.power_readings if t > cutoff]

            if not recent:
                return None

            return {
                'current': recent[-1],
                'avg': sum(recent) / len(recent),
                'min': min(recent),
                'max': max(recent),
            }

    def print_stats(self):
        """Print power stats (called from GLib timeout)"""
        now = time.time()
        if now - self.last_print_time < self.print_interval:
            return True

        self.last_print_time = now
        stats = self.get_stats()
        if stats:
            print(f"[POWER] Current: {stats['current']:.1f}W | "
                  f"Avg: {stats['avg']:.1f}W | "
                  f"Min: {stats['min']:.1f}W | "
                  f"Max: {stats['max']:.1f}W")
        return True  # Continue timeout


# =============================================================================
# FPS Tracking (inline, no external dependency)
# =============================================================================

class FPSTracker:
    """Simple FPS tracker per stream"""

    def __init__(self, num_streams):
        self.start_time = defaultdict(lambda: time.time())
        self.frame_count = defaultdict(int)
        self.fps = defaultdict(float)

    def update(self, stream_id):
        self.frame_count[stream_id] += 1
        elapsed = time.time() - self.start_time[stream_id]
        if elapsed >= 5.0:  # Update FPS every 5 seconds
            self.fps[stream_id] = self.frame_count[stream_id] / elapsed #basically how many frames passed over the last 5 seconds
            self.frame_count[stream_id] = 0
            self.start_time[stream_id] = time.time()

    def get_fps(self, stream_id):
        return self.fps.get(stream_id, 0.0)

    def print_fps(self):
        if self.fps:
            fps_str = ", ".join([f"stream{k}: {v:.1f}" for k, v in sorted(self.fps.items())])
            print(f"[FPS] {fps_str}")
        return True  # Continue timeout


# =============================================================================
# TCP JPEG Receiver (ESP32 → appsrc)
# =============================================================================

class TcpJpegReceiver:
    """
    Listens on a TCP port for the ESP32 JPEG stream.

    ESP32 frame format (from tcp_stream_task in watchtower.c):
        [4-byte uint32 LE size][JPEG data of that size]

    Each received frame is pushed into a GStreamer appsrc element and
    an ACK log line is printed so progress is visible in the journal.
    """

    def __init__(self, port, appsrc):
        self.port = port
        self.appsrc = appsrc
        self.running = False
        self.thread = None
        self.frame_count = 0
        # Nanosecond timestamp step for 12 FPS (matches ESP32 cap in tcp_stream_task)
        self._frame_duration_ns = Gst.SECOND // 12

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        print(f"[TCP] Waiting for ESP32 JPEG stream on port {self.port}...")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def _server_loop(self):
        """Accept one ESP32 connection at a time; restart on disconnect."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(1)
        server.settimeout(1.0)

        while self.running:
            try:
                conn, addr = server.accept()
                print(f"[TCP] ESP32 connected from {addr[0]}:{addr[1]}")
                self._recv_loop(conn, addr)
                print(f"[TCP] ESP32 disconnected from {addr[0]}:{addr[1]}")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[TCP] Server error: {e}")

        server.close()

    def _recv_loop(self, conn, addr):
        """Read [4-byte size][JPEG data] frames and push each to appsrc."""
        conn.settimeout(5.0)
        try:
            while self.running:
                # Read 4-byte little-endian size header
                header = self._recv_exact(conn, 4)
                if not header:
                    break

                size = struct.unpack('<I', header)[0]
                if size == 0 or size > TCP_MAX_FRAME_BYTES:
                    print(f"[TCP] Bad frame size {size} from {addr[0]}, dropping connection")
                    break

                jpeg_data = self._recv_exact(conn, size)
                if not jpeg_data:
                    break

                self.frame_count += 1

                # ACK log: one line per received frame
                print(f"[TCP] ACK frame #{self.frame_count:05d} | "
                      f"{size:6d} bytes | from {addr[0]}")

                # Wrap in GstBuffer and push to appsrc
                buf = Gst.Buffer.new_wrapped(jpeg_data)
                pts = self.frame_count * self._frame_duration_ns
                buf.pts = pts
                buf.dts = pts
                buf.duration = self._frame_duration_ns

                ret = self.appsrc.emit('push-buffer', buf)
                if ret != Gst.FlowReturn.OK:
                    print(f"[TCP] appsrc push returned {ret}, stopping")
                    break

        except socket.timeout:
            print(f"[TCP] Timeout waiting for frame from {addr[0]}")
        except Exception as e:
            if self.running:
                print(f"[TCP] Recv error from {addr[0]}: {e}")
        finally:
            conn.close()

    def _recv_exact(self, conn, n):
        """Read exactly n bytes from socket, return None on EOF."""
        data = b''
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data


# =============================================================================
# Global State
# =============================================================================

fps_tracker = None
power_monitor = None
tcp_receiver = None
no_display = False
silent = False
file_loop = False


# =============================================================================
# Probe Callback
# =============================================================================

def format_timestamp(pts_ns):
    """Convert PTS (nanoseconds) to HH:MM:SS.mmm format"""
    if pts_ns == Gst.CLOCK_TIME_NONE:
        return "--:--:--.---"
    total_ms = pts_ns // 1_000_000
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    min_ = total_min % 60
    hrs = total_min // 60
    return f"{hrs:02d}:{min_:02d}:{sec:02d}.{ms:03d}"


def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Stateful probe: Tracks unique object_ids for faces and people.
    Prints an alarm only if an object is detected for > 1.0 second.
    """
    global silent, object_timers
        
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    current_wall_time = time.time()
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        l_obj = frame_meta.obj_meta_list
        
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Process both Persons and Faces
            if obj_meta.class_id in [PGIE_CLASS_ID_PERSON, PGIE_CLASS_ID_FACE]:
                obj_id = obj_meta.object_id
                label = PEOPLENET_LABELS[obj_meta.class_id]
                
                # Initialize timer for new unique IDs
                if obj_id not in object_timers:
                    object_timers[obj_id] = current_wall_time
                else:
                    duration = current_wall_time - object_timers[obj_id]
                    
                    # Alarm logic for detections > 1 second
                    if duration >= 1.0 and not silent:
                        rect = obj_meta.rect_params
                        # Dynamic print statement for "Found face #x" or "Found person #x"
                        print(f"[ALARM] Found {label} #{obj_id} | Duration: {duration:.2f}s | "
                              f"BBox: ({int(rect.left)}, {int(rect.top)}, {int(rect.width)}, {int(rect.height)})")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Cleanup stale IDs (not seen for > 5s) to manage memory on Jetson hardware
        for oid in list(object_timers.keys()):
            if current_wall_time - object_timers[oid] > 5.0:
                del object_timers[oid]

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# =============================================================================
# Pipeline Helpers
# =============================================================================

def bus_call(bus, message, loop): ##this is subject to function callback, where it gets constantly called
    """Handle GStreamer bus messages"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n[INFO] End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\n[ERROR] {err}")
        if debug:
            print(f"[DEBUG] {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        if not silent:
            print(f"[WARNING] {err}")
    return True


def cb_newpad(decodebin, decoder_src_pad, data):
    """Callback when decodebin creates a new pad"""
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()

    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """Callback when decodebin adds a child element"""
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element and source_element.find_property('drop-on-latency'):
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    """Create a source bin for the given URI"""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(f"Unable to create source bin {bin_name}\n")
        return None

    # Use nvurisrcbin for file-loop support, otherwise uridecodebin
    global file_loop
    if file_loop:
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        if uri_decode_bin:
            uri_decode_bin.set_property("file-loop", 1)
            uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")

    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
        return None

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None

    return nbin


def is_aarch64():
    """Check if running on ARM64 (Jetson)"""
    import platform
    return platform.machine() == 'aarch64'


# =============================================================================
# Main Pipeline
# =============================================================================

def main(input_sources, config_file=None, tcp_port=None):
    global fps_tracker, power_monitor, tcp_receiver, no_display, file_loop

    # Ensure TensorRT engine is built before starting pipeline
    if not ensure_engine_exists():
        sys.stderr.write("[ERROR] Cannot proceed without TensorRT engine\n")
        return 1

    tcp_mode = tcp_port is not None
    number_sources = 1 if tcp_mode else len(input_sources)
    fps_tracker = FPSTracker(number_sources)

    # Start power monitoring (tegrastats)
    power_monitor = PowerMonitor(interval_ms=1000)
    power_monitor.start()

    # Initialize GStreamer
    Gst.init(None)

    print("=" * 70)
    print("PeopleNet Detection Pipeline - DeepStream + TensorRT")
    print("=" * 70)
    if tcp_mode:
        print(f"Input: TCP JPEG stream on port {tcp_port} (ESP32)")
    else:
        print(f"Input sources: {number_sources}")
        for i, src in enumerate(input_sources):
            print(f"  [{i}] {src}")
    print(f"Config: {config_file or PEOPLENET_CONFIG}")
    print(f"Display: {'disabled' if no_display else 'enabled'}")
    if not tcp_mode:
        print(f"File loop: {'enabled' if file_loop else 'disabled'}")
    print("-" * 70)

    # Create pipeline
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return 1

    # Create streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")
        return 1
    pipeline.add(streammux)

    # -------------------------------------------------------------------------
    # Source: TCP JPEG stream from ESP32
    # -------------------------------------------------------------------------
    if tcp_mode:
        # appsrc feeds raw JPEG buffers from the TCP receiver thread
        appsrc = Gst.ElementFactory.make("appsrc", "tcp-jpeg-src")
        if not appsrc:
            sys.stderr.write("Unable to create appsrc\n")
            return 1
        appsrc.set_property("caps", Gst.Caps.from_string("image/jpeg, framerate=12/1"))
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("block", True)   # block if downstream is full

        # jpegparse: marks frame boundaries so the decoder gets clean frames
        jpegparse = Gst.ElementFactory.make("jpegparse", "jpeg-parser")
        if not jpegparse:
            sys.stderr.write("Unable to create jpegparse\n")
            return 1

        # HW JPEG decoder on Jetson outputs NVMM NV12 directly
        if is_aarch64():
            jpeg_dec = Gst.ElementFactory.make("nvjpegdec", "jpeg-decoder")
            if not jpeg_dec:
                print("[WARN] nvjpegdec unavailable, falling back to CPU jpegdec")
                jpeg_dec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
        else:
            jpeg_dec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
        if not jpeg_dec:
            sys.stderr.write("Unable to create JPEG decoder\n")
            return 1

        # nvvideoconvert: normalises to NVMM NV12 expected by nvstreammux
        nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "src-nvvidconv")
        if not nvvidconv_src:
            sys.stderr.write("Unable to create nvvideoconvert for TCP source\n")
            return 1

        for elem in [appsrc, jpegparse, jpeg_dec, nvvidconv_src]:
            pipeline.add(elem)

        # Link: appsrc → jpegparse → jpeg_dec → nvvidconv_src → streammux
        appsrc.link(jpegparse)
        jpegparse.link(jpeg_dec)
        jpeg_dec.link(nvvidconv_src)

        sinkpad = streammux.request_pad_simple("sink_0")
        if not sinkpad:
            sys.stderr.write("Unable to get streammux sink_0\n")
            return 1
        srcpad = nvvidconv_src.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to get nvvidconv src pad\n")
            return 1
        srcpad.link(sinkpad)

        is_live = True

        # Create the TCP receiver — it will push buffers into appsrc
        tcp_receiver = TcpJpegReceiver(tcp_port, appsrc)

    # -------------------------------------------------------------------------
    # Source: URI-based (file / RTSP) — existing behaviour
    # -------------------------------------------------------------------------
    else:
        is_live = False
        for i in range(number_sources):
            uri_name = input_sources[i]

            if uri_name.startswith("rtsp://"):
                is_live = True

            if not uri_name.startswith(("file://", "rtsp://", "http://", "https://")):
                uri_name = f"file://{os.path.abspath(uri_name)}"

            source_bin = create_source_bin(i, uri_name)
            if not source_bin:
                sys.stderr.write(f"Unable to create source bin for {uri_name}\n")
                return 1

            pipeline.add(source_bin)

            padname = f"sink_{i}"
            sinkpad = streammux.request_pad_simple(padname)
            if not sinkpad:
                sys.stderr.write(f"Unable to create sink pad {padname}\n")
                return 1

            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad\n")
                return 1

            srcpad.link(sinkpad)

    # Create queues for buffering (one for each pipeline stage)
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")

    # Create primary inference engine (nvinfer - pure TensorRT, no Triton)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create nvinfer\n")
        return 1
    
    ################ \/ Object Tracker Init
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    queue_tracker = Gst.ElementFactory.make("queue", "queue_tracker")
    
    if not tracker or not queue_tracker:
        sys.stderr.write("Unable to create tracker or tracker queue\n")
        return 1

    # Configure tracker using your file
    tracker.set_property('ll-config-file', TRACKER_CONFIG)
    # Using NvDCF is recommended for DeepStream 7.1
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    ################ /\

    # Create tiler for multi-stream display
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler\n")
        return 1

    # Create video converter
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")
        return 1

    # Create OSD (on-screen display)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")
        return 1
    nvosd.set_property('process-mode', 0)
    nvosd.set_property('display-text', 1)

    # Create sink
    if no_display:
        print("[INFO] Using fakesink (no display)")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if sink:
            sink.set_property('enable-last-sample', 0)
            sink.set_property('sync', 0)
    else:
        if is_aarch64():
            print("[INFO] Using nv3dsink (Jetson display)")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("[INFO] Using EGL sink (x86 display)")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not sink:
        sys.stderr.write("Unable to create sink element\n")
        return 1

    # Configure streammux
    if is_live or tcp_mode:
        print("[INFO] Live source detected")
        streammux.set_property('live-source', 1)

    if file_loop and is_aarch64() and not tcp_mode:
        streammux.set_property('nvbuf-memory-type', 4)
    #scale down video before inferencing using VIC to increase efficiency even more.
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    # Configure pgie
    pgie.set_property('config-file-path', config_file or PEOPLENET_CONFIG)
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(f"[INFO] Overriding infer-config batch-size {pgie_batch_size} with {number_sources}")
        pgie.set_property("batch-size", number_sources)

    # Configure tiler //this is for the OSD, so at the end, the website on the smartphone will show 4x composited look from each of the 4 cameras
    tiler_rows = int(math.sqrt(number_sources)) #sqroot resolution from each source
    tiler_columns = int(math.ceil(number_sources / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    if is_aarch64():
        tiler.set_property("compute-hw", 2)
    else:
        tiler.set_property("compute-hw", 1)

    sink.set_property("qos", 0)

    # Add elements to pipeline
    for elem in [queue1, pgie, queue_tracker, tracker, queue2, tiler, queue3, nvvidconv, queue4, nvosd, queue5, sink]:
        pipeline.add(elem)

    # Link elements
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue_tracker)
    queue_tracker.link(tracker)
    tracker.link(queue2)
    queue2.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(sink)

    # Add probe to pgie src pad (pad after the nvstreammux sink)
    tracker_src_pad = tracker.get_static_pad("src")
    if tracker_src_pad:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
    else:
        sys.stderr.write("Unable to get tracker src pad\n")

    # Create event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add FPS print callback every 5 seconds
    GLib.timeout_add(5000, fps_tracker.print_fps)

    # Add power stats print callback every 10 seconds
    GLib.timeout_add(10000, power_monitor.print_stats)

    print("\n[INFO] Starting pipeline...")
    print("[INFO] Press Ctrl+C to stop\n")

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Start TCP receiver after pipeline is playing so appsrc is ready
    if tcp_mode:
        tcp_receiver.start()

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

    if tcp_mode and tcp_receiver:
        tcp_receiver.stop()

    # Stop power monitoring
    if power_monitor:
        power_monitor.stop()

    print("[INFO] Pipeline stopped")
    print("=" * 70)

    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="peoplenet-detection",
        description="PeopleNet detection with DeepStream + TensorRT (no Docker, no Triton)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  peoplenet-detection -i /path/to/video.mp4
  peoplenet-detection -i video1.mp4 video2.mp4  # Multi-stream
  peoplenet-detection -i rtsp://camera/stream   # RTSP source
  peoplenet-detection -i video.mp4 --no-display --file-loop
  peoplenet-detection --tcp-port 5001           # Live JPEG stream from ESP32
        """
    )

    parser.add_argument(
        "-i", "--input",
        nargs="+",
        metavar="URI",
        help="Input video file(s) or RTSP URI(s)"
    )

    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"Accept live JPEG stream from ESP32 on this TCP port (default: {TCP_VIDEO_PORT}). "
             f"Mutually exclusive with -i."
    )

    parser.add_argument(
        "-c", "--config",
        default=None,
        metavar="CONFIG",
        help=f"nvinfer config file (default: {PEOPLENET_CONFIG})"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Disable video display (headless mode)"
    )

    parser.add_argument(
        "--file-loop",
        action="store_true",
        default=False,
        help="Loop input file(s) continuously"
    )

    parser.add_argument(
        "-s", "--silent",
        action="store_true",
        default=False,
        help="Suppress per-frame output"
    )

    args = parser.parse_args()

    if args.tcp_port is None and not args.input:
        parser.error("Either -i/--input or --tcp-port is required")
    if args.tcp_port is not None and args.input:
        parser.error("--tcp-port and -i/--input are mutually exclusive")

    global no_display, silent, file_loop
    no_display = args.no_display
    silent = args.silent
    file_loop = args.file_loop

    return args.input or [], args.config, args.tcp_port


if __name__ == '__main__':
    input_sources, config, tcp_port = parse_args()
    sys.exit(main(input_sources, config, tcp_port))