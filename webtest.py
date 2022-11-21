from streamlit_webrtc import webrtc_streamer
from MoveNet_Processing_Utils import movenet_processing
import av

def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        out_image = img.copy()
        
        out_image, _ = movenet_processing(out_image, draw_movenet_skeleton=True, max_people=1)
        
        return av.VideoFrame.from_ndarray(out_image, format="bgr24")


ctx = webrtc_streamer(
        key="real-time",
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        video_frame_callback=callback,
        # For Deploying
        rtc_configuration={
             "iceServers": [
            {
              "urls": "stun:openrelay.metered.ca:80",
            },
            {
              "urls": "turn:openrelay.metered.ca:80",
              "username": "openrelayproject",
              "credential": "openrelayproject",
            },
            {
              "urls": "turn:openrelay.metered.ca:443",
              "username": "openrelayproject",
              "credential": "openrelayproject",
            },
            {
              "urls": "turn:openrelay.metered.ca:443?transport=tcp",
              "username": "openrelayproject",
              "credential": "openrelayproject",
            },
          ]
        }
    )
