from streamlit_webrtc import webrtc_streamer

webrtc_streamer(
        key="real-time",
        video_frame_callback=callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
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
