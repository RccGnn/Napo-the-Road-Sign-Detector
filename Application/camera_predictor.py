import av
import cv2
import streamlit as st
import threading
from datetime import datetime
import pandas as pd
import queue

history_queue = queue.Queue()

from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer
)

from PIL import Image
from predictor import predict

"""---"""

class SignDetector(VideoProcessorBase):

    def __init__(self, model_name):

        self.model_name = model_name
        self.counter = 0
        self.result = "Analisi..."
        self.lock = threading.Lock()
        self.prediction_lock = threading.Lock()

        # storico live
        self.last_signal = None
        self.last_time = datetime.now()


    def run_prediction(self, image):

        try:

            results, _ = predict(
                image,
                self.model_name
            )

            signal = results[0][0]
            confidence = results[0][1]

            now = datetime.now()

            # Salva solo se cambia segnale
            # oppure se sono passati 5 secondi

            if (
                    signal != self.last_signal
                    or
                    (now - self.last_time).seconds > 5
            ):

                if "history" not in st.session_state:
                    st.session_state.history = []

                history_queue.put(
                    {
                        "Ora": now.strftime("%H:%M:%S"),
                        "Segnale": signal,
                        "Probabilità (%)": round(confidence, 2),
                        "Modello": self.model_name,
                        "Sorgente": "Telecamera Live"
                    }
                )

                self.last_signal = signal
                self.last_time = now

            text = (
                results[0][0]
                +
                " "
                +
                str(round(results[0][1], 2))
                +
                "%"
            )


            with self.lock:
                self.result = text


        except Exception as e:

            with self.lock:
                self.result = str(e)

        finally:
            self.prediction_lock.release()


    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        self.counter += 1


        # Una predizione ogni 30 frame
        if self.counter % 60 == 0:

            if self.prediction_lock.acquire(blocking=False):
                rgb = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                pil_image = Image.fromarray(
                    rgb
                )

                thread = threading.Thread(
                    target=self.run_prediction,
                    args=(pil_image,)
                )

                thread.start()



        with self.lock:

            text = self.result


        cv2.putText(
            img,
            text,
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


def run_camera(model_name):

    class CameraProcessor(SignDetector):
        def __init__(self):
            super().__init__(model_name)

    webrtc_streamer(
        key="road-sign-camera",
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480
            },
            "audio": False
        },
        async_processing=False
    )