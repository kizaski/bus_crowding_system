import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime
import cv2
import sys
import getopt
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:5000/api/people_counts"

def people_counter(input_video: Path, use_horizontal, show=False, send_to_server=False, use_vertical=True, d_line_ratio=2, mode="webcam"):

    model = YOLO("models/yolo11n.pt")

    in_out_state = {'prev_in': 0, 'prev_out': 0}

    # Annotators
    byte_tracker = sv.ByteTrack()
    bounding_box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1.5)

    if mode == "webcam":
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        video_info = sv.VideoInfo(width=width, height=height, fps=fps)
        now = datetime.now().strftime("%H-%M_%d-%B-%Y")
        sink = sv.VideoSink(
            target_path=f"results/webcam_output_{now.strip()}.mp4",
            video_info=video_info
        )

        START = None
        END = None
        line_zone = None
        if use_horizontal:
            START = sv.Point(0, int(video_info.height / d_line_ratio))
            END = sv.Point(video_info.width, int(video_info.height / d_line_ratio))
        elif use_vertical:
            START = sv.Point(int(video_info.width / d_line_ratio), 0)
            END = sv.Point(int(video_info.width / d_line_ratio), video_info.height)
        
        if show:
            cv2.namedWindow("Webcam + Supervision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Webcam + Supervision", 800, 600)  # Width x Height in pixels

        line_zone = sv.LineZone(start=START, end=END)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id in
                zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            line_zone.trigger(detections)
            annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

            if send_to_server:
                in_out_state["prev_in"], in_out_state["prev_out"] = send_to_api(
                    in_count=line_zone.in_count,
                    out_count=line_zone.out_count,
                    prev_in=in_out_state["prev_in"],
                    prev_out=in_out_state["prev_out"]
                )

            if show:
                cv2.imshow("Webcam + Supervision", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save video from webcam
            # with sink:
            #     sink.write_frame(annotated_frame)

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "video":        
        SOURCE_VIDEO_PATH = input_video

        now = datetime.now().strftime("%H-%M_%d-%B")
        TARGET_VIDEO_PATH = f"results/sv-{input_video.stem}-{now.strip()}.mp4"

        video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

        START = None
        END = None
        line_zone = None
        if use_horizontal:
            START = sv.Point(0, int(video_info.height / d_line_ratio))
            END = sv.Point(video_info.width, int(video_info.height / d_line_ratio))
        elif use_vertical:
            START = sv.Point(int(video_info.width / d_line_ratio), 0)
            END = sv.Point(int(video_info.width / d_line_ratio), video_info.height)
        
        line_zone = sv.LineZone(start=START, end=END)

        cv2.namedWindow("Video + Supervision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video + Supervision", 800, 600)

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id in
                zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            line_zone.trigger(detections)
            annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

            if send_to_server:
                in_out_state["prev_in"], in_out_state["prev_out"] = send_to_api(
                    in_count=line_zone.in_count,
                    out_count=line_zone.out_count,
                    prev_in=in_out_state["prev_in"],
                    prev_out=in_out_state["prev_out"]
                )

            if show:
                cv2.imshow("Video + Supervision", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt("Stopped by user")

            return annotated_frame

        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=callback
        )
    else:
        raise RuntimeError("Mode must be <webcam|video>")

def send_to_api(in_count, out_count, prev_in, prev_out):
    if in_count != prev_in or out_count != prev_out:
        timestamp = datetime.now().isoformat()
        data = {
            "timestamp": timestamp,
            "in_count": in_count,
            "out_count": out_count
        }
        try:
            response = requests.post(API_URL, json=data, timeout=5)
            response.raise_for_status()
            print(f"Sent to API: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data: {e}")
    return in_count, out_count

def print_usage():
    print("Usage:")
    print("  svcounter.py -m <webcam|video> -i <path to video> "
          "[(-v|--vertical) | (-z|--horizontal)] "
          "[-x <line ratio>] [-y <line ratio>]")
    print("\nOptions:")
    print("  -m, --mode           Required: 'webcam' or 'video'")
    print("  -i, --input-video    Path to input video (required if mode is 'video')")
    print("  -v                   Use a vertical line (x-axis based)")
    print("  -z                   Use a horizontal line (y-axis based)")
    print("  -r, --line-ratio     Line position ratio (e.g., 1.5 means height or width / 1.5)")
    print("  -s, --send           Send to flask server at localhost")
    print("  -h                   Show this help message")

def main(argv):
    mode = None
    input_video = Path()
    use_vertical = False
    use_horizontal = False
    d_line_ratio = None
    send_to_server = False

    try:
        opts, args = getopt.getopt(argv, "hm:i:vzr:s", ["mode=", "input-video=", "line-ratio=", "send"])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg.lower()
        elif opt in ("-i","--input-video"):
            input_video = Path(arg)
        elif opt == "-v":
            use_vertical = True
        elif opt == "-z":
            use_horizontal = True
        elif opt in ("-r", "--line-ratio"):
            d_line_ratio = float(arg)
        elif opt in ("-s", "--save"):
            send_to_server = True

    if mode not in ["webcam", "video"]:
        print_usage()
        sys.exit(2)

    if not use_horizontal and not use_vertical:
        use_horizontal = True

    if use_horizontal and use_vertical:
        raise ValueError("Only one of use_horizontal or use_vertical can be True.")

    if mode == "video":
        if not input_video or not input_video.is_file():
            print(f"Warning: File at '{input_video}' does not exist or was not provided.")
            print("Info: Using fallback './videos/FootfallVideo.mp4'")
            input_video = Path("./videos/FootfallVideo.mp4")

            if not input_video.is_file():
                print(f"Error: Fallback file at '{input_video}' also does not exist.")
                sys.exit(1)

    people_counter(
        mode=mode,
        input_video=input_video,
        use_vertical=use_vertical,
        use_horizontal=use_horizontal,
        d_line_ratio=d_line_ratio or 2,
        show=True,
        send_to_server=send_to_server
    )

if __name__ == "__main__":
    main(sys.argv[1:])
