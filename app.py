import os
import socket

import gradio as gr
from PIL import Image
from ultralytics import YOLO

from video_view import process_video


def get_launch_port(default_port=7860):
    port_value = os.environ.get("PORT")
    if port_value:
        return int(port_value)

    port = default_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
        return port


def get_launch_host():
    return "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"

def image_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):
    '''
    img_path: str, path to the image
    model_id: str, model id
    image_size: int, image size
    conf_threshold: float, confidence threshold (its the minimum confidence score for a bounding box to be considered)
    iou_threshold: float, IoU threshold (its the minimum IoU score for a bounding box to be considered) [IoU = Intersection over Union]
    '''
    if model_id == 'best':
        model_path = os.path.join('waste_detection', 'yolo', 'weights', 'best.pt')
    else:
        model_path = os.path.join('waste_detection', 'yolo', 'weights', 'last.pt')
    
    model = YOLO(model_path)
    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=image_size,
    )
    for r in results:
        im_array = r.plot()
        # save the image
        im = Image.fromarray(im_array[..., ::-1])

        detected_classes = []
        boxes = getattr(r, "boxes", None)
        cls = getattr(boxes, "cls", None)
        if cls is not None:
            detected_classes = list(set([r.names[i] for i in cls.tolist()]))
        if detected_classes:
            detected_classes_str = ", ".join(detected_classes)
            msg = "Detected items: " + detected_classes_str
        else:
            msg = "No items detected"
    return im, msg

def setup_video_inference(video, size, show_preview, save_video):
    folder = process_video(video, size, show_preview, save_video)
    output_files = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    )
    if output_files:
        return output_files[0]

    raise RuntimeError(f"No output video was generated in {folder}")

def build_interface():
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    img_path = gr.Image(type="filepath", label="Image", sources=['upload','clipboard'])
                    model_path = gr.Dropdown(
                        label="Model",
                        choices=[
                            'best',
                            'last',
                        ],
                        value="best",
                    )
                    image_size = gr.Slider(
                        label="Image Size",
                        minimum=320,
                        maximum=1280,
                        step=32,
                        value=640,
                    )
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.4,
                    )
                    iou_threshold = gr.Slider(
                        label="IoU Threshold",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                    )
                    yolo_infer = gr.Button(value="Detect waste in Image")

                with gr.Column():
                    output_numpy = gr.Image(type="numpy",label="Output")
                    output_text = gr.Label(label="Output Text")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(sources=['upload'], format="mp4")
                    frame_size = gr.Slider(label="Frame Size", minimum=320, maximum=1000, step=32, value=640)
                    show_preview = gr.Checkbox(label="Show Preview (not recommended for laptops)", value=False)
                    save_video = gr.Checkbox(label="Save Video", value=True)
                    launch_video = gr.Button(value="Detect waste in Video", )
                with gr.Column():
                    output_video = gr.Video(label="Output Video", sources=[])
                   
        yolo_infer.click(
            fn=image_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy, output_text],
        )

        launch_video.click(
            fn=setup_video_inference,
            inputs=[video, frame_size, show_preview, save_video],
            outputs=[output_video]
        )

def create_app():
    with gr.Blocks() as gradio_app:
        gr.HTML(
            """
        <h1 style='text-align: center'>
        Waste Detection 
        </h1> 
        """
        )
        build_interface()
    return gradio_app


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name=get_launch_host(), server_port=get_launch_port())
