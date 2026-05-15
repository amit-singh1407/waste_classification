import gradio as gr
from ultralytics import YOLO            
from PIL import Image
import tempfile
import cv2

def combine(img_files):
    img_array = []
    import os
    for filename in img_files:
        img = cv2.imread(filename.name)
        if img is None:
            raise ValueError(f"Unable to read image file: {filename.name}")
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)
    output_file = "test.mp4"
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"h264")
    out = cv2.VideoWriter(output_file, fourcc, 15, size) 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return output_file

def image_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):
    '''
    img_path: str, path to the image
    model_id: str, model id
    image_size: int, image size
    conf_threshold: float, confidence threshold (its the minimum confidence score for a bounding box to be considered)
    iou_threshold: float, IoU threshold (its the minimum IoU score for a bounding box to be considered) [IoU = Intersection over Union]
    '''
    if model_id == 'best':
        model = YOLO(r'waste_detection\yolo\weights\best.pt')
    else:
        model = YOLO(r'waste_detection\yolo\weights\last.pt')
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

    return im

def yolo_infer_video(video, model_id, image_size, conf_threshold, iou_threshold):
    '''
    video: str, path to the video
    model_id: str, model id
    image_size: int, image size
    conf_threshold: float, confidence threshold (its the minimum confidence score for a bounding box to be considered)
    iou_threshold: float, IoU threshold (its the minimum IoU score for a bounding box to be considered) [IoU = Intersection over Union]
    '''
    model = YOLO(r'waste_detection\yolo\weights\best.pt')
    results = model.predict(
        source=video,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=image_size,
    )
    img_files = []
    for r in results:
        im_array = r.plot()
        # save the image
        im = Image.fromarray(im_array[..., ::-1])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        im.save(temp_file.name)
        img_files.append(temp_file)
    return combine(img_files)


def upload_file(file):
    return file.name
    
def app():
    with gr.Blocks():

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
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.File()
                    upload_button = gr.UploadButton("Click to Upload a Video", file_types=["video"], file_count="single")
                    upload_button.upload(upload_file, upload_button, video)
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
                    yolo_infer2 = gr.Button(value="Detect waste in Video")
                with gr.Column():
                    # combine the images to create a video
                    output_video = gr.Video()



        
        
        yolo_infer2.click(
            fn=yolo_infer_video,
            inputs=[
                video,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_video],
        )

        yolo_infer.click(
            fn=image_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
        )

        

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Object Detection Using YOLO
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True, inbrowser=True)