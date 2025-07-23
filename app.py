import gradio as gr
from PIL import Image
from predict import predict_image , predict_video , generate_pdf
from scripts import clear_all

def export_summary_pdf():
    path = generate_pdf()
    return path


with gr.Blocks() as app:
    # Header block
    with gr.Row():
        with gr.Column(scale=0, min_width=100 , variant="compact"):
            gr.Image("static/guc-logo.png",height=100, show_download_button=False, show_fullscreen_button=False, container=False, label= False)    
        
        with gr.Column(scale=1):
            gr.Markdown("""
                    ### Youssef Ibrahim  
                    **Supervised by:**  
                    Dr. Hossam Eldin Hassan Abdelmunim
                    """)
        with gr.Column(scale=2, min_width=50):
            gr.Markdown("<h1 style='font-size: 3em;'>DeepFake Detection</h1>")
       
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["Custom CNN", "XceptionNet", "VGGFace"],
                    value="Custom CNN",
                    label="Model",
                    scale=1,
                    interactive=True,
                    allow_custom_value=False,
                    filterable=False
                )
                device_selector = gr.Dropdown(
                    choices=["GPU", "CPU"],
                    value="GPU",
                    label="Device",
                    scale=1,
                    interactive=True,
                    allow_custom_value=False,
                    filterable=False
                )
                threshold_slider = gr.Slider(
                    minimum=0.5,
                    maximum=0.9,
                    value=0.7,
                    step=0.01,
                    label="Threshold",
                    scale=2,
                    interactive=True,
                )

    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("Image"):
                image_input = gr.Image(type="pil", label="Upload Image", scale=1)
                with gr.Tabs():
                    with gr.TabItem("Annotated Image"):
                        image_with_boxes = gr.Image(label="Annotated Image", type="pil", scale=1)
                    with gr.TabItem("Faces"):
                        image_faces = gr.Gallery(label="Faces", columns=3,scale=1)
            with gr.TabItem("Video"):
                video_input = gr.Video(label="Upload Video", scale=1)
                with gr.Tabs():
                    with gr.TabItem("Frames"):
                        video_frames = gr.Gallery(label="Frames", columns=5,scale=1)
                    with gr.TabItem("Annotated Frames"):
                        annotated_video_frames = gr.Gallery(label="Annotated Frames", columns=5, scale=1)
                    with gr.TabItem("Faces"):
                        video_faces = gr.Gallery(label="Faces", columns=5,scale=1)
                    
     
    with gr.Row():
        detect_button = gr.Button("🔍 Start analysis")
        clear_button = gr.Button("🧹 Clear")


    with gr.Row():
        output_text = gr.Textbox(label="Prediction Status",scale=6)
        export_pdf_button = gr.Button("📄 Generate Report",scale=1)
        hidden_pdf_file = gr.File(visible=False)
    

    detect_button.click(fn=predict_image, inputs=image_input,outputs=[image_faces, image_with_boxes, output_text])
    detect_button.click(fn=predict_video, inputs=video_input, outputs=[video_frames, video_faces, annotated_video_frames, output_text])
    clear_button.click(fn=clear_all,inputs=[],outputs=[image_input,video_input,image_faces,image_with_boxes,video_faces,video_frames,annotated_video_frames,output_text])
    pdf_file_output = gr.File(label="Report", visible=True)
    export_pdf_button.click(fn=export_summary_pdf, outputs=pdf_file_output)


    
    #image_input.change(fn=predict_image, inputs=image_input, outputs=[image_faces,image_with_boxes, output_text])
    #video_input.change(fn=predict_video, inputs=video_input, outputs=[video_frames, video_faces,annotated_video_frames, output_text])

if __name__ == "__main__":
    app.launch()
