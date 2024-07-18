import gradio as gr
import torch
import torchvision
import train_model  #this contains our model structure
import yaml

with open('config.yml','r') as conf:
    config_info = yaml.load(conf, Loader=yaml.SafeLoader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = config_info["model_savepath"]
classes = config_info["classes"]

def predict(input_img):
    in_img = train_model.transform(input_img)
    model = torch.jit.load(MODEL_PATH)
    model.to(device)
    model.eval()
    in_img = torch.reshape(in_img,(1,in_img.shape[0],in_img.shape[1],in_img.shape[2]))
    in_img = in_img.to(device)
    #print(type(in_img))
    output = torch.nn.functional.sigmoid(model(in_img))
    output = output.cpu().detach().numpy()[0]
    print(output)
    return input_img, {classes[i]: output[i] for i in range(len(classes))}

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select image", sources=['upload', 'webcam'], type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Label(label="Result", num_top_classes=2)],
    title="What Insect is it?",
)

if __name__ == "__main__":
    gradio_app.launch()