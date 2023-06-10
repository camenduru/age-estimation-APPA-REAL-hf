#!/usr/bin/env python

import functools
import os
import pathlib

import cv2
import dlib
import gradio as gr
import huggingface_hub
import numpy as np
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F

DESCRIPTION = '# [Age Estimation](https://github.com/yu4u/age-estimation-pytorch)'


def get_model(model_name='se_resnext50_32x4d',
              num_classes=101,
              pretrained='imagenet'):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def load_model(device):
    model = get_model(model_name='se_resnext50_32x4d', pretrained=None)
    path = huggingface_hub.hf_hub_download(
        'public-data/yu4u-age-estimation-pytorch', 'pretrained.pth')
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    return model


def load_image(path):
    image = cv2.imread(path)
    h_orig, w_orig = image.shape[:2]
    size = max(h_orig, w_orig)
    scale = 640 / size
    w, h = int(w_orig * scale), int(h_orig * scale)
    image = cv2.resize(image, (w, h))
    return image


def draw_label(image,
               point,
               label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8,
               thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0),
                  cv2.FILLED)
    cv2.putText(image,
                label,
                point,
                font,
                font_scale, (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA)


@torch.inference_mode()
def predict(image, model, face_detector, device, margin=0.4, input_size=224):
    image = cv2.imread(image, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    image_h, image_w = image.shape[:2]

    # detect faces using dlib detector
    detected = face_detector(image, 1)
    faces = np.empty((len(detected), input_size, input_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(
            ), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), image_w - 1)
            yw2 = min(int(y2 + margin * h), image_h - 1)
            faces[i] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1],
                                  (input_size, input_size))

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(image, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

        # predict ages
        inputs = torch.from_numpy(
            np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
        outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
        ages = np.arange(0, 101)
        predicted_ages = (outputs * ages).sum(axis=-1)

        # draw results
        for age, d in zip(predicted_ages, detected):
            draw_label(image, (d.left(), d.top()), f'{int(age)}')
    return image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)
face_detector = dlib.get_frontal_face_detector()
fn = functools.partial(predict,
                       model=model,
                       face_detector=face_detector,
                       device=device)

image_dir = pathlib.Path('sample_images')
examples = [path.as_posix() for path in sorted(image_dir.glob('*.jpg'))]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input', type='filepath')
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Result')

    gr.Examples(examples=examples,
                inputs=image,
                outputs=result,
                fn=fn,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')
    run_button.click(fn=fn, inputs=image, outputs=result, api_name='predict')
demo.queue(max_size=15).launch()
