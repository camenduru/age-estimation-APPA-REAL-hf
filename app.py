#!/usr/bin/env python

import argparse
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

ORIGINAL_REPO_URL = 'https://github.com/yu4u/age-estimation-pytorch'
TITLE = 'yu4u/age-estimation-pytorch'
DESCRIPTION = f'This is an unofficial demo for {ORIGINAL_REPO_URL}.'
ARTICLE = None

TOKEN = os.environ['TOKEN']
MODEL_REPO = 'hysts/yu4u-age-estimation-pytorch'
MODEL_FILENAME = 'pretrained.pth'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


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
    path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                           MODEL_FILENAME,
                                           use_auth_token=TOKEN)
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
    image = cv2.imread(image.name, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
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


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)
    face_detector = dlib.get_frontal_face_detector()

    func = functools.partial(predict,
                             model=model,
                             face_detector=face_detector,
                             device=device)
    func = functools.update_wrapper(func, predict)

    image_dir = pathlib.Path('sample_images')
    examples = [path.as_posix() for path in sorted(image_dir.glob('*.jpg'))]

    gr.Interface(
        func,
        gr.inputs.Image(type='file', label='Input'),
        gr.outputs.Image(label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
