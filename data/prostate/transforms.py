import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F
from PIL import Image

def transforms_back_rot(ema_output,rot_mask, flip_mask):

    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2,1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output


def transforms_for_noise(inputs_u2, std):

    gaussian = np.random.normal(0, std, (inputs_u2.shape[0], 3, inputs_u2.shape[-1], inputs_u2.shape[-1]))
    gaussian = torch.from_numpy(gaussian).float().cuda()
    inputs_u2_noise = inputs_u2 + gaussian

    return inputs_u2_noise

def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1,2])

    return ema_inputs, rot_mask, flip_mask


def transforms_for_scale(ema_inputs, image_size):

    scale_mask = np.random.uniform(low=0.9, high=1.1, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(item) for item in scale_mask]
    scale_mask = [item-1 if item % 2 != 0 else item for item in scale_mask]
    half_size = int(image_size / 2)

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1,2,0))
        # crop
        if scale_mask[idx] > image_size:
            
            new_img1 = np.expand_dims(np.pad(img[:, :, 0],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img2 = np.expand_dims(np.pad(img[:, :, 1],
                                             (int((scale_mask[idx]-image_size)/2),
                                             int((scale_mask[idx]-image_size)/2)), 'edge'), axis=-1)
            new_img3 = np.expand_dims(np.pad(img[:, :, 2],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img = np.concatenate([new_img1, new_img2, new_img3], axis=-1)
            img = new_img
        else:
            img = img[half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2),:]

        # resize
        resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        # to tensor
        ema_outputs[idx] = torch.from_numpy(resized_img.transpose((2, 0, 1))).cuda()

    return ema_outputs, scale_mask

def transforms_back_scale(ema_inputs, scale_mask, image_size):
    half_size = int(image_size/2)
    returned_img = np.zeros((ema_inputs.shape[0],  image_size, image_size, 2))

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().detach().numpy(), (1,2,0))
        # resize
        resized_img = cv2.resize(img, (int(scale_mask[idx]), int(scale_mask[idx])), interpolation=cv2.INTER_CUBIC)

        if scale_mask[idx] > image_size:
            returned_img[idx] = resized_img[int(scale_mask[idx]/2)-half_size:int(scale_mask[idx]/2)+half_size,
            int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx]/2) + half_size, :]

        else:
            returned_img[idx, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2), :] = resized_img
        # to tensor
        ema_outputs[idx] = torch.from_numpy(returned_img[idx].transpose((2,0,1))).cuda()

    return ema_outputs, scale_mask

def postprocess_scale(input, scale_mask, image_size):
    half_size = int(input.shape[-1]/2)
    new_input = torch.zeros((input.shape[0], 2, input.shape[-1], input.shape[-1]))

    for idx in range(input.shape[0]):

        if scale_mask[idx] > image_size:
            new_input = input
    
        else:
            new_input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)] \
            = input[idx, :, half_size-int(scale_mask[idx]/2):half_size + int(scale_mask[idx]/2),
            half_size-int(scale_mask[idx]/2): half_size + int(scale_mask[idx]/2)]

    return new_input.cuda()
