### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import yaml
import random
import cv2
from PIL import Image
import time
import string
import carla
import pygame
import numpy as np
import torchvision.transforms as transforms
import torch

sensor_data = {}

weather_presets = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset

}

class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

def convert_image_to_array(image):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    return img

def add_frame(image,name):
    img = convert_image_to_array(image)
    sensor_data[name] = img

def labels_to_cityscapes_palette(image):
    classes = {
        0: [0, 0, 0],
        1: [128, 64, 128],
        2: [244, 35, 232],
        3: [70, 70, 70],
        4: [102, 102, 156],
        5: [190, 153, 153],
        6: [153, 153, 153],
        7: [250, 170, 30],
        8: [220, 220, 0],
        9: [107, 142, 35],
        10: [152, 251, 152],
        11: [70, 130, 180],
        12: [220, 20, 60],
        13: [255, 0, 0],
        14: [0, 0, 142],
        15: [0, 0, 70],
        16: [0, 60, 100],
        17: [0, 80, 100],
        18: [0, 0, 230],
        19: [119, 11, 32],
        20: [110, 190, 160],
        21: [170, 120, 50],
        22: [55, 90, 80],
        23: [45, 60, 150],
        24: [157, 234, 50],
        25: [81, 0, 81],
        26: [150, 100, 100],
        27: [230, 150, 140],
        28: [180, 165, 180]
    }
    array = image[:,:,0]
    print(array.shape)
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result.astype(np.uint8)

def inst_labels_to_colors(image):
    instance_masks = image[:,:,1]
    num_instances = np.max(image) + 1
    instance_colors = np.random.randint(0, 255, size=(num_instances, 3))

    colorized_masks = np.zeros((instance_masks.shape[0], instance_masks.shape[1], 3), dtype=np.uint8)
    for i in range(num_instances):
        mask = (instance_masks == i)
        colorized_masks[mask] = instance_colors[i]

    return colorized_masks.astype(np.uint8)

carla_settings = None
try:
    with open('carla_settings.yaml', 'r') as file:
        carla_settings = yaml.safe_load(file)
except:
    print("Carla settings config (yaml) file does not exist...")

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
if opt.dataset_mode == 'temporal':
    opt.dataset_mode = 'test'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
input_nc = 1 if opt.label_nc != 0 else opt.input_nc

actor_list = []
done = False

def is_vehicle_moving(vehicle, threshold=0.1, capture_static = True):
    if capture_static == True:
        return True

    velocity = vehicle.get_velocity()
    speed = velocity.length()
    return speed > threshold

try:
    export_data = str(carla_settings['dataset']['export_data'])

    if export_data:
        if not os.path.isdir("datasets/Carla"):
            os.makedirs("datasets/Carla")
            print("Creating CARLA dictionary...")
        if not os.path.isdir("datasets/Carla/Frames"):
            os.makedirs("datasets/Carla/Frames")
            print("Creating Frames dictionary...")
        if not os.path.isdir("datasets/Carla/Synthesized"):
            os.makedirs("datasets/Carla/Synthesized")
            print("Creating Synthesized dictionary...")
        if not os.path.isdir("datasets/Carla/Semantic"):
            os.makedirs("datasets/Carla/Semantic")
            print("Creating Semantic dictionary...")
        if not os.path.isdir("datasets/Carla/Instance"):
            os.makedirs("datasets/Carla/Instance")
            print("Creating Instance dictionary...")

    client = carla.Client(str(carla_settings['connection']['ip']), int(carla_settings['connection']['port']))
    client.set_timeout(float(carla_settings['connection']['timeout']))
    sync_mode = bool(carla_settings['world']['synchronous_mode'])
    run_model_every_n = int(carla_settings['general']['run_model_every_n'])
    visualize_results = bool(carla_settings['general']['visualize_results'])
    colorize_masks = bool(carla_settings['general']['colorize_masks'])
    export_step = int(carla_settings['dataset']['export_step'])
    capture_when_static = bool(carla_settings['dataset']['capture_when_static'])
    speed_threshold = float(carla_settings['dataset']['speed_threshold'])
    world = client.get_world()

    world = client.load_world(str(carla_settings['world']['town']))
    settings = world.get_settings()
    settings.synchronous_mode = sync_mode
    settings.fixed_delta_seconds = float(carla_settings['world']['fixed_delta_seconds'])
    world.apply_settings(settings)

    world.set_weather(weather_presets[str(carla_settings['world']['weather_preset'])])

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter(str(carla_settings['general']['vehicle']))[0]
    transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points()) - 1)]
    vehicle = world.spawn_actor(vehicle_bp, transform)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)

    vehicle.set_autopilot(True)

    image_w = int(carla_settings['general']['cam_width'])
    image_h = int(carla_settings['general']['cam_height'])
    num_sensors = 0

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_w))
    camera_bp.set_attribute('image_size_y', str(image_h))

    camera_transform = carla.Transform(carla.Location(x=float(carla_settings['general']['cam_x']), z=float(carla_settings['general']['cam_z'])))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)
    print('created %s' % camera.type_id)
    num_sensors += 1

    camera_bp2 = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp2.set_attribute('image_size_x', str(image_w))
    camera_bp2.set_attribute('image_size_y', str(image_h))
    camera_semseg = world.spawn_actor(camera_bp2, camera_transform, attach_to=vehicle)
    actor_list.append(camera_semseg)
    print('created %s' % camera_semseg.type_id)
    num_sensors += 1


    camera_bp3 = blueprint_library.find('sensor.camera.instance_segmentation')
    camera_bp3.set_attribute('image_size_x', str(image_w))
    camera_bp3.set_attribute('image_size_y', str(image_h))
    camera_inst = world.spawn_actor(camera_bp3, camera_transform, attach_to=vehicle)
    actor_list.append(camera_inst)
    print('created %s' % camera_inst.type_id)
    num_sensors += 1

    camera.listen(lambda image: add_frame(image,"frame"))
    camera_semseg.listen(lambda image: add_frame(image,"label"))
    camera_inst.listen(lambda image: add_frame(image,"label_inst"))

    pygame_width = int(carla_settings['pygame']['window_width'])
    pygame_height = int(carla_settings['pygame']['window_height'])

    renderObject = RenderObject(pygame_width, pygame_height)
    pygame.init()
    gameDisplay = pygame.display.set_mode((pygame_width, pygame_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    gameDisplay.fill((0, 0, 0))
    gameDisplay.blit(renderObject.surface, (0, 0))
    pygame.display.flip()

    current_step = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()

        if sync_mode == True:
            for i in range(run_model_every_n):
                world.tick()
                current_step += 1
                while True:
                    if len(sensor_data) == num_sensors:
                        break
                if not i == run_model_every_n -1:
                    sensor_data = {}
        vehicle.set_autopilot(True)


        label = sensor_data['label'][:,:,0]
        inst_label = np.full((label.shape[0],label.shape[1],3),255)

        #the carla label id's are not out of the box compatible with cityscapes
        #its better to modify the classes ID's from the engine since this will drop the performance
        old_class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] #carla classes
        new_integer_ids = [0,7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,4,5,0,0,7,6,15,10,14] #cityscapes classes

        label = np.select([label == old_idx for old_idx in old_class_indexes],[new_id for new_id in new_integer_ids], default=label)

        label = np.stack((label,) * 3, axis=-1)

        label = np.expand_dims(label, axis=0)
        inst_label = np.expand_dims(inst_label, axis=0)

        label = np.transpose(label, (0, 3, 1, 2))
        inst_label = np.transpose(inst_label, (0, 3, 1, 2))

        label = label.astype(np.float32)
        inst_label = inst_label.astype(np.float32)

        label = torch.tensor(label)
        inst_label = torch.tensor(inst_label)

        _, _, height, width = label.size()
        A = Variable(label).view(1, -1, input_nc, height, width)
        B = None
        inst = Variable(inst_label).view(1, -1, 1, height, width) if len(inst_label.size()) > 2 else None

        generated = model.inference(A,B, inst)

        c = 3 if opt.input_nc == 3 else 1
        generated = util.tensor2im(generated[0].data[0])
        cv_generated = cv2.resize(generated, (pygame_width, pygame_height))
        renderObject.surface = pygame.surfarray.make_surface(cv_generated.swapaxes(0, 1))
        gameDisplay.fill((0, 0, 0))
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()

        if visualize_results:
            cv_frame = cv2.resize(sensor_data['frame'], (512, 256))
            cv_synthesized = cv2.resize(generated, (512, 256))

            if colorize_masks:
                cv_label = cv2.resize(labels_to_cityscapes_palette(sensor_data['label']), (512, 256))
                cv_label_inst = cv2.resize(sensor_data['label_inst'], (512, 256))
            else:
                cv_label = cv2.resize(sensor_data['label'], (512, 256))
                cv_label_inst = cv2.resize(sensor_data['label_inst'], (512, 256))

            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            cv_synthesized = cv2.cvtColor(cv_synthesized, cv2.COLOR_BGR2RGB)
            cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGR2RGB)
            cv_label_inst = cv2.cvtColor(cv_label_inst, cv2.COLOR_BGR2RGB)

            top_row = np.hstack((cv_frame, cv_synthesized))
            bottom_row = np.hstack((cv_label, cv_label_inst))
            stacked_image = np.vstack((top_row, bottom_row))
            cv2.imshow('Visualization', stacked_image)

        if current_step % export_step == 0 and sync_mode == True:
            if export_data and is_vehicle_moving(vehicle,speed_threshold,capture_when_static):
                random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                pil_image = Image.fromarray(sensor_data['frame'])
                pil_image.save("datasets\\Carla\\Frames\\"+random_string+".jpg")

                pil_image = Image.fromarray(sensor_data['label_inst'])
                pil_image.save("datasets\\Carla\\Instance\\"+random_string+".jpg")

                pil_image = Image.fromarray(sensor_data['label'])
                pil_image.save("datasets\\Carla\\Semantic\\"+random_string+".jpg")

                pil_image = Image.fromarray(generated)
                pil_image.save("datasets\\Carla\\Synthesized\\"+random_string+".jpg")

        if sync_mode == True:
            sensor_data = {}
        model = create_model(opt) #if we let the model to predict based on the previous frames on CARLA the results are bad.
except Exception as e:
    print(e)
    print('destroying actors')
    camera.destroy()
    camera_semseg.destroy()
    vehicle.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('done.')