import os
from pytube import YouTube
from tqdm import tqdm
import cv2
import numpy as np
import json

def download_dataset():

    def download_video(video_path, save_root):
        video_name = video_path.split("=")[1]
        # if os.path.exists(os.path.join(save_root, video_name+".mp4")): return

        yt = YouTube(video_path)
        try:
            stream = yt.streams.first()
            stream.download(save_root,video_name+".mp4",skip_existing=True, timeout=10, max_retries=0)
        except:
            pass

    # dataset = "acid"
    dataset = "RealEstate10K"
    root = "/Users/lhy/Desktop/dataset/"+dataset+"_data"

    video_list = []
    for split in os.listdir(root):
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root): continue

        for txt_name in tqdm(os.listdir(split_root), desc=split):
            if not txt_name.endswith("txt"): continue
            txt_path = os.path.join(split_root, txt_name)
            with open(txt_path, "r") as file:
                video_path = file.readline().replace("\n","")
                video_list.append(video_path)
        
    '''
    acid file count:  28940
    video count: 1510 -> 30G

    RealEstate10K file count:  79267
    video count: 7255 -> 150GB
    '''
    print(dataset+" file count: ", len(video_list))
    video_list = list(set(video_list))
    print("video count:", len(video_list))

    save_root = "/Users/lhy/Desktop/vs/workspace/video"

    # for video_path in video_list:
    #     video_name = video_path.split("=")[1]
    #     if os.path.exists(os.path.join(save_root, video_name+".mp4")): video_list.remove(video_path)

    video_list.sort()

    # index  = 3530
    # print("index", index)
    # video_list = video_list[index:]
    for video_path in tqdm(video_list):
        download_video(video_path, save_root)

def check_video():
    save_root = "/Users/lhy/Desktop/vs/workspace/video"
    video_list = []
    for video_name in tqdm(os.listdir(save_root)):
        if video_name in ["0Cfv8dbxTSc.mp4"]: continue
        if not video_name.endswith(".mp4"): continue
        video_path = os.path.join(save_root, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_num = cap.get(7)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1540)
        ret, frame1 = cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-820)
        ret, frame2 = cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-100)        
        ret, frame3 = cap.read()

        if frame3 is None:
            video_list.append(video_path)
        elif (frame1 == frame2).all() and (frame1 == frame3).all():
            video_list.append(video_path)
    print("video_list", len(video_list))
    print(video_list)
    # for video_path in video_list:
    #     os.system("rm"+" "+video_path)


def extract_frame():

    root = "/Users/lhy/Desktop/dataset/acid_data"
    save_root = "/Users/lhy/Desktop/vs/workspace/video"
    for split in os.listdir(root):
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root): continue

        for txt_name in tqdm(os.listdir(split_root), desc=split):
            txt_path = os.path.join(split_root, txt_name)
            if os.path.isdir(txt_path): continue
            with open(txt_path, "r") as file:
                video_url = file.readline().replace("\n","")
                video_infos = file.readlines()
            video_name = video_url.split("=")[1]
            video_path = os.path.join(save_root, video_name+".mp4")
            if not os.path.exists(video_path): continue

            item_id = txt_name.split(".")[0]
            item_root = os.path.join(root, split, item_id)
            os.makedirs(item_root, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            camera_pose = {}

            img_root = os.path.join(item_root, "train")
            os.makedirs(img_root, exist_ok=True)

            for video_info in video_infos:
                video_info = video_info.replace("\n","").split(" ")
                cap.set(cv2.CAP_PROP_POS_MSEC, int(int(video_info[0])/1000)) ## ??

                ret, frame = cap.read()
                h, w, c = frame.shape

                cv2.imwrite(os.path.join(img_root, video_info[0]+".png"), frame)

                intrinsics = [
                    [w*float(video_info[1]), 0, w*float(video_info[3])],
                    [0, h*float(video_info[2]), h*float(video_info[4])],
                    [0, 0, 1]
                ]
                entrinsics = [
                    [float(video_info[7]), float(video_info[8]), float(video_info[9]), float(video_info[10])],
                    [float(video_info[11]), float(video_info[12]), float(video_info[13]), float(video_info[14])],
                    [float(video_info[15]), float(video_info[16]), float(video_info[17]), float(video_info[18])],
                    [0, 0, 0, 1]
                ]

                camera_pose[video_info[0]] = {
                    "intrinsics": intrinsics,
                    "entrinsics": entrinsics
                }
            with open(os.path.join(item_root, "transforms_train.json"), "w") as file:
                json.dump(camera_pose, file, indent=4)

            cap.release()
            os.system("mv "+txt_path+" "+os.path.join(item_root, txt_name))

download_dataset()
# check_video()
