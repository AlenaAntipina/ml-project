#THis code is to check if the video is corrupted or not..
#If the video is corrupted delete the video.
import torch
import glob
from torchvision import transforms
import cv2

from config import AppConfig


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])


#Check if the file is corrupted or not
def validate_video(vid_path,train_transforms):
    transform = train_transforms
    count = 20
    video_path = vid_path
    frames = []
    for i,frame in enumerate(frame_extract(video_path)):
        frames.append(transform(frame))
        if(len(frames) == count):
            break
    frames = torch.stack(frames)
    frames = frames[:count]
    return frames


#extract a from from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def main_actions(config: AppConfig):
    video_files = [str(x) for x in config.dataset_path.glob("**/*.mp4")]
    print("Total no of videos :" , len(video_files))

    count = 0
    for i in video_files:
        try:
            count+=1
            validate_video(i,train_transforms)
        except:
            print("Number of video processed: " , count ," Remaining : " , (len(video_files) - count))
            print("Corrupted video is : " , i)
            continue
    print((len(video_files) - count))



def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()