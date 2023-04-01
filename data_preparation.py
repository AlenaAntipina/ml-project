from config import AppConfig
import os
import cv2
from PIL import Image
from retinaface import RetinaFace
import numpy as np

def getFiles(dir):
    dirFiles = os.listdir(dir)
    files = list()
    for file in dirFiles:
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            files = files + getFiles(path)
        else:
            files.append(path)              
    return files

def generate_video(path_to_images, input_video_name, output_path):
    main_dir = os.getcwd()
    os.chdir(path_to_images)
    path = path_to_images

    for file in os.listdir('.'):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            im = Image.open(os.path.join(path, file))

            imResize = im.resize((112, 112), Image.ANTIALIAS)
            imResize.save( file, 'jpeg', quality = 95) 

    image_folder = path 
    video_name = input_video_name + '.mp4'	
    images = [img for img in os.listdir(image_folder)
			if img.endswith(".jpg") or
				img.endswith(".jpeg")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    print("path for new video: ", os.path.join(output_path, video_name))
    video = cv2.VideoWriter(os.path.join(output_path, video_name), -1, 15, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release() 
    os.chdir(main_dir)


def extract_face(path, output_video_path):
    main_dir = os.getcwd()
    video_name = os.path.splitext(os.path.basename(path))[0]

    outputPath = './extracted'
    path_to_save = os.path.join(outputPath, video_name)

    # поменять два ифа местами, сначал проверка что видео есть, потом создание файла
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    if not os.path.exists(os.path.join(main_dir, output_video_path, video_name + ".mp4")):
        all_images = []
        print('[INFO] extracting frames from video...', " | videofile - ", video_name)
        video = cv2.VideoCapture(path)
        while True:
            success, frame = video.read()
            if success:
                image = {
                    "file": frame,
                    "source": path,
                    "sourceType": "video",
                    "outputPath": os.path.join(outputPath,video_name),
                    "filename": video_name
                }
                all_images.append(image)
            else:
                break
        video.release()
        cv2.destroyAllWindows()

        images = []
        each_frame = int(len(all_images) / 100) + 1
        for i in range(len(all_images)):
            if i % each_frame == 0:
                images.append(all_images[i])

        total = 0
        cwd = os.getcwd()
        for (i, image) in enumerate(images):
            print("[INFO] processing image {}/{}".format(i + 1, len(images)))
            results = RetinaFace.detect_faces(image["file"])

            array = cv2.cvtColor(image['file'], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(array)
            
            if type(results) != dict:
                continue

            j = 1
            for key, face in results.items():
                scale = float(1.0)
                (startX, startY, endX, endY) = face['facial_area']
                bW = endX - startX
                bH = endY - startY
                centerX = startX + (bW / 2.0)
                centerY = startY + (bH / 2.0)
                left = centerX - bW / 2.0 * scale
                top = centerY - bH / 2.0 * scale
                right = centerX + bW / 2.0 * scale
                bottom = centerY + bH / 2.0 * scale
                face = img.crop((left, top, right, bottom))
                fW, fH = face.size
                
                if fW < 10 or fH < 10:
                    continue

                outputFilename = '{}_{:04d}_{}.jpg'.format(image["filename"], i, j)

                outputDir = os.path.dirname(os.path.join(cwd, image["outputPath"], "extracted"))
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)
                outputPath = os.path.join(outputDir, outputFilename)
                face.save(outputPath)
                total += 1
                j += 1

        generate_video(os.path.join(main_dir, "extracted", video_name), video_name, os.path.join(main_dir, output_video_path))


def main_actions(config: AppConfig): 
    print("start preparation")
    path_list=config.dataset_path
    out_dir = config.dataset_output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    videos_path_list = []    
    videos_path_list += [x for x in config.dataset_path.glob("**/*.mp4")]
    for video in videos_path_list:
        extract_face(str(video), out_dir)

def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()