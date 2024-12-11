import os, shutil, cv2
import numpy as np

def extract_video_segment(input_path, output_path, start_time, end_time):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Ошибка открытия видео!")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #type:ignore

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame < end_frame and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Сохранён отрезок видео: {output_path}")

weights_file = 'runs\\detect\\train\\weights\\best.pt'

if os.path.exists('results'):
    shutil.rmtree('results')

if os.path.exists('yolov9\\runs\\detect'):
    shutil.rmtree('yolov9\\runs\\detect')

while True:
    choice = int(input("Фото или видео?(1 или 2)"))

    if choice == 1:
        images_folder = 'SignsDataset\\images\\test'
        temp_folder = 'temp'
        save_to_folder = 'photo'
        see_results = 'yolov9\\runs\\detect\\photo'
        limit = 50

        os.makedirs(temp_folder, exist_ok=True)

        all_images = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        selected_images = all_images[:limit]

        for image in selected_images:
            shutil.copy(os.path.join(images_folder, image), os.path.join(temp_folder, image))

        os.system(f"python yolov9\\detect.py --weights {weights_file} --source {temp_folder} --name {save_to_folder}")

        shutil.rmtree(temp_folder)
        if not os.path.exists(see_results):
            print(f"Папка {see_results} не найдена. Проверьте, завершился ли процесс YOLO успешно.")
            exit()

        result_images = [os.path.join(see_results, f) for f in os.listdir(see_results) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if not result_images:
            print("Нет изображений для просмотра.")
            exit()

        def view_results(images):
            idx = 0

            while True:
                img = cv2.imdecode(np.fromfile(images[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow("Results Viewer", img)  #type:ignore

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 233:
                    break
                elif key == 244 or key == 97:
                    idx = (idx - 1) % len(images)
                elif key == 226 or key == 100:
                    idx = (idx + 1) % len(images)

            cv2.destroyAllWindows()

        view_results(result_images)

    elif choice == 2:
        results_folder = 'results\\video'
        videos_folder = "SignsDataset\\Videos"
        new_video_path = 'new_video\\video.mp4'
        type_video = int(input("День или ночь?(1 или 2)"))
        if type_video == 1:
            video = "Day.mp4"
        elif type_video == 2:
            video = "Night.mp4"
        else: video = "Day.mp4"

        extract_video_segment(f"{videos_folder}\\{video}", "new_video\\video.mp4", start_time = 0, end_time = 240)

        os.system(f"python yolov9\\detect.py --weights {weights_file} --source {new_video_path} --project {results_folder}")

        cap = cv2.VideoCapture(f"{results_folder}\\exp\\video.mp4")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("YOLO Detection Results", frame)
            key = cv2.waitKey(25)
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else: continue