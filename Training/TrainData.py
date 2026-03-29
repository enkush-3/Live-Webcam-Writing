import os
import csv
import copy
import argparse
import itertools
import cv2 as cv
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()
    # Датасет байрлаж буй үндсэн хавтасны зам
    parser.add_argument("--dataset_dir", type=str, default="leapGestRecog", help='Path to leapGestRecog folder')
    # Гаралт болох CSV файлын нэр
    parser.add_argument("--out_csv", type=str, default="dataset.csv", help='Output CSV file path')
    
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    dataset_dir = args.dataset_dir
    csv_path = args.out_csv

    # MediaPipe тохиргоо (Зураг унших тул static_image_mode=True байна)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
    )

    # CSV файлыг шинээр үүсгэж нээх
    with open(csv_path, 'w', newline="") as f:
        writer = csv.writer(f)
        
        success_count = 0
        fail_count = 0

        # 00, 01, ..., 09 гэсэн Subject хавтаснуудаар гүйх
        if not os.path.exists(dataset_dir):
            print(f"Алдаа: {dataset_dir} хавтас олдсонгүй!")
            return

        subject_folders = sorted(os.listdir(dataset_dir))
        for subject in subject_folders:
            subject_path = os.path.join(dataset_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            # 01_palm, 02_l, ... гэсэн дохионы хавтаснуудаар гүйх
            gesture_folders = sorted(os.listdir(subject_path))
            for gesture in gesture_folders:
                gesture_path = os.path.join(subject_path, gesture)
                if not os.path.isdir(gesture_path):
                    continue

                # Хавтасны нэрнээс Label-ийг гаргаж авах (Жишээ нь "01_palm" -> 0, "10_down" -> 9)
                try:
                    label_id = int(gesture.split('_')[0]) - 1
                except ValueError:
                    continue # Нэрний бүтэц таарахгүй бол алгасах

                print(f"Уншиж байна: Subject {subject} -> Gesture {gesture} (Label: {label_id})...")

                # Зургуудаар гүйх
                # Зургуудаар гүйх
                image_files = sorted(os.listdir(gesture_path))
                for img_file in image_files:
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    img_path = os.path.join(gesture_path, img_file)
                    original_image = cv.imread(img_path)
                    
                    if original_image is None:
                        continue

                    # --- ЭНД 2 УДАА ДАВТАЖ БАЙНА ---
                    # 1-рт: Эх хувь (Зүүн гар гэж бодъё)
                    # 2-рт: Flip хийсэн хувь (Баруун гар болж хувирна)
                    for flip_mode in [None, 1]: 
                        if flip_mode is None:
                            process_img = original_image
                        else:
                            process_img = cv.flip(original_image, flip_mode)

                        # MediaPipe ажиллуулах
                        image_rgb = cv.cvtColor(process_img, cv.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)

                        if results.multi_hand_landmarks is not None:
                            for hand_landmarks in results.multi_hand_landmarks:
                                # Координатуудыг авах ба нормалчилах
                                landmark_list = calc_landmark_list(process_img, hand_landmarks)
                                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                                
                                # CSV-д бичих (Энд flip_mode бүрт нэг мөр бичигдэнэ)
                                writer.writerow([label_id, *pre_processed_landmark_list])
                                success_count += 1
                        else:
                            fail_count += 1

    print("=========================================")
    print(f"Ажиллагаа дууслаа! Өгөгдөл '{csv_path}' файлд хадгалагдлагдлаа.")
    print(f"Амжилттай танигдсан: {success_count} зураг")
    print(f"Гар танигдаагүй (Алгассан): {fail_count} зураг")
    print("=========================================")

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Бугуйны (0,0) цэгээр төвлөрүүлэх
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # 1 хэмжээст лист болгох
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Хамгийн их утгаар хувааж нормалчилах
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

if __name__ == '__main__':
    main()