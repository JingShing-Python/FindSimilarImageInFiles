import cv2
import os
import numpy as np

def compare_images(image1, image2):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    if img1 is None or img2 is None:
        return None

    descriptor = cv2.SIFT_create()
    kp1, des1 = descriptor.detectAndCompute(img1, None)
    kp2, des2 = descriptor.detectAndCompute(img2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    min_match_count = 10
    if len(good_matches) > min_match_count:
        similarity = len(good_matches) / min(len(des1), len(des2))
        similarity = min(similarity, 1.0)  # Set similarity upper limit to 1.0 to avoid exceeding 100%
        return int(similarity * 100)

    return None

def find_similar_images(target_image, folder_path):
    similar_images = {}

    if not os.path.isfile(target_image):
        print("Target image does not exist.")
        return similar_images

    for root, _, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            similarity = compare_images(target_image, image_path)
            if similarity is not None:
                similar_images[image_path] = similarity

    return similar_images

if __name__ == "__main__":
    target_image_path = "image/a.jpg"
    folder_path = "find"

    similar_images = find_similar_images(target_image_path, folder_path)

    if not similar_images:
        print("No similar images found.")
    else:
        print("Similar images found:")
        for image_path, similarity in similar_images.items():
            print(f"Similarity: {similarity}, Location: {image_path}")
