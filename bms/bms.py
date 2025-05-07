import cv2
import numpy as np
from skimage import measure


def resize_and_convert_to_lab(image, target_width=600):
    h, w = image.shape[:2]
    if w != target_width:
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab


def generate_boolean_maps(image_lab, thresholds_per_channel=4, apply_opening=True):
    """
    Generuje mapy binarne oraz ich inwersje dla każdego kanału LAB i progu.
    Dodatkowo wykonuje otwarcie morfologiczne dla każdej mapy.
    """
    h, w, _ = image_lab.shape
    boolean_maps = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    channel_maps = cv2.split(image_lab)

    for channel_index, channel in enumerate(channel_maps):
        min_val, max_val = np.min(channel), np.max(channel)
        thresholds = np.linspace(min_val, max_val, thresholds_per_channel + 2)[1:-1]  # pomijamy min i max

        for t in thresholds:
            # Mapa: piksele większe niż próg
            binary = (channel > t).astype(np.uint8) * 255
            if apply_opening:
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            boolean_maps.append((channel_index, t, binary))

            # Mapa inwersji: piksele mniejsze/równe niż próg
            binary_inv = (channel <= t).astype(np.uint8) * 255
            if apply_opening:
                binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
            boolean_maps.append((channel_index, -t, binary_inv))  # próg ujemny tylko informacyjnie

    return boolean_maps


def compute_attention_map(boolean_maps, image_shape):
    """
    Agreguje istotność poprzez zliczanie zamkniętych regionów (niedotykających krawędzi)
    we wszystkich mapach binarnych.
    """
    h, w = image_shape[:2]
    attention_map = np.zeros((h, w), dtype=np.float32)

    for _, _, binary in boolean_maps:
        labeled = measure.label(binary, connectivity=2)
        props = measure.regionprops(labeled)

        for region in props:
            minr, minc, maxr, maxc = region.bbox
            if minr == 0 or minc == 0 or maxr == h or maxc == w:
                continue  # pomijamy komponenty dotykające krawędzi

            for y, x in region.coords:
                attention_map[y, x] += 1

    return attention_map


def post_process(saliency_map):
    """
    Wygładza i oczyszcza mapę istotności.
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(saliency_map, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (81, 81), sigmaX=0)

    enhanced = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    #enhanced = np.power(enhanced / 255.0, 0.9) * 255
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


if __name__ == "__main__":
    image = cv2.imread("../pictures/example2.jpg")
    if image is None:
        print("Nie znaleziono pliku.")
        exit()

    lab = resize_and_convert_to_lab(image)
    L, A, B = cv2.split(lab)

    # cv2.imshow("Kanal L (jasnosc)", L)
    # cv2.imshow("Kanal A (zielony–czerwony)", A)
    # cv2.imshow("Kanal (niebieski–zolty)", B)

    boolean_maps = generate_boolean_maps(lab, thresholds_per_channel=20)

    # for i, (channel_idx, threshold, binary_map) in enumerate(boolean_maps):
    #     channel_name = ["L", "A", "B"][channel_idx]
    #     sign = ">" if threshold > 0 else "<="
    #     window_title = f"Mapa binarna {channel_name} {sign} {abs(int(threshold))}"
    #     cv2.imshow(window_title, binary_map)

    print(f"Wygenerowano {len(boolean_maps)} map binarnych (po {2*4} dla każdego kanału = {len(boolean_maps)} łącznie).")

    attention = compute_attention_map(boolean_maps, lab.shape)
    attention_norm = cv2.normalize(attention, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Mapa uwagi (attention map)", attention_norm)

    post = post_process(attention_norm)
    cv2.imshow("Po postprocessingu", post)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
