import cv2
import numpy as np

def visualize_one_query(query_path, gallery_paths, distmat_line_result, gt_paths, distmat_line_gt, y_true):
    """
        query_path: file path of query
        gallery_paths: file paths of galleries
        distmat_line_result: distance info for each gallery, length should be equalt to gallery_paths

        gt_paths: file paths of the ground truth galleries for current query
        distmat_line_gt: distance info for each gt gallery, length should be equalt to gt_paths
        y_true: whether it's matched for each gallery in gallery_paths
    """
    query_image = cv2.imread(query_path)
    thumbnail_img = __build_img_list_thumbnail(gallery_paths, distmat_line_result, y_true)
    
    thumbnail_img_gt = __build_img_list_thumbnail(gt_paths, distmat_line_gt)
    empty_block = np.zeros((200, 100, 3)).astype(np.uint8())
    for i in range(0, len(gallery_paths) - len(gt_paths)):
        thumbnail_img_gt = np.concatenate((thumbnail_img_gt, empty_block), axis=1) if thumbnail_img_gt is not None else empty_block

    # concat into one line
    resized_query_image = cv2.resize(query_image, (100, 200))
    empty_split_bar = np.zeros((200, 100, 3)).astype(np.uint8())
    query_and_answer = np.concatenate((resized_query_image, empty_split_bar, thumbnail_img), axis=1)

    # concat gt into the second line
    resized_query_image = cv2.resize(query_image, (100, 200))
    empty_split_bar = np.zeros((200, 100, 3)).astype(np.uint8())
    query_and_gt = np.concatenate((resized_query_image, empty_split_bar, thumbnail_img_gt), axis=1)

    query_and_answer_and_gt = np.concatenate((query_and_answer, query_and_gt), axis=0)

    return query_and_answer_and_gt

# refer to  id_statistics
def __build_img_list_thumbnail(image_path_list, title_list=None, y_true=None):

    if title_list is not None:
        assert len(image_path_list) == len(title_list), "The length of image_path_list and title_list are not same.."
    if y_true is not None:
        assert len(image_path_list) == len(y_true), "The length of image_path_list and y_true are not same.."

    thumbnail_size = 100, 200
    thumbnail_empty = np.zeros((200, 100, 3)).astype(np.uint8())

    img_count = len(image_path_list)
    row = 1
    column = img_count

    i = 0
    thumbnail_gallery = None
    thumbnail_line = None
    for (index, image_path) in enumerate(image_path_list):
        img_origin = cv2.imread(image_path)
        if img_origin is None:
            print(f"img is None: {image_path}")
            thumbnail_replace = (np.ones((200, 100, 3)) * 100).astype(np.uint8())
            img_origin = thumbnail_replace
        img = cv2.resize(img_origin, thumbnail_size)

        # add score
        if title_list is not None:
            text = str(round(title_list[index], 2))
            cv2.putText(img, text, (15, 30), 0, 1, (0, 0, 255), 3)
        # add y_true
        if y_true is not None:
            color = (0, 255, 0) if y_true[index] else (0, 0, 255)
            cv2.rectangle(img, (0, 0), thumbnail_size, color, thickness=3)

        if i < column:
            i += 1
            thumbnail_line = np.concatenate((thumbnail_line, img), axis=1) if thumbnail_line is not None else img
        else:
            thumbnail_gallery = np.concatenate((thumbnail_gallery, thumbnail_line),
                                                axis=0) if thumbnail_gallery is not None else thumbnail_line
            thumbnail_line = img
            i = 1

    # pad the line
    if thumbnail_gallery is not None:
        if thumbnail_line.shape[1] != thumbnail_gallery.shape[1]:
            while thumbnail_line.shape[1] < thumbnail_gallery.shape[1]:
                thumbnail_line = np.concatenate((thumbnail_line, thumbnail_empty), axis=1)

    thumbnail_gallery = np.concatenate((thumbnail_gallery, thumbnail_line),
                                        axis=0) if thumbnail_gallery is not None else thumbnail_line

    return thumbnail_gallery
