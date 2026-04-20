def iou(box_a, box_b):
    # Tọa độ vùng giao nhau
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])    # Sửa: max của 2 cạnh trên
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3]) # Sửa: min của 2 cạnh dưới
    
    # Tính diện tích giao nhau (đảm bảo không âm nếu boxes không chạm nhau)
    inter_w = max(0, x_right - x_left)
    inter_h = max(0, y_bottom - y_top)
    intersection_area = inter_w * inter_h

    # Tính diện tích mỗi box
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Tính Union
    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area if union_area > 0 else 0