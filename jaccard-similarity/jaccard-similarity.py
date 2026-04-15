def jaccard_similarity(set_a, set_b):
    # 1. Loại bỏ phần tử trùng lặp trong chính mỗi mảng (Unique items)
    unique_a = []
    for x in set_a:
        if x not in unique_a:
            unique_a.append(x)
            
    unique_b = []
    for x in set_b:
        if x not in unique_b:
            unique_b.append(x)
    
    # 2. Tìm phần giao từ hai mảng đã làm sạch
    intersection = []
    for item in unique_a:
        if item in unique_b:
            intersection.append(item)

    numerator = len(intersection)
    
    # 3. Tính mẫu số theo công thức: |A| + |B| - |A giao B|
    denom = len(unique_a) + len(unique_b) - numerator 

    if denom == 0:
        return 0.0

    return numerator / denom