import cv2
import math
import numpy as np
from scipy import stats

def calculate_ttest_pvalues(df, group_col='Group', condition_col='Condition', value_col='Value'):
    """
    计算各组间的t检验p值
    
    参数:
        df: 包含数据的DataFrame
        group_col: 分组列名 (如'H3K27ac'和'CTCF')
        condition_col: 条件列名 (如'CTRL'和'VPA')
        value_col: 数值列名
    
    返回:
        包含p值的DataFrame
    """
    # 获取唯一的分组和条件
    groups = df[group_col].unique()
    conditions = df[condition_col].unique()
    
    # 存储结果的列表
    results = []
    
    # 对每个分组进行组内条件比较
    for group in groups:
        group_data = df[df[group_col] == group]
        
        # 提取各条件的数据
        data = [group_data[group_data[condition_col] == cond][value_col] 
                for cond in conditions]
        
        # 执行独立样本t检验
        t_stat, p_val = stats.ttest_ind(*data)
        
        # 将结果添加到列表
        results.append({
            'Group': group,
            'Comparison': f"{conditions[0]} vs {conditions[1]}",
            't-statistic': t_stat,
            'p-value': p_val
        })
    
    # 转换为DataFrame
    return pd.DataFrame(results)
    
def ellipse(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(big_contour)
    (xc,yc),(d1,d2),angle = ellipse
    result = img.copy()
    cv2.ellipse(result, ellipse, (0, 255, 0), 3)
    xc, yc = ellipse[0]
    cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
    return result, angle

def rectangle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def rotate(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotated = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, rotated, (w, h))
    return rotated

def ZeroPaddingResizeCV(img, size=(600, 600), interpolation=None, n=3):
    isize = img.shape
    ih, iw = isize[0], isize[1]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)
 
    img = cv2.resize(img, (new_w, new_h), interpolation)
    if n==3:
        new_img = np.zeros((h, w, n), np.uint8)
        new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img
    else:
        new_img = np.zeros((h, w), np.float32)
        new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img
    return new_img


def nucleus_intensity_distribution(thresh, img):
    mask = thresh.copy()
    feature_intensity = []
    # nucleus whole intensity
    img_with_mask = mask * img
    non_zero_values = img_with_mask[np.nonzero(img_with_mask)]
    average = np.mean(non_zero_values)
    feature_intensity.append(average)

    # nucleus part intensity
    h, w = thresh.shape[:2]
    num_parts = 5
    for i in range(num_parts, 0, -1):
        scale = i/5
        new_h, new_w = int(h * scale), int(w * scale)
        d_h, d_w = (h - new_h) // 2, (w - new_w) // 2
        resized = cv2.resize(thresh, (new_w, new_h))
        copy = np.zeros_like(thresh)
        copy[d_h:d_h + new_h, d_w:d_w + new_w] = resized*2
        copy[copy == 0] = 1
        mask *= copy
    for i in range(1,num_parts+1):
        mask[mask==2**i]=(255*i/num_parts)
    part_intensity=[]
    for i in range(1,num_parts+1):
        color = int(255*i/num_parts)
        mask_part = cv2.inRange(mask, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))/255
        img_part = img * mask_part
        non_zero_values = img_part[np.nonzero(img_part)]
        average = np.mean(non_zero_values)
        part_intensity.append(average)
    feature_intensity.extend(part_intensity)

    # nucleus part intensity distribution
    total_intensity = sum(part_intensity)
    part_intensity_distribution = [x / total_intensity for x in part_intensity]
    feature_intensity.extend(part_intensity_distribution)
    
    return feature_intensity