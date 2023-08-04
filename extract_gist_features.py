from concurrent.futures import ThreadPoolExecutor
from GIST import GIST
import numpy as np
import os
import pickle
import cv2

param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension":32
}
gist = GIST(param) 
DATA = '15-Scene'
cnt = -1

features = []
label = []

def process_image(file_path, cnt):
    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(100,100))
    feature = gist._gist_extract(img)
    return feature, cnt

with ThreadPoolExecutor() as executor:
    futures = []
    for c in os.listdir(DATA):
        cnt += 1
        for file_name in os.listdir(os.path.join(DATA, c)):
            file_path = os.path.join(DATA, c, file_name)
            futures.append(executor.submit(process_image, file_path, cnt))
    
    for future in futures:
        feature, cnt = future.result()
        features.append(feature)
        label.append(cnt)

print(np.array(features).shape)
print(np.array(label).shape)

pickle.dump(features, open('features.dump', 'wb'))
pickle.dump(label, open('labels.dump', 'wb'))