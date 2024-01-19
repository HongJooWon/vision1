# -*- coding: utf-8 -*-
# --------------------------------------------
# CREATED AT : 2023-12-13
# CREATED BY : Attic Inc.
# VERSION : 6.2
# --------------------------------------------


import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import copy 
import json 
import sys

from imutils import contours
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


from ultralytics import YOLO
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



class Attic_Detection() : 
    
    model = None
    pred_bbox = None
    
    def __init__(self, model_source, test_image_source) : 
        self.model_source = model_source
        self.test_image_source = test_image_source
        self.yalm_file = 'yolov8l.yaml'

        self.load_trained_model()

    
    def load_trained_model(self) : 
    
        self.model = YOLO(self.yalm_file)  # build a new model from YAML
        self.model = YOLO(self.model_source)  # load a pretrained model (recommended for training)

    
    
    def predict_object(self) :  
        results = self.model.predict(self.test_image_source,
                                    conf=0.25, 
                                    imgsz=640, 
                                    iou=0.7, 
                                    show=False, 
                                    save=True,
                                    save_txt=True,  # bbox in text format
                                    save_conf=True,
                                    save_crop=False, 
                                    show_labels=False,
                                    show_conf=False,
                                    line_width=1,
                                    agnostic_nms=True,
                                    retina_masks=True,
                                    boxes=True
                                    )
    
        return results 
    
    # Making Bounding Box Prediction Result per Image File Name
    def get_pred_bbox(self) : 
        
        results = self.predict_object()

        fn_list = []
        bbox_list = []
        for i in range(len(results)) : 
            fn_list.append(results[i].path[-19:])
            tmp_bbox = results[i].boxes 

            a_box_list = []
            for j in range(len(tmp_bbox)) : 
                box = tmp_bbox[j]
                box = box.xyxy.cpu().detach().numpy()[0]
                box = [int(k) for k in box]

                # -----------------------------------
                # Convert to coco format (x, y, w, h)
                # -----------------------------------
                x1, y1, x2, y2 = box 
                w = x2- x1
                h = y2 - y1
                box = [x1, y1, w, h]
                a_box_list.append(box)
            bbox_list.append(a_box_list)
            
        self.pred_bbox = dict(zip(fn_list, bbox_list))



def read_coco_annotation(filename) : 

    with open(filename, 'r') as f:
        annotation = json.load(f)

    if False : 
        print(annotation.keys())
        print(len(annotation['images']), len(annotation['annotations']), len(annotation['categories']), len(annotation['info']), )
        print(annotation['categories'])
        print(annotation['info']) # 의미없음 
        print(annotation['images'][0])
        print(annotation['annotations'][0])
        print(annotation['annotations'][-1])

    return annotation 


def list_up_files(dir_path) : 

    filelist = []
    for file_path in os.listdir(dir_path): 
        if os.path.isfile(os.path.join(dir_path, file_path)):
            # add filename to list
            filelist.append(file_path)

    filelist.sort()

    return filelist



def get_gt_bbox(image_source, gt_source) : 

    target_img_list = list_up_files(image_source)
    target_img_list = [i for i in target_img_list if i[-3:] == 'JPG']
    annot = read_coco_annotation(gt_source)

    # 4. GT에서 이미지 파일별로 gt box 만들기 
    # 4.1 GT에서 이미지 파일 이름 추출 
    image_fname_list = []
    image_id_list = []
    for i in annot['images'] : 
        image_fname_list.append(i['file_name'])
        image_id_list.append(i['id'])
    img_fname_dict = dict(zip(image_id_list, image_fname_list))

    # 4.2 이미지 파일별로 GT BBox 추출 
    bbox_list = []
    filename_list = []
    for target_id in img_fname_dict.keys():
        target_bbox_list = []
        for i in annot['annotations'] : 
            if i['image_id'] == target_id : 
                rounded_bbox = [np.round(v, 2) for v in i['bbox']]
                target_bbox_list.append(rounded_bbox)
        bbox_list.append(target_bbox_list)
        filename_list.append(img_fname_dict[target_id])

    gt_bbox = dict(zip(filename_list, bbox_list))
 
    return gt_bbox 


def object_size_distribution(bbox_dict): 

    size_dist = []
    for filename in bbox_dict.keys() : 
        tmp_box_list = bbox_dict[filename]
        size_list=[]
        for box in tmp_box_list : 
            _, _, w, h = box 
            if w >= h : 
                size = w 
            else : 
                size = h 
            size_list.append(size)
        size_dist.append(size_list)

    size_distribution = dict(zip(bbox_dict.keys(), size_dist))

    return size_distribution


def plot_hist_and_save(size_list, filename = './hist.jpg') : 

    plt.figure(figsize=(4, 3))
    plt.hist(size_list, bins=50)
    plt.ylim([0, 50])
    plt.grid()

    if True : 
        plt.savefig(filename)
        plt.close()
    elif False : 
        plt.show()




def make_coco_predition_annotation(gt_annot, pred_result_dict) : 

    info = {'contributor': '', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''}
    licenses = [{'name': '', 'id': 0, 'url': ''}]
    categories = [{'id': 1, 'name': 'floc', 'supercategory': ''}]

    #--------------------------------------------------------------
    # 1. Make Image Annotation
    #--------------------------------------------------------------
    #-----------------------------------------------
    # To Extract image id : soruce gt annotation
    # ==> GT 에서 image_id 추출 
    # ==> tmp_id_list : GT에 있는 이미지 아이디 리스트. cocoeval 할 때 사용. 
    #-----------------------------------------------
    tmp_id_list = [i['id'] for i in gt_annot['images']]
    tmp_fn_list = [i['file_name'] for i in gt_annot['images']]

    annot_image_list = []
    for fn in pred_result_dict.keys() : 

        tmpind = tmp_fn_list.index(fn)
        image_id = tmp_id_list[tmpind]
        annot_image = {
            'id' : image_id, 
            'width' : 780, 
            'height' : 1350, 
            'file_name' : fn, 
            'license' : licenses[0]['id'], 
            'flickr_url' : '', 
            'coco_url' : '', 
            'date_captured' : ''
        }
        annot_image_list.append(annot_image)

    #--------------------------------------------------------------
    # 2. Make Annotations ==> Prediction은 List로 만들어야 됨. 
    # [ {
    # "image_id": 1,
    # "category_id": 0,
    # "bbox": [
    # 466.484375,
    # 76.4969711303711,
    # 224.9879150390625,
    # 94.5906753540039
    # ],
    # "score": 0.9025385975837708,
    # "segmentation": []
    # }
    # .......
    # ]
    #--------------------------------------------------------------

    is_crowd = 0
    annotation_id = 1
    annot_bbox_list = []
    category_id = 1 # Just Because a Signle Category. 
    for fn in pred_result_dict.keys() : 
        tmp_bbox_list = pred_result_dict[fn]

        tmpind = tmp_fn_list.index(fn)
        image_id = tmp_id_list[tmpind]
        for bb in tmp_bbox_list : 
            single_annot = {
                'id' : annotation_id, 
                'image_id' : image_id, 
                'category_id' : category_id,
                'segmentation' : [], 
                'area' : bb[2]*bb[3], # 원래는 segmentation으로 계산이 되어야 함. 
                'bbox' : bb, 
                'score' : 0.950 # single class라서 0.95
            }
            annotation_id += 1
            annot_bbox_list.append(single_annot)
    #--------------------------------------------------------------
    # 3. Combine all annotations into a signle variable
    #--------------------------------------------------------------
    if False : # GT와 동일한 포맷으로 만들 때...
        pred_rst_annot = {
            'licenses' : licenses, 
            'info' : info, 
            'categories' : categories, 
            'images' : annot_image_list, 
            'annotations' : annot_bbox_list,
        }
        print('Annotation length', len(pred_rst_annot['annotations']))
    else : # COCO Predciton format으로 만들 때 
        pred_rst_annot = annot_bbox_list.copy()

    return pred_rst_annot


def calculate_map(gt_source, prediction_annotation) : 
  
    coco_gt = COCO(gt_source)
    coco_pred = coco_gt.loadRes(prediction_annotation)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



def plot_bbox_and_save(image_source, save_path, pred_bbox_dict, bbox_color=(255, 0, 0)) : 

    file_list = list_up_files(image_source)

    for f in file_list : 
        #-----------------------------------------    
        # Draw BBox 
        #----------------------------------------- 
        selected_filename = image_source + f
        target_img = cv2.imread(selected_filename)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)


        selected_bbox = pred_bbox_dict[f]
        # Change format from coco to pascal if necessary
        selected_bbox = [ [int(x), int(y), int(x+w), int(y+h)]  for x, y, w, h in selected_bbox]
        for abox in selected_bbox : 
            xmin, ymin, xmax, ymax = abox
            cv2.rectangle(target_img, (xmin, ymin), (xmax, ymax), bbox_color, 2)

        #-----------------------------------------    
        # Write Image 
        #-----------------------------------------
        cv2.imwrite( save_path + f, target_img)



if __name__ == '__main__': 

    # --------------------------------------
    # Folder Description 
    # --------------------------------------
    # Input 
    # 1. ORIGINAL IMAGES : ../yolo_data/test/images/
    # 2. Annotation : ../yolo_data/test/annotations/test.json
    # 3. Pretrained Model : ./yolo8_x_model_epoch_30_best.pt

    # Output
    # 4. Bouding Box plotted Images of Ground Truth : ../output/bbox_plotted/gt/
    # 5. Bouding Box plotted Images of predicted flocs: ../output/bbox_plotted/pred/
    # 6. Histogram Images of Ground Truth : ../output/hist/gt/
    # 7. Histogram Images of Predicted : ../output/hist/pred/
    # 8. Floc Size Distribution CSV File : ../output/predicted_size_distribution.csv

    # --------------------------------
    # Input 
    # --------------------------------
    BasePath = sys.argv[1]
    FolderPath = sys.argv[2]
    ProjectName = sys.argv[3]

    # BasePath = 'D:/회사/vision2/storage'
    # ProjectName = '112233'
    # --------------------------------
    # Floc Preiction 
    # --------------------------------
    PRETRAINED_MODEL_SOURCE =  BasePath + '/python/yolo8_x_model_epoch_30_best.pt' # m model 'epoch_20_best.pt'
    # TEST_SOURCE = '../yolo_data/test/images/'
    TEST_SOURCE = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    detection = Attic_Detection(PRETRAINED_MODEL_SOURCE, TEST_SOURCE)
    detection.get_pred_bbox()
    

    # --------------------------------
    # Get GT Annotations 
    # --------------------------------
    # image_source = '../yolo_data/test/images/'
    image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    gt_source = BasePath + '/yolo_data/test/annotations/test.json'
    gt_bbox = get_gt_bbox(image_source, gt_source)
    # print(gt_bbox.keys())
    # print(gt_bbox['cropped_F8A3996.JPG'])


    # --------------------------------
    # Floc Size Distribution Calculation
    # --------------------------------
    pred_bbox_dict = detection.pred_bbox
    gt_size_dist = object_size_distribution(gt_bbox)
    pred_size_dist = object_size_distribution(pred_bbox_dict)

    # print(len(pred_size_dist['cropped_F8A3996.JPG']))
    # print(len(gt_size_dist['cropped_F8A3996.JPG']))
    # print(pred_size_dist['cropped_F8A3996.JPG'])
    # print(gt_size_dist['cropped_F8A3996.JPG'])


    # --------------------------------
    # Plot and Save Size Distribution 
    # --------------------------------
    # Save gt floc size distribution 
    output_folder_gt = BasePath + '/output/hist/gt/' + ProjectName + '/'
    for f in gt_size_dist.keys() : 
        plot_hist_and_save(gt_size_dist[f], output_folder_gt + 'size_gt_' + f[:-3])

    # Save predicted floc size distribution 
    output_folder_predicted = BasePath + '/output/hist/pred/' + ProjectName + '/'
    for f in pred_size_dist.keys() : 
        plot_hist_and_save(pred_size_dist[f], output_folder_predicted + 'size_pred_' + f[:-3])

    # -----------------------------------------------
    # Save Predicted Size Distribution as a csv file 
    # -----------------------------------------------
    pred_size_distribution_file_name = BasePath + '/output/report/' + ProjectName + '.csv'

    val = [pred_size_dist[i] for i in pred_size_dist.keys()]
    for i in range(len(val)-1) : 
        if i == 0 : 
            df = pd.concat([pd.DataFrame(val[i]), pd.DataFrame(val[i+1])], axis=1)
        else : 
            df = pd.concat([df, pd.DataFrame(val[i+1])], axis=1)
    df.columns = pred_size_dist.keys()
    df.to_csv(pred_size_distribution_file_name, index=False)



    # --------------------------------
    # Plot BBox and Save Images
    # --------------------------------
    # image_source = '../yolo_data/test/images/'
    image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    save_path = BasePath + '/output/bbox_plotted/pred/' + ProjectName + '/'
    pred_bbox_dict = detection.pred_bbox
    plot_bbox_and_save(image_source, save_path, pred_bbox_dict)


    # image_source = '../yolo_data/test/images/'
    image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    save_path = BasePath + '/output/bbox_plotted/gt/' + ProjectName + '/'
    gt_source = BasePath + '/yolo_data/test/annotations/test.json'
    gt_bbox = get_gt_bbox(image_source, gt_source)
    plot_bbox_and_save(image_source, save_path, gt_bbox, (0, 0, 255))

    print('The End')
    # --------------------------------
    # Calc mAP  
    # --------------------------------

    if False : 
        gt_source = '/mnt/d/tr_data/garam_coco/final_dataset/test/annotations/test.json'
        gt_annot = read_coco_annotation(gt_source)

        pred_bbox_dict = detection.pred_bbox
        pred_annotation = make_coco_predition_annotation(gt_annot, pred_bbox_dict)
        calculate_map(gt_source, pred_annotation)
