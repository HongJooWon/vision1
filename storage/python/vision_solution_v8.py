# -*- coding: utf-8 -*-
# --------------------------------------------
# CREATED AT : 2023-12-15
# CREATED BY : Attic Inc.
# VERSION : 7.0
# --------------------------------------------


import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import copy 
import json 

from imutils import contours
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


from ultralytics import YOLO
from datetime import datetime
import sys
import os
import io

# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

os.environ['KMP_DUPLICATE_LIB_OK']='True'



os.chdir('C:/Users/USER1/Downloads/vision_solution_v8/python')






class NotDetectedError(Exception):
    def __init__(self, msg='No object has been detected'):
        self.msg=msg

    def __str__(self):
        return self.msg


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

        try :
            self.model = YOLO(self.model_source)  # load a pretrained model (recommended for training)
        except FileNotFoundError : 
            print(self.model_source, ' File Not Found.', end=' ')
            exit_program()
    

    def predict_object(self) :

        try : 
            img_files = list_up_files(self.test_image_source)
            img_files = [i for i in img_files if (i[-3:] == 'JPG') or (i[-3:] == 'jpg') or (i[-3:] == 'PNG') or (i[-3:] == 'png')]
            if len(img_files) == 0 : 
                raise FileNotFoundError            

            else : 
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
                                            boxes=True  # or show_boxes
                                            )
        except FileNotFoundError: 
            print(self.test_image_source, ' Image Files Are Not Found : Only JPG or PNG Images are accepted.', end=' ')
            exit_program()
    
        return results 
    
    # Making Bounding Box Prediction Result per Image File Name
    def get_pred_bbox(self) : 

        try : 
            results = self.predict_object()
            num = 0 
            for result in results : 
                num += len(result.boxes) 

            if num < 1 : 
                raise NotDetectedError
            else : 
                fn_list = []
                bbox_list = []
                for i in range(len(results)) : 
                    
                    _, tail = os.path.split(results[i].path)
                    fn_list.append(tail)
                    # fn_list.append(results[i].path[-19:])
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

        except NotDetectedError : 
            print('No Obtect Has been Detedted.')
        print('>>> Objects Are Successfully Detected.')


def exit_program():
    print("Exiting the program.")
    sys.exit(0)


def read_coco_annotation(filename) : 

    try : 
        with open(filename, 'r') as f:
            annotation = json.load(f)

    except FileNotFoundError : 
        print(filename, ' Not Found Error')

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
    target_img_list = [i for i in target_img_list if (i[-3:] == 'JPG') or (i[-3:] == 'PNG') or (i[-3:] == 'jpg') or (i[-3:] == 'png')]

    try : 
        if len(target_img_list) < 1 : 
            raise FileNotFoundError
        else : 

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

    except FileNotFoundError:
        print('Image Files are Not Found at ', image_source) 

    print('>>> GT Bounding Box Successfully Extracted.')

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
    
    print('>>> Floc Size Distribution Successfully Analyzed.')

    return size_distribution



def plot_hist_and_save(size_dist_dic, save_path = './') : 

    for f in size_dist_dic.keys() : 
        plot_hist_and_save_each_img(size_dist_dic[f], save_path + 'size_dist_' + f[:-3])

    print('>>> Floc Size Histogram has been saved successfully.')    



def plot_hist_and_save_each_img(size_list, filename = 'hist.jpg') : 

    plt.figure(figsize=(4, 3))
    plt.hist(size_list, bins=50)
    plt.ylim([0, 50])
    plt.grid()

    if True : 
        try : 
            plt.savefig(filename)
            # print('>>> ', filename, 'Has been saved successfully.')
        except FileNotFoundError : 
            print(filename, 'Output Folder Does Not Exist.')

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



def save_dist_csv(size_dist, file_name) :

    val = [size_dist[i] for i in size_dist.keys()]
    for i in range(len(val)-1) : 
        if i == 0 : 
            df = pd.concat([pd.DataFrame(val[i]), pd.DataFrame(val[i+1])], axis=1)
        else : 
            df = pd.concat([df, pd.DataFrame(val[i+1])], axis=1)
    df.columns = pred_size_dist.keys()

    try : 
        df.to_csv(file_name, index=False)
    except FileNotFoundError : 
        print(file_name, 'Folder Does Not exist.')

    _, fn_only = os.path.split(file_name)
    print('>>>', fn_only, 'Saved Successfully.')



def plot_bbox_and_save(image_source, save_path, pred_bbox_dict, bbox_color=(255, 0, 0)) : 

    file_list = list_up_files(image_source)
    file_list = [i for i in file_list if (i[-3:] == 'JPG') or (i[-3:] == 'PNG') or (i[-3:] == 'jpg') or (i[-3:] == 'png')]

    try : 
        if len(file_list) < 1 : 
            raise FileNotFoundError

        else : 

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
                try : 
                    cv2.imwrite(save_path + f, target_img)
                    # print('>>> ', save_path + f, 'has been saved successfully.')
                except FileNotFoundError: 
                    print(save_path, 'Folder Does Not Exist.')

    except FileNotFoundError:
        print('Image Files are Not Found at ', image_source) 

    print('>>> Plotted Bounding Box Images Saved Successfully.')        



# ------------------------------------------------------
# Sentimentation Aanlsysis 
# ------------------------------------------------------

def get_batch_images(batch_path='./') :

    batch_files = list_up_files(batch_path)
    batch_files = [i for i in batch_files if (i[-3:] == 'JPG') or (i[-3:] == 'PNG') or (i[-3:] == 'jpg') or (i[-3:] == 'png')]

    try : 
        if len(batch_files) < 1 : 
            raise FileNotFoundError
        else : 
            img_list = []
            for f in batch_files : 
                img = cv2.imread(batch_path + f)
                img_list.append(img)


    except FileNotFoundError:
        print('Image Files are Not Found at ', batch_path) 

    batch_img_dict = dict(zip(batch_files, img_list))
    # Sort the Dictionary by Keys
    batch_img_dict = dict(sorted(batch_img_dict.items()))

    print('>>> Batch Images Are Loaded Successfully.')

    return batch_img_dict


def crop_bbox(img, top_left_x, top_left_y, bottom_right_x, bottom_right_y) : 

    _img = copy.deepcopy(img)
    return _img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


def calc_surface(batch_images) : 

    IsBlack = True # 결정을 흰색으로 할지, 검정색으로 할지. 
    # Set Criteria for surface
    CRITERIA = 200 # Pixel 밝기 
    # CRITERIA 200으로 하면 10초 이내에서는 표면을 잘 찾지만, 40초 이상 지나면 실제보다 한참 위에 있음
    # CRITERIA 50으로 하면 앞쪽에서는표면을 잘 못 찾지만, 40초 이상 지나면 잘 찾음

    FLOC_RATIO_MIN = 0.5
    FLOC_RATIO_MAX = 0.6
    # COUNT 기준 : CRITERIA = 200 일 때,  0.001 이상이면 floc surface = cropped_list의 인덱스 + 1
    # floc의 높이 : floc surface(=index of cropped_list + 1)

    MOVING_WINDOW_X = 780
    MOVING_WINDOW_Y = 30
    STEP_SIZE = 1 # moving window가 Y 축 방향으로 몇개씩 움직일지.....

    floc_height = []
    for f in batch_images.keys() : 

        img = batch_images[f]
        size_y = img.shape[0] # HEIGHT 1350
        size_x = img.shape[1] # WIDHT 780

        # print(f, end=' >>> ')
        # --------------------------------------------------
        # Making a List of Moving Windows
        # --------------------------------------------------
        wbox_list = [ [0, i, size_x, i+MOVING_WINDOW_Y] for i in range(0, (size_y - MOVING_WINDOW_Y), STEP_SIZE)]

        cropped_list = []
        for bbox_ind in range(len(wbox_list)) : 

            # print(wbox_list[bbox_ind][0], wbox_list[bbox_ind][1], wbox_list[bbox_ind][2], wbox_list[bbox_ind][3])
            cropped_img = crop_bbox(img, wbox_list[bbox_ind][0], wbox_list[bbox_ind][1], wbox_list[bbox_ind][2], wbox_list[bbox_ind][3])
            cropped_list.append(cropped_img)

            if False : 
                cv2.imwrite(SAVE_FILE_NAME+str(bbox_ind)+'.jpg', cropped_img)    

        # --------------------------------------------------
        # Calculate The surface of Flocs
        # --------------------------------------------------
        for i in range(len(cropped_list)) : 
            
            hist,bins = np.histogram(cropped_list[i].ravel(),256,[0,256]) 
            
            if IsBlack : 
                count = [ hist[i] for i in range(0, CRITERIA+1)] # Black 기준으로 했을 때 
            else : 
                count = [ hist[i] for i in range(CRITERIA, 255)] # White 기준으로 했을 때 

            count = np.sum(count)
            floc_ratio = count / (MOVING_WINDOW_X * MOVING_WINDOW_Y * 3)


            if (floc_ratio > FLOC_RATIO_MIN) & (floc_ratio < FLOC_RATIO_MAX) : 

                a_floc_height = i+1
                floc_height.append(a_floc_height)
                break

    floc_height_dict = dict(zip(batch_images.keys(), floc_height))

    print('>>> Floc Surface Calculated Successfully.')
    return floc_height_dict


def plot_surface_and_save(img_dict, floc_height_dict, save_path): 


    # Make Direcotory If Not Exist
    try : 
        os.makedirs(save_path)
    except FileExistsError: 
        print(save_path, 'Already Exist.')

    # Calculate the Floc Surface
    moving_window_x = 780
    for f in floc_height_dict.keys() : 

        a_floc_height = floc_height_dict[f]
        img = img_dict[f]

        start_p = (0, a_floc_height)
        end_p = (moving_window_x-1, a_floc_height)

        # Plot the Surface Line
        plotted_img = plot_a_horizontal_line(img, start_p, end_p, line_color=(255, 0, 0))

        # Save the Surface Plotted Image
        cv2.imwrite(save_path + f[:-4]+ '_' + str(a_floc_height) + '.jpg', plotted_img)
    
    print('>>> Surface Plotted Images Are Saved Successfully.')


def plot_a_horizontal_line(img, start_point, end_point, line_color=(0, 255, 0), line_thinkness=3) :
    # Created at 2023-1016
    # img : Should be (H, W, C)

    _img = np.copy(img)
    if False : 
        print(_img.shape) # img size(H,W,C):  (1350, 780, 3) 

    
    cv2.line(_img, 
             start_point, 
             end_point, 
             line_color, 
             line_thinkness)     

    return _img


def save_plot_surface_as_csv(floc_height_dict, save_path): 

    # Make Direcotory If Not Exist
    try : 
        os.makedirs(save_path)
    except FileExistsError: 
        print(save_path, 'Already Exist.')

    batch_name = os.path.basename(os.path.dirname(save_path))
    heights = list(floc_height_dict.values())
    secs = np.arange(1, len(heights)+1)
    floc_height_df = pd.DataFrame(np.array([secs, heights]).T, columns=['secs', 'height'])
    floc_height_df.to_csv(save_path + 'floc_height_' + batch_name + '.csv' ,index=False)

    print('>>> Surface Heights Are Saved(CSV) Successfully.')

    return floc_height_df








if __name__ == '__main__': 

    # # --------------------------------------
    # # Folder Description 
    # # --------------------------------------
    # # Input 
    # # 1. ORIGINAL IMAGES : ../yolo_data/test/images/
    # # 2. Annotation : ../yolo_data/test/annotations/test.json
    # # 3. Pretrained Model : ./yolo8_x_model_epoch_30_best.pt

    # # Output
    # # 4. Bouding Box plotted Images of Ground Truth : ../output/bbox_plotted/gt/
    # # 5. Bouding Box plotted Images of predicted flocs: ../output/bbox_plotted/pred/
    # # 6. Histogram Images of Ground Truth : ../output/hist/gt/
    # # 7. Histogram Images of Predicted : ../output/hist/pred/
    # # 8. Floc Size Distribution CSV File : ../output/predicted_size_distribution.csv

    # BasePath = sys.argv[1]
    # FolderPath = sys.argv[2]
    # ProjectName = sys.argv[3]
    BasePath = 'D:/회사/vision2/storage'
    # ProjectName = 'proj_20231222'

    # # --------------------------------
    # # Floc Preiction 
    # # --------------------------------
    # PRETRAINED_MODEL_SOURCE = BasePath + '/python/yolo8_x_model_epoch_30_best.pt' # m model 'epoch_20_best.pt'
    # TEST_SOURCE = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    # detection = Attic_Detection(PRETRAINED_MODEL_SOURCE, TEST_SOURCE)
    # detection.get_pred_bbox()

    # # --------------------------------
    # # Get GT Annotations 
    # # --------------------------------
    # image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    # gt_source = BasePath + '/yolo_data/test/annotations/test.json'
    # gt_bbox = get_gt_bbox(image_source, gt_source)
    # # print(gt_bbox.keys())
    # # print(gt_bbox['cropped_F8A3996.JPG'])


    # # --------------------------------
    # # Floc Size Distribution Calculation
    # # --------------------------------
    # pred_bbox_dict = detection.pred_bbox
    # gt_size_dist = object_size_distribution(gt_bbox)
    # pred_size_dist = object_size_distribution(pred_bbox_dict)

    # # print(len(pred_size_dist['cropped_F8A3996.JPG']))
    # # print(len(gt_size_dist['cropped_F8A3996.JPG']))
    # # print(pred_size_dist['cropped_F8A3996.JPG'])
    # # print(gt_size_dist['cropped_F8A3996.JPG'])



    # # --------------------------------
    # # Plot and Save Size Distribution 
    # # --------------------------------
    # # Save gt floc size distribution 
    # output_folder_gt = BasePath + '/output/hist/gt/' + ProjectName + '/'
    # plot_hist_and_save(gt_size_dist, output_folder_gt)

    # # Save predicted floc size distribution 
    # output_folder_predicted = BasePath + '/output/hist/pred/' + ProjectName + '/'
    # plot_hist_and_save(pred_size_dist, output_folder_predicted)


    # # -----------------------------------------------
    # # Save Predicted Size Distribution as a csv file 
    # # -----------------------------------------------
    # pred_size_distribution_file_name = BasePath + '/output/report/' + ProjectName + '.csv'
    # save_dist_csv(pred_size_dist, pred_size_distribution_file_name)


    # # --------------------------------
    # # Plot BBox and Save Images
    # # --------------------------------
    # image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    # save_path = BasePath + '/output/bbox_plotted/pred/' + ProjectName + '/'
    # pred_bbox_dict = detection.pred_bbox
    # plot_bbox_and_save(image_source, save_path, pred_bbox_dict)

    # image_source = BasePath + '/yolo_data/test/images/' + ProjectName + '/'
    # save_path = BasePath + '/output/bbox_plotted/gt/' + ProjectName + '/'
    # gt_source = BasePath + '/yolo_data/test/annotations/test.json'
    # gt_bbox = get_gt_bbox(image_source, gt_source)
    # plot_bbox_and_save(image_source, save_path, gt_bbox, (0, 0, 255))
    # print('The End')



    # -----------------------------------------------------
    # Folder Description For Sentimentation Speed Analysis
    # -----------------------------------------------------
    # Input 
    # 1. BATCH_SOURCE : '../input/batch_0901_303/' # input 폴더의 하위 폴더가 배치ID

    # Output
    # 3. Floc Surface Image : ../output/sentimentation/images/batch_0901_303/
    # 4. Floc Surface CSV : ../output/sentimentation/csv/batch_0901_303/floc_height_batch_0901_303.csv
    


    print('Sentimentation Speed Analysis.......')
    # --------------------------------
    # Sentimentation Speed Analysis
    # --------------------------------
    # 1. Get Source Image list
    BATCH_SOURCE = BasePath +'/input/batch_0831_003/'
    BATCH_NAME = os.path.basename(os.path.dirname(BATCH_SOURCE))
    batch_images = get_batch_images(BATCH_SOURCE)


    # 2. Decide the height of images
    floc_height_dict = calc_surface(batch_images)

    # 3. Plot the Surface and Save the Image 
    save_path = BasePath +'/output/sentimentation/images/' + BATCH_NAME + '/'
    plot_surface_and_save(batch_images, floc_height_dict, save_path)

    # 4. Save the Floc Surface as a CSV File
    save_path = BasePath +'/output/sentimentation/csv/' + BATCH_NAME + '/'
    floc_height_df = save_plot_surface_as_csv(floc_height_dict, save_path)

    # 5. Print Sentimentation Speed
    sentimentation_speed = (floc_height_df.loc[35, 'height'] - floc_height_df.loc[0, 'height']) / 35
    print('배치ID : {:}, 침강속도 : {:} Pixel/Seconds'.format(BATCH_NAME, sentimentation_speed))
