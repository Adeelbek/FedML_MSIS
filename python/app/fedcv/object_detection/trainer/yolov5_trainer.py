import copy
import logging
import math
import random
import time
import os
import datetime

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.general import (LOGGER, Profile, check_amp,
                                        check_dataset, check_file,
                                        check_img_size, check_yaml, colorstr,
                                        non_max_suppression, scale_coords,
                                        xywh2xyxy)

from model.yolov5.utils.loggers import Loggers
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                        yolov5_ap_per_class)

import fedml
from fedml.core import ClientTrainer
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.general import Profile, non_max_suppression, xywh2xyxy, scale_coords
from model.yolov5.utils.metrics import ConfusionMatrix, yolov5_ap_per_class, ap_per_class, box_iou
from fedml.core.mlops.mlops_profiler_event import  MLOpsProfilerEvent
from model.yolov5 import val as validate # imported to use original yolov5 validation function!!!

from model.yolov5.utils.loggers import Loggers
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                        yolov5_ap_per_class)

use_shoaib_code=True
if use_shoaib_code:      

    import sys
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/')
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/yolov5')
    sys.path.append('Yolov5_DeepSORT_PseudoLabels/deep_sort')
    
    import shutil
    
    from model.yolov5 import val_pseudos as pseudos  # imported to use modified yolov5 validation function!!!
    from Yolov5_DeepSORT_PseudoLabels import trackv2_from_file as recover
    from Yolov5_DeepSORT_PseudoLabels.merge_forward_backward_v2 import merge 

    from data.data_loader import create_dataloader
    import yaml
    
    def use_new_data(args,model,compute_loss,train_data,test_data):
        
        
        # -------------------------------- Shoaib Code --------------------------------------- #
        args.curr_step=0
        args.old_train_data = train_data
        args.old_test_data  = test_data
        
        round_above_min = args.round_idx > args.new_data_min_rounds
        clinet_in_list = args.rank in args.psuedo_generate_clients or args.rank in args.psuedo_recovery_on_clients
        
        if args.use_new_data and round_above_min  and clinet_in_list: 
            try:
                args.org_data_train = train_data.dataset.img_files
                args.org_data_test = test_data.dataset.img_files
                
                args.new_data_train = check_dataset(args.new_data_conf)['train']
                args.new_data_test = check_dataset(args.new_data_conf)['val']
                
                # TODO: Should we use pseudo labels or GT ?
                if args.use_new_data_pseudos and ((args.rank in args.psuedo_generate_clients) or (args.rank in args.psuedo_recovery_on_clients)):
                    
                    if (args.client_map50_all[args.client_id]<args.min_map_for_pseudo_generation and args.round_idx>0)or(args.round_idx==0 and args.weights=="none"):
                        _log_it(args,f"mAP too low. More training required before generating pseudo labels.")
                        return train_data, test_data
                    
                    else:
                        # Remove old labels
                        # if os.path.isdir(args.save_dir/'labels'):   
                        _dir = args.save_dir/'labels'/f'Trainer_{args.client_id}--Round_{args.round_idx}'  
                        if os.path.isdir(_dir):     
                            _log_it(args,f"Removing old label files if present at {_dir}")
                            shutil.rmtree(_dir)
                        
                        
                        # Load new dataset equal to args.new_data_num_images_train from args.new_data
                        _log_it(args,f"Loading {args.new_data_num_images_train} images from new train dataset at [{args.new_data}]")
                        new_train_dataloader = get_new_data(args.new_data_train,args,shuffle=False)
                        
                        _log_it(args,f"Loading {args.new_data_num_images_test } images from new test  dataset at [{args.new_data}]")
                        new_test_dataloader  = get_new_data(args.new_data_test, args,shuffle=False)
                        
                        args.save_dir_train = args.save_dir/'train'
                        args.save_dir_test  = args.save_dir/'test'
                            
                        # ------------------ Generate only high confidence pseudo labels without confidence values -----------------
                        if (args.use_new_data_pseudos and (args.rank in args.psuedo_generate_clients)):
                            # Generate HIGH Confidence Pseudo Labels without confidence values for new dataset
                            _f_train =      pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                            model           =   model,
                                                            dataloader      =   new_train_dataloader,
                                                            compute_loss    =   compute_loss, 
                                                            args            =   args,
                                                            conf_thresh     =   0.5,
                                                            save_conf       =   False,
                                                            save_dir        =   args.save_dir_train
                                                            )
                            if _f_train==[]: 
                                return train_data   #FIXME: Return original dataloader
                            else:
                                args.pseudo_label_path_train = os.path.split(_f_train)[0]    
                                # Merge un-recovered pseudo labels of new data with original data
                                # print(args.pseudo_label_path)
                                new_data_path_train = args.pseudo_label_path_train


                                _f_test =   pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                            model           =   model,
                                                            dataloader      =   new_test_dataloader,
                                                            compute_loss    =   compute_loss, 
                                                            args            =   args,
                                                            conf_thresh     =   0.5,
                                                            save_conf       =   False,
                                                            save_dir        =   args.save_dir_test
                                                            )
                                
                                args.pseudo_label_path_test = os.path.split(_f_test)[0]    
                                # Merge un-recovered pseudo labels of new data with original data
                                # print(args.pseudo_label_path)
                                new_data_path_test = args.pseudo_label_path_test
                                data_type="pseudo labels"



                        # ----------------- Generate Pseudo Labels and perform missing box recovery --------------------
                        if args.use_new_data_recover and (args.rank in args.psuedo_recovery_on_clients): #TODO: Should we apply recovery  on the generated pseudo labels 

                            # Train Dataset
                            # Generate Pseudo Labels for new dataset
                            args.path_low_train =   pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                                    model           =   model,
                                                                    dataloader      =   new_train_dataloader,
                                                                    compute_loss    =   compute_loss, 
                                                                    args            =   args,
                                                                    conf_thresh     =   args.conf_thresh_low,
                                                                    save_dir        =   args.save_dir_train
                                                                    )
                            if args.path_low_train==[]: 
                                return train_data, test_data   #FIXME: Return original dataloader
                            # Run Forward and Backward Bounding Box Recovery
                            args.new_pseudos_recovered_train = recover_labels( args , args.path_low_train )
                            # Merge recovered pseudos of new data with original data
                            new_data_path_train = args.new_pseudos_recovered_train
                            data_type="recovered pseudo labels"


                        
                            # Test Dataset
                            # Generate Pseudo Labels for new dataset
                            args.path_low_test =    pseudo_labels(  data            =   check_dataset(args.new_data_conf),
                                                                    model           =   model,
                                                                    dataloader      =   new_test_dataloader,
                                                                    compute_loss    =   compute_loss, 
                                                                    args            =   args,
                                                                    conf_thresh     =   args.conf_thresh_low,
                                                                    save_dir        =   args.save_dir_test
                                                                    )
                            # Run Forward and Backward Bounding Box Recovery
                            args.new_pseudos_recovered_test = recover_labels( args , args.path_low_test )
                            # Merge recovered pseudos of new data with original data
                            new_data_path_test = args.new_pseudos_recovered_test
                            data_type="recovered pseudo labels"



                else:# Don't use pseudo labels
                    if (args.rank in args.psuedo_generate_clients) and (args.rank in args.psuedo_generate_clients):
                        _log_it(args,f"Client not in the list to generate pseudo labels or recover them.")
                        
                    _log_it(args,f"Removing old label files if present at {args.save_dir/'labels'}")
                    # Use ground truth data as new training data
                    new_data_path, new_data_path_test = get_X_GTs(args)
                    _log_it(args,f"Reading {args.new_data_num_images_train} GT for new train dataset at {args.new_data_train} and save at {new_data_path_train}")
                    _log_it(args,f"Reading {args.new_data_num_images_test} GT for new test dataset at {args.new_data_test} and save at {new_data_path_test}")
                    data_type="ground truth"
                
                #merge
                # train data = train data + pseudo labels    
                mixed_train_data_path, numimgs = merge_training_lists(args, org_files = args.org_data_train, new_data_path = new_data_path_train)        
                mixed_test_data_path , numimgs = merge_training_lists(args, org_files = args.org_data_test , new_data_path = new_data_path_test )        

                _log_it(args,f"Creating the dataloader using original data (ground truth) and new training data ({data_type}) from {new_data_path}.")
                
                train_data = get_new_data(mixed_train_data_path,args,numimgs=numimgs)
                test_data  = get_new_data(mixed_test_data_path ,args,numimgs=numimgs) #FIXME:
            
                
            except Exception as e:
                _log_it(args,f"Trouble loading the pseudo labels.\n\tError: {e}")
        else:
            _log_it(args,f"Continue with original data only.")
        
        
        
        args.new_train_data = train_data
        args.new_test_data = test_data
        return train_data, test_data
            
    def _log_it(args,msg):
        args.curr_step+=1
        msg =   "\n\t"                                                                +\
                colorstr( "bright_green"  , "bold" , f"Step {args.curr_step}: "      )+\
                colorstr( "bright_blue"   , "bold" , f"Round {args.round_idx}: "     )+\
                colorstr( "bright_cyan"   , "bold" , f"Client {args.client_id}\n\t"  )+\
                colorstr( args.color_str  , "bold" , str(msg)                        )+\
                "\n"
        LOGGER.info(msg)
        # args.logging.info(msg)
        
    def get_X_GTs(args):
        import random
        tmp_gt_files_train = args.tmp_gt_files.split('.')[0]+f"_train_client{args.client_id}.txt"
        tmp_gt_files_test = args.tmp_gt_files.split('.')[0]+f"_test_client{args.client_id}.txt"
        _f = open(tmp_gt_files_train, 'w')
        [_f.write(x) for x in random.sample(population=open(args.new_data,'r').readlines(), k=args.new_data_num_images_train)]
        _f.close()
        __f = open(tmp_gt_files_test, 'w')
        [__f.write(x) for x in random.sample(population=open(args.new_data_test,'r').readlines(), k=args.new_data_num_images_train)]
        __f.close()
        return tmp_gt_files_train, tmp_gt_files_test
        
    def partition_data(data_path,total_num=[],shuffle=True):
        if os.path.isfile(data_path):
            with open(data_path) as f:
                data = sorted(f.readlines())
            n_data = len(data)
        else:
            n_data = len(os.listdir(data_path))
        if total_num==[]:
            total_num = n_data
        if shuffle:
            idxs = np.random.permutation(total_num)
        else:
            idxs    = np.random.permutation(total_num)
            _start  = idxs[np.random.permutation(total_num)[0]]
            _idxs   = [] 
            for i in range(_start,_start+total_num):
                _idxs.append(i)
            idxs    = np.array(_idxs)
            print(f"\tTaking indices from {_start} to {_start+total_num}")
        batch_idxs = np.array_split(idxs, 1)
        net_dataidx_map = {i: batch_idxs[i] for i in range(1)}
        return net_dataidx_map


    def copy_images(args,target):
        new_files=[]
        merged_dir = os.path.abspath(os.path.split(args.path_low)[0])
        target = os.path.abspath(target)
        for _file in os.listdir(merged_dir):
            if _file.endswith('.jpg'):
                shutil.copyfile(os.path.join(merged_dir,_file), os.path.join( target, os.path.basename(_file)) )
                

    def move_images(args,target):
        new_files=[]
        merged_dir = os.path.abspath(os.path.split(args.path_low)[0])
        target = os.path.abspath(target)
        for _file in os.listdir(merged_dir):
            if _file.endswith('.jpg'):
                shutil.move(os.path.join(merged_dir,_file), os.path.join( target, os.path.basename(_file)) )
                
                
    def merge_training_lists(args, org_files = [], new_data_path = []):
        
        if org_files==[]:
            org_files = args.org_data, new_data_path = args.new_pseudos_recovered
        
        isfile = os.path.isfile(new_data_path)
        
        if isfile:
            with open(new_data_path) as f:
                new_files = f.readlines()
        else:
            # new_files = os.listdir(new_data_path)
            new_files=[]
            for _file in os.listdir(new_data_path):
                if _file.endswith('.jpg'):
                    new_files.append(os.path.join(os.path.abspath(new_data_path),_file))
        
        _log_it(args,f"Merging {len(org_files)} images from {args.data_conf} with {len(new_files)} images from {new_data_path}.")
        tmp_merge_file = args.tmp_merge_file.split('.')[0]+f"_client{args.client_id}.txt"
        with open(tmp_merge_file , 'w') as outfile:
            for line in org_files:
                outfile.write(line+'\n')
            for line in new_files:
                outfile.write(line+'\n')
                # outtext = os.path.join(new_data_path,line)+'\n'
                # outfile.write(os.path.abspath(outtext))
                
        numimgs = len(org_files)+len(new_files)
        return tmp_merge_file, numimgs

    def get_new_data(data_path,args,numimgs=0,shuffle=True):
        # data_path            = args.new_data
        imgsz_test           = args.new_dataloader_args[0]
        total_batch_size     = args.new_dataloader_args[1]
        gs                   = args.new_dataloader_args[2]
        args                 = args.new_dataloader_args[3]
        hyp                  = args.new_dataloader_args[4]
        rect                 = args.new_dataloader_args[5]
        rank                 = -1
        pad                  = args.new_dataloader_args[7]
        workers              = args.new_dataloader_args[8]
        total_num            = args.new_data_num_images_train if numimgs==0 else numimgs
        net_dataidx_map_test = partition_data(data_path,total_num=total_num,shuffle=shuffle)
        testloader           = create_dataloader(
                                                data_path,
                                                imgsz_test,
                                                total_batch_size,
                                                gs,
                                                args,  # testloader
                                                hyp=hyp,
                                                rect=rect,
                                                rank=rank,
                                                pad=pad,
                                                net_dataidx_map=net_dataidx_map_test[0],
                                                workers=workers,
                                                )[0]
        return testloader


    def modify_dataset(dataloader,new_path):
        
        count_missing_file = 0
        # Replace GT with Pseudos
        for i, _label_file in enumerate(dataloader.dataset.label_files):
            
            # Path for new label file
            new_label_file = os.path.join ( os.path.realpath(new_path), os.path.basename(_label_file) )
            
            # Make file if there are no pseudos for this file
            if not os.path.isfile(new_label_file): 
                open(new_label_file, 'w')
                count_missing_file+=1
                
            # Read labels from pseudo file
            _label_old    = dataloader.dataset.labels[i]
            _label_pseudo = np.array([x.split() for x in open(new_label_file, 'r').readlines()],dtype='float32')
            
            # Replace labels in dataset
            dataloader.dataset.labels[i] = _label_pseudo
            
            # Replace label file address in dataset
            dataloader.dataset.label_files[i] = new_label_file
            
        return dataloader

    def pseudo_labels(data,model,dataloader,compute_loss,args,conf_thresh=0.5,save_conf=True, save_dir=[]):
        """
        Generate pseudo labels
        """
        
        epoch_no        =   args.round_idx
        half            =   False
        single_cls      =   False
        plots           =   False
        host_id         =   args.client_id
        batch_size      =   args.batch_size
        imgsz           =   args.img_size[0]
        
        if save_dir==[]: save_dir=args.save_dir
        
        conf = 'low' if conf_thresh<0.5 else 'high'   # if 'low', images will also be copied with the generated labels
        
        _log_it(args,f"Generating {conf} confidence labels at {conf_thresh} threshold.")
        
        results, maps, _t, _file =  pseudos.run(data            = data          ,
                                                batch_size      = batch_size    ,
                                                imgsz           = imgsz         ,
                                                half            = half          ,
                                                model           = model         ,
                                                single_cls      = single_cls    ,
                                                dataloader      = dataloader    ,
                                                save_dir        = save_dir      ,
                                                plots           = plots         ,
                                                compute_loss    = compute_loss  ,
                                                
                                                save_txt        = True          ,
                                                save_conf       = save_conf     ,
                                                epoch_no        = epoch_no      ,
                                                host_id         = host_id       ,
                                                conf_thres      = conf_thresh   ,   
                                                confidence      = conf                                     
                                                )
        
        if _file ==[]:
            _log_it(args,f"No labels generated due to poor predictions. Model needs more training.")
            
        return _file

    def recover_labels(args,low_conf_pred_path):

        
        from pathlib import Path            
        class opt_recovery(object):

            agnostic_nms        = False
            augment             = False
            classes             = None
            config_deepsort     = args.new_deepsort_config
            device              = args.device.index
            dnn                 = False
            evaluate            = False
            fourcc              = 'mp4v'
            half                = False
            imgsz               = args.new_dataloader_args[0]
            max_hc_boxes        = 1000
            name                = 'Recover'
            project             = Path('runs/track')
            save_img            = False
            save_vid            = False
            show_vid            = False
            visualize           = False
            
            deep_sort_model     = "resnet50_MSMT17"
            yolo_model          = []
            conf_thres          = args.conf_thres
            iou_thres           = args.iou_thres
            save_txt            = True
            exist_ok            = True
            reverse             = False
            
            epoch_no            =   0
            source              =   os.path.split(low_conf_pred_path)[0]
            host_id             =   args.client_id
            
            # source = save_dir / 'labels' 
            # source = source / f'Trainer_{host_id}--epoch_{epoch_no}'
            # source = source / f'low_0.01'
            # source = source / f'high_0.5'
            # source.mkdir(parents=True, exist_ok=True)
            source = str(source)
            output = source
            # txt_path = str(Path(opt.output)) + '/' + os.path.basename(path)[:-4]+'.txt'
        try:
            # Recover in Forward
            opt_recovery.output = os.path.split( opt_recovery.source )[0] +'/Recover-FW'
            opt_recovery.reverse= False
            with torch.no_grad():
                _log_it(args,f"Performing pseudo label recovery in Forward direction at {opt_recovery.output}.")
                recover.detect(opt_recovery,args)     
        except Exception as e:    
            print(f"Error in FW Recovery - {e}")

        try:
            # Recover in Backward
            opt_recovery.output = os.path.split( opt_recovery.source )[0] +'/Recover-BW'
            opt_recovery.reverse= True
            with torch.no_grad():
                _log_it(args,f"Performing pseudo label recovery in Backward direction at {opt_recovery.output}.")
                recover.detect(opt_recovery,args)
        except Exception as e:    
            print(f"Error in BW Recovery - {e}")
        
        try:
            # Merge
            class opt_merge(object):
                forward     = os.path.split( opt_recovery.source )[0]+'/Recover-FW'
                backward    = os.path.split( opt_recovery.source )[0]+'/Recover-BW'
                merged      = os.path.split( opt_recovery.source )[0]+'/Recover-Merged'
            _log_it(args,f"Merge the results from Yolo's pseudo labels with the recovered labels from forward and backward recovery.")
            merge(opt_merge)
        except Exception as e:    
            print(f"Error in Merging - {e}")
        
        _log_it(args,f"Move the images to the Recover-Merged directory for next training.")
        move_images(args,opt_merge.merged)
            
        return opt_merge.merged
    
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where(
            (iou >= iouv[i]) & correct_class
        )  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


class YOLOv5Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super(YOLOv5Trainer, self).__init__(model, args)
        self.hyp = args.hyp
        self.args = args
        self.round_loss = []
        self.round_idx = 0
        args.client_map50_all = dict()
        args.client_map_all   = dict()
        args.client_map50_all[args.rank]=0
        args.client_map_all[args.rank]=0

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, test_data, device, args):
        
        
        # %% =============== Logging information ==========================
        host_id = int(list(args.client_id_list)[1])
        args.client_id = host_id
        logging.info("Start training on Trainer {}".format(host_id))
        logging.info(f"Hyperparameters: {self.hyp}, Args: {self.args}")
        LOGGER.info(colorstr('hyperparameters: ')+ ', '.join(f'{k}={v}' for k, v in self.hyp.items()))  ############
        model = self.model
        
        
        
        # %% =============== Hyperparameters ==========================
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp
        epochs = args.epochs  # number of epochs



        # %% =============== Adding parameter groups in Optimizer ==========================
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if args.client_optimizer == "adam":
            optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2



        # %% ============================= Freeze Layers ====================================
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False



        # %% =============== Define Learning Rate scheduler based on total epochs ==================
        total_epochs = epochs * args.comm_round
        lf = (lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"] )  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)



        # %% =============== Model Loss and Move model to device ======================
        compute_loss = ComputeLoss(model)
        model.to(device)
        model.train()
        
        
        
        # %% ======================== Shoaib's part =======================================
        args.client_data_size_org_train = len(train_data.dataset.labels)
        args.client_data_size_org_test = len(test_data.dataset.labels)
        args.logging = logging
        if use_shoaib_code: # TODO: 
            train_data, test_data = use_new_data(args,model,compute_loss,train_data, test_data)
            args.client_data_size_mod_train = len(train_data.dataset.labels)
            args.client_data_size_mod_test = len(test_data.dataset.labels)
        # if args.round_idx==1 and args.rank==args.psuedo_gen_on_clients[0]: fedml.core.mlops.mlops_profiler_event.MLOpsProfilerEvent.log_to_wandb(args.__dict__)
        
        
        
        

        # %% ======================= Validation Part - Before Training ====================================

        if use_shoaib_code: 
            LOGGER.info(colorstr("bright_green","bold", f"\tTesting on {args.client_data_size_mod_test} images. Original testing data is {args.client_data_size_org_test} for client {args.client_id}.\n"))
            
        logging.info("Start val on Trainer before training {}".format(host_id))
        save_dir = self.args.save_dir
        data_dict = check_dataset(args.data_conf)
        self._val(  data=data_dict,
                    batch_size=args.batch_size,
                    imgsz=args.img_size[0],
                    half=False,
                    model=model,
                    single_cls=False,
                    dataloader=test_data,
                    save_dir=save_dir,
                    plots=False,
                    compute_loss=compute_loss, 
                    args = args,
                    phase= "Before_Training_"
                    )




        
        # %%
        # =================================================================================
        # ============================== Training Part ====================================
        # =================================================================================
        epoch_loss = []
        mloss = torch.zeros(3, device=device)  # mean losses
        logging.info("\tEpoch gpu_mem box obj cls total targets img_size time")
        for epoch in range(args.epochs):
            model.train()
            t = time.time()
            batch_loss = []
            logging.info("\tTrainer_ID: {0}, Epoch: {1}".format(host_id, epoch))
            
            if use_shoaib_code: 
                LOGGER.info(colorstr("bright_green","bold", f"\n\tTraining on {args.client_data_size_mod_train} images. Original training data is {args.client_data_size_org_train} for client {args.client_id}.\n"))
            
            for (batch_idx, batch) in enumerate(train_data):
                # ============= Read images and move to GPU ===================
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5


                # ============= Clear the previous gradient values ============
                optimizer.zero_grad()
                
                
                # ========================== Forward ==========================
                # with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device).float())  # loss scaled by batch_size


                # ========================== Backward ==========================
                loss.backward()
                
                
                # ======================= Update weights =======================
                optimizer.step()
                batch_loss.append(loss.item())


                # ====================== Calculate Losses ======================
                mloss = (mloss * batch_idx + loss_items) / (batch_idx + 1)  # update mean losses
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 5) % ("%g/%g" % (epoch, epochs - 1),mem,*mloss,targets.shape[0],imgs.shape[-1])
                logging.info(s)

            # ============= Update learning rate =============
            scheduler.step()





            # %% ============================== Results Logging ==================================
            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(f"Trainer {host_id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}")

            logging.info("#" * 20)

            try: logging.info(f"Trainer {host_id} epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch")
            except: pass
            logging.info("#" * 200)
            
            MLOpsProfilerEvent.log_to_wandb({
                                            f"client_{host_id}_round_idx":  self.round_idx,
                                            f"client_{host_id}_box_loss":   np.float(mloss[0]),
                                            f"client_{host_id}_obj_loss":   np.float(mloss[1]),
                                            f"client_{host_id}_cls_loss":   np.float(mloss[2]),
                                            f"client_{host_id}_total_loss": np.float(mloss.sum()),
                                            f"round_idx":   self.round_idx,
                                            f"box_loss":    np.float(mloss[0]),
                                            f"obj_loss":    np.float(mloss[1]),
                                            f"cls_loss":    np.float(mloss[2]),
                                            f"total_loss":  np.float(mloss.sum()),
                                            f"Round_No":    args.round_idx,
                                            f"Epoch_No":    epoch,
                                            })



            # # %% ============================== Saving Weights ===================================
            # if (epoch + 1) % self.args.checkpoint_interval == 0:
            #     model_path = (self.args.save_dir/ "weights"/ f"model_client_{host_id}_epoch_{epoch}.pt")
            #     logging.info(f"Trainer {host_id} epoch {epoch} saving model to {model_path}")
               
            #     # Old saving method     # torch.save(model.state_dict(), model_path)
            #     # Modified saving method
            #     ckpt = {'epoch': epoch,
            #             'model': copy.deepcopy(model).half(),
            #             'optimizer': optimizer.state_dict()}
            #     torch.save(ckpt, model_path)
            #     del ckpt




        # %% ============================== Saving Weights ===================================
        logging.info("End training on Trainer {}".format(host_id))
        
        model_path = self.args.save_dir / "weights" / f"model_client_{host_id}_round_{self.round_idx}.pt"
        logging.info(f"Trainer {host_id} saving model to {model_path}")
        
        # Old saving method     # torch.save(model.state_dict(), model_path)
        # Modified saving method
        ckpt = {'model': copy.deepcopy(model).half(),
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, model_path)
        del ckpt





        # %% =================== Validation Part - After Training ====================================

        if use_shoaib_code: 
            LOGGER.info(colorstr("bright_green","bold", f"\tTesting on {args.client_data_size_mod_test} images. Original testing data is {args.client_data_size_org_test} for client {args.client_id}.\n"))
            
        # if (epoch + 1) % self.args.frequency_of_the_test == 0:
        # if (epoch + 1) % self.args.epochs == 0:
        logging.info("Start val on Trainer {}".format(host_id))
        #self.val(test_data, device, args)
        save_dir  = self.args.save_dir
        data_dict = check_dataset(args.data_conf)
        self._val(  data=data_dict,
                    batch_size=args.batch_size,
                    imgsz=args.img_size[0],
                    half=False,
                    model=model,
                    single_cls=False,
                    dataloader=test_data,
                    save_dir=save_dir,
                    plots=False,
                    compute_loss=compute_loss, 
                    args = args,
                    phase= "After_Training_"
                    )

        # ============================== Logging Results ===================================
        epoch_loss = np.array(epoch_loss)
        logging.info(f"Epoch loss: {epoch_loss}")

        fedml.mlops.log({
                        f"round_idx": self.round_idx,
                        f"train_box_loss": np.float(epoch_loss[-1, 0]),
                        f"train_obj_loss": np.float(epoch_loss[-1, 1]),
                        f"train_cls_loss": np.float(epoch_loss[-1, 2]),
                        f"train_total_loss": np.float(epoch_loss[-1, :].sum()),
                        })

        self.round_loss.append(epoch_loss[-1, :])
        if self.round_idx == args.comm_round:
            self.round_loss = np.array(self.round_loss)
            logging.info(f"Trainer {host_id} round {self.round_idx} finished, round loss: {self.round_loss}")




        # %% ========================= End Trainer Function ===================================
        return









    def _val(self, 
             data, 
             batch_size, 
             imgsz, 
             half, 
             model, 
             single_cls, 
             dataloader, 
             save_dir, 
             plots, 
             compute_loss, 
             args,
             phase=''):
        
        host_id = int(list(args.client_id_list)[1])
        results, maps, _ = validate.run(data = data,
                                    batch_size = args.batch_size,#128,
                                    imgsz = imgsz,
                                    half = half,
                                    model = model,
                                    single_cls = single_cls,
                                    dataloader = dataloader,
                                    save_dir = save_dir,
                                    plots = plots,
                                    compute_loss = compute_loss)
        
        
        args.client_map50_all[host_id]  = np.float(results[2])
        args.client_map_all[host_id]    = np.float(results[3])
        
        MLOpsProfilerEvent.log_to_wandb(
                {
                    f"{phase}client_{host_id}_mean_precision":     np.float(results[0]),
                    f"{phase}client_{host_id}_mean_recall":        np.float(results[1]),
                    f"{phase}client_{host_id}_map@50":             np.float(results[2]),
                    f"{phase}client_{host_id}_map":                np.float(results[3]),
                        
                    f"{phase}client_{host_id}_test_box_loss":      np.float(results[4]),
                    f"{phase}client_{host_id}_test_obj_loss":      np.float(results[5]),
                    f"{phase}client_{host_id}_test_cls_loss":      np.float(results[6]),
                    
                    f"{phase}client_{host_id}_training_data_org":  args.client_data_size_org_train,
                    f"{phase}client_{host_id}_training_data_mod":  args.client_data_size_mod_train,
                    f"{phase}client_{host_id}_testing_data_org":   args.client_data_size_org_test,
                    f"{phase}client_{host_id}_testing_data_mod":   args.client_data_size_mod_test,
                    
                    
                    f"{phase}mean_precision":      np.float(results[0]),
                    f"{phase}mean_recall":         np.float(results[1]),
                    f"{phase}map@50_all_classes":  np.float(results[2]),
                    f"{phase}map_all_classes":     np.float(results[3]),
                    
                    f"{phase}test_box_loss":       np.float(results[4]),
                    f"{phase}test_obj_loss":       np.float(results[5]),
                    f"{phase}test_cls_loss":       np.float(results[6]),
 
                    f"{phase}training_data_org":   args.client_data_size_org_train,
                    f"{phase}training_data_mod":   args.client_data_size_mod_train,
                    f"{phase}testing_data_org":    args.client_data_size_org_test,
                    f"{phase}testing_data_mod":    args.client_data_size_mod_test,
                    
                    
                    f"{phase}client_{host_id}_round_idx": args.round_idx,
                    f"{phase}Round_No": args.round_idx,
                }
            )
        
        for i,cls in enumerate(check_dataset(args.data_conf)['names']):
            MLOpsProfilerEvent.log_to_wandb({
                                            f"client_{host_id}_map_{cls}": maps[i],
                                            f"map_{cls}": maps[i],
                                            f"client_{host_id}_round_idx": args.round_idx,
                                            f"Round_No": args.round_idx,
                                            })
        
        logging.info(f"mAPs of all class in a list {maps}")

    
    def val(self, test_data, device, args):
        host_id = int(list(args.client_id_list)[1])
        logging.info(f"Trainer {host_id} val start")
        model = self.model
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp

        model.eval()
        model.to(device)
        compute_loss = ComputeLoss(model)
        loss = torch.zeros(3, device=device)
        jdict, stats, ap, ap_class = [], [], [], []
        s = ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "P",
            "R",
            "mAP50",
            "mAP50-95",
        )
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        conf_thres = 0.001
        iou_thres = 0.6
        max_det = 300

        nc = model.nc  # number of classes
        iouv = torch.linspace(
            0.5, 0.95, 10, device=device
        )  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        names = {
            k: v
            for k, v in enumerate(
                model.names if hasattr(model, "names") else model.module.names
            )
        }
        dt = Profile(), Profile(), Profile()
        seen = 0
        pbar = tqdm(test_data, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for (batch_idx, batch) in enumerate(pbar):
            (im, targets, paths, shapes) = batch
            im = im.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            nb, _, height, width = im.shape  # batch size, channels, height, width

            # inference
            with torch.no_grad(): 
                preds, train_out = model(im)
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            preds = non_max_suppression(
                preds,
                conf_thres,
                iou_thres,
                labels=lb,
                multi_label=True,
                agnostic=False,
                max_det=max_det,
            )

            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = (labels.shape[0],pred.shape[0])  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(
                        im[si].shape[1:], tbox, shape, shapes[si][1]
                    )  # native-space labels
                    labelsn = torch.cat(
                        (labels[:, 0:1], tbox), 1
                    )  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                stats.append(
                    (correct, pred[:, 4], pred[:, 5], labels[:, 0])
                )  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *stats, plot=False, save_dir=self.args.save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            MLOpsProfilerEvent.log_to_wandb(
            {
                f"client_{host_id}_mean_precision": np.float(mp),
                f"client_{host_id}_mean_recall":np.float(mr),
                f"clinet_{host_id}_mAP@50":np.float(map50),
                f"client_{host_id}_mAP@0.5:0.95":np.float(map),   
                f"mean_precision": np.float(mp),
                f"mean_recall":np.float(mr),
                f"mAP@50":np.float(map50),
                f"mAP@0.5:0.95":np.float(map),    
                f"Round_No": args.round_idx,  
            }
        )
        nt = np.bincount(
            stats[3].astype(int), minlength=nc
        )  # number of targets per class
        
        

        # Print results
        logging.info(s)
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        logging.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
        return
