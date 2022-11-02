import copy
import logging
import math
import time

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
from model.yolov5 import \
    val_pseudos as \
    pseudos  # imported to use modified yolov5 validation function!!!
    
from Yolov5_DeepSORT_PseudoLabels import trackv2_from_file as recover
from Yolov5_DeepSORT_PseudoLabels.merge_forward_backward_v2 import merge 

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
from model.yolov5 import \
    val_pseudos as \
    pseudos  # imported to use modified yolov5 validation function!!!
    
from Yolov5_DeepSORT_PseudoLabels import trackv2_from_file as recover
from Yolov5_DeepSORT_PseudoLabels.merge_forward_backward_v2 import merge 


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

def pseudo_labels(data,batch_size,imgsz,half,model,single_cls,dataloader,save_dir,plots,compute_loss,args,epoch_no,host_id):
    """
    Generate pseudo labels
    """
    for conf,thresh in zip ( ['low','high'], [0.001, 0.5]):
        logging.info(f'Trainer {host_id } generating {conf} confidence labels for epoch {epoch_no}.')
        results, maps, _ =  pseudos.run(data            = data          ,
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
                                        save_conf       = True          ,
                                        epoch_no        = epoch_no      ,
                                        host_id         = host_id       ,
                                        conf_thres      = thresh        ,   
                                        confidence      = conf                                     
                                        )

def recover_labels(dataloader,save_dir,epoch_no,host_id):
    from pathlib import Path            
    class opt_recovery(object):

        agnostic_nms        = False
        augment             = False
        classes             = None
        config_deepsort     = 'Yolov5_DeepSORT_PseudoLabels/deep_sort/configs/deep_sort.yaml'
        device              = ''
        dnn                 = False
        evaluate            = False
        fourcc              = 'mp4v'
        half                = False
        imgsz               = [640, 640]
        max_hc_boxes        = 1000
        name                = 'Recover'
        project             = Path('runs/track')
        save_img            = False
        save_vid            = False
        show_vid            = False
        visualize           = False
        
        deep_sort_model     = "resnet50_MSMT17"
        yolo_model          = "best.pt"
        conf_thres          = 0.5
        iou_thres           = 0.6
        save_txt            = True
        exist_ok            = True
        reverse             = False
        
        source=save_dir / 'labels' 
        source = source / f'Trainer_{host_id}--epoch_{epoch_no}'
        source = source / f'low_0.001'
        # source = source / f'high_0.5'
        source.mkdir(parents=True, exist_ok=True)
        source = str(source)
        output = source
    
    # Recover in Forward
    opt_recovery.output = opt_recovery.source+'-FW'
    opt_recovery.reverse= False
    with torch.no_grad():
        recover.detect(opt_recovery)
        print(f"\n\n\n")

    # Recover in Backward
    opt_recovery.output = opt_recovery.source+'-BW'
    opt_recovery.reverse= True
    with torch.no_grad():
        recover.detect(opt_recovery)
        print(f"\n\n\n")

    # Merge
    class opt_merge(object):
        forward     = opt_recovery.source+'-FW'
        backward    = opt_recovery.source+'-BW'
        merged      = opt_recovery.source+'-Merged'
    merge(opt_merge)

    # Replace pseudo labels in dataset
    modify_dataset(dataloader,opt_merge.merged)
        
    return dataloader

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

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, test_data, device, args):
        host_id = int(list(args.client_id_list)[1])
        logging.info("Start training on Trainer {}".format(host_id))
        logging.info(f"Hyperparameters: {self.hyp}, Args: {self.args}")
        LOGGER.info(colorstr('hyperparameters: ')+ ', '.join(f'{k}={v}' for k, v in self.hyp.items()))  ############
        model = self.model
        
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp
        epochs = args.epochs  # number of epochs

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if args.client_optimizer == "adam":
            optimizer = optim.Adam(
                pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
            )  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(
                pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
            )

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )
        del pg0, pg1, pg2

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False

        total_epochs = epochs * args.comm_round

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2)
            * (1 - hyp["lrf"])
            + hyp["lrf"]
        )  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        model.to(device)
        model.train()

        compute_loss = ComputeLoss(model)

        
        # # if epoch>0 or True:    
        # # FIXME: Modify yolo here
        # # train data = train data + pseudo labels
        
        # # Remove old labels
        # if os.path.isdir('runs/train/exp/labels'): shutil.rmtree('runs/train/exp/labels')
        
        # # Generate Pseudo Labels
        # pseudo_labels(  data            =   check_dataset(args.opt["data"]),
        #                 batch_size      =   args.batch_size,
        #                 imgsz           =   args.img_size[0],
        #                 half            =   False,
        #                 model           =   model,
        #                 single_cls      =   args.opt['single_cls'],
        #                 dataloader      =   train_data,
        #                 save_dir        =   self.args.save_dir,
        #                 plots           =   False,
        #                 compute_loss    =   compute_loss, 
        #                 args            =   args,
        #                 epoch_no        =   0,
        #                 host_id         =   host_id
        #                 )
        
        # # Run Forward and Backward Bounding Box Recovery
        # train_data = \
        # recover_labels( dataloader      =   train_data,
        #                 save_dir        =   self.args.save_dir,
        #                 epoch_no        =   0,
        #                 host_id         =   host_id
        #                 )
        
    
        epoch_loss = []
        mloss = torch.zeros(3, device=device)  # mean losses
        logging.info("Epoch gpu_mem box obj cls total targets img_size time")
        for epoch in range(args.epochs):
            model.train()
            t = time.time()
            batch_loss = []
            logging.info("Trainer_ID: {0}, Epoch: {1}".format(host_id, epoch))

            for (batch_idx, batch) in enumerate(train_data):
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

                optimizer.zero_grad()
                # with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device).float()
                )  # loss scaled by batch_size

                # Backward
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                mloss = (mloss * batch_idx + loss_items) / (
                    batch_idx + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 5) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                logging.info(s)

            scheduler.step()

            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(
                f"Trainer {host_id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}"
            )

            logging.info("#" * 20)

            try:
                logging.info(
                    f"Trainer {host_id} epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch"
                )
            except:
                pass
            logging.info("#" * 200)
            
            MLOpsProfilerEvent.log_to_wandb(
                {
                    f"client_{host_id}_round_idx": self.round_idx,
                    f"client_{host_id}_box_loss": np.float(mloss[0]),
                    f"client_{host_id}_obj_loss": np.float(mloss[1]),
                    f"client_{host_id}_cls_loss": np.float(mloss[2]),
                    f"client_{host_id}_total_loss": np.float(mloss.sum())
                }
            )

            if (epoch + 1) % self.args.checkpoint_interval == 0:
                model_path = (
                    self.args.save_dir
                    / "weights"
                    / f"model_client_{host_id}_epoch_{epoch}.pt"
                )
                logging.info(
                    f"Trainer {host_id} epoch {epoch} saving model to {model_path}"
                )
                torch.save(model.state_dict(), model_path)

            if (epoch + 1) % self.args.frequency_of_the_test == 0:
                logging.info("Start val on Trainer {}".format(host_id))
                #self.val(test_data, device, args)
                data_dict = None
                save_dir = self.args.save_dir
                # save_dir = Path(args.opt["save_dir"])
                # weights = args.opt["weights"]
                # loggers = Loggers(save_dir, weights, args.opt, args.hyp, LOGGER)
                #data_dict = loggers.remote_dataset
                data_dict = data_dict or check_dataset(args.data_conf)
                # data_dict = data_dict or check_dataset(args.opt["data"])
                # logging.info(f"Training path: {data_dict['train']}' and Validation path: {data_dict['val']}")
                # half, single_cls, plots, callbacks = False, args.opt['single_cls'], False, None
                half, single_cls, plots, callbacks = False, False, False, None
                self._val(data=data_dict,
                          batch_size=args.batch_size,
                          imgsz=args.img_size[0],
                          half=half,
                          model=model,
                          single_cls=single_cls,
                          dataloader=test_data,
                          save_dir=save_dir,
                          plots=plots,
                          compute_loss=compute_loss, 
                          args = args
                          )
                
                
                

        logging.info("End training on Trainer {}".format(host_id))
        torch.save(
            model.state_dict(),
            self.args.save_dir / "weights" / f"model_client_{host_id}_round_{self.round_idx}.pt",
        )

        # plot for client
        # plot box, obj, cls, total loss
        epoch_loss = np.array(epoch_loss)
        # logging.info(f"Epoch loss: {epoch_loss}")

        fedml.mlops.log(
            {
                f"round_idx": self.round_idx,
                f"train_box_loss": np.float(epoch_loss[-1, 0]),
                f"train_obj_loss": np.float(epoch_loss[-1, 1]),
                f"train_cls_loss": np.float(epoch_loss[-1, 2]),
                f"train_total_loss": np.float(epoch_loss[-1, :].sum()),
            }
        )

        self.round_loss.append(epoch_loss[-1, :])
        if self.round_idx == args.comm_round:
            self.round_loss = np.array(self.round_loss)
            # logging.info(f"round_loss shape: {self.round_loss.shape}")
            logging.info(
                f"Trainer {host_id} round {self.round_idx} finished, round loss: {self.round_loss}"
            )

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
             args):
        
        host_id = int(list(args.client_id_list)[1])
        results, maps, _ = validate.run(data = data,
                                    batch_size = 128,
                                    imgsz = imgsz,
                                    half = half,
                                    model = model,
                                    single_cls = single_cls,
                                    dataloader = dataloader,
                                    save_dir = save_dir,
                                    plots = plots,
                                    compute_loss = compute_loss)
        
        MLOpsProfilerEvent.log_to_wandb(
                {
                    f"client_{host_id}_mean_precision": np.float(results[0]),
                    f"client_{host_id}_mean_recall": np.float(results[1]),
                    f"client_{host_id}_map@50": np.float(results[2]),
                    f"client_{host_id}_map": np.float(results[3]),
                    #f"client_{host_id}_test_box_loss": np.float(results[4]),
                    #f"client_{host_id}_test_obj_loss": np.float(results[5]),
                    #f"client_{host_id}_test_cls_loss": np.float(results[6]),
                    
                }
            )
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
                f"client_{host_id}_mAP@0.5:0.95":np.float(map)       
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
