import logging

import numpy as np
import torch
from pathlib import Path

import fedml
from model.yolov5.utils.general import (
    box_iou,
    non_max_suppression,
    xywh2xyxy,
    clip_coords,
)
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.metrics import ap_per_class
from model.yolov5 import val as validate
from fedml.core import ServerAggregator
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from model.yolov5.utils.general import (LOGGER, check_amp, check_dataset, check_file, check_img_size, check_yaml, colorstr)


class YOLOAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)
        #self._test(self, self.test_data, self.device)
        #try:
        # aggr_model_path = (
        #             self.args.save_dir
        #             / "weights"
        #             / f"model_aggr_{self.args.comm_round}.pt"
        #         )
        # torch.save(
        #     self.model.state_dict(),
        #     aggr_model_path,
        #     )
        # #except:
        # logging.info(f"\nOOOOOOOOOOOOOOOOOOOO| Aggregator saves aggregated weights {self.args.comm_round}|OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n")
    def test(self, test_data, device, args):
        #self._test(test_data=test_data, device=device)
        self._val(test_data, device, args)
        #pass
    
    def _val(self, test_data, device, args):
        data_dict = None
        save_dir = Path(args.save_dir)
        # save_dir = Path(args.opt["save_dir"])
        model = self.model
        model.eval()
        model.to(device)
        compute_loss = ComputeLoss(model)
        data_dict = data_dict or check_dataset(args.data_conf) 
        # data_dict = data_dict or check_dataset(args.opt["data"])
        half, single_cls, plots = False, False, False
        host_id = int(list(args.client_id_list)[1])
        logging.info(f"\n#########################| Server ID {host_id} performs evaluation |############################\n")
        results, maps, _ = validate.run(data=data_dict,
                                        batch_size=args.batch_size,
                                        imgsz=args.img_size[0],
                                        half=half,
                                        model=model,
                                        single_cls=single_cls,
                                        dataloader=test_data,
                                        save_dir=save_dir,
                                        plots=plots,
                                        compute_loss=compute_loss,
                                        device=device
                                        )
        #return results, maps
        
        MLOpsProfilerEvent.log_to_wandb(
                {
                    f"Server_{host_id}_mean_precision": np.float(results[0]),
                    f"Server_{host_id}_mean_recall": np.float(results[1]),
                    f"Server_{host_id}_map@50": np.float(results[2]),
                    f"map@50_all_classes": np.float(results[2]),
                    f"Server_{host_id}_map": np.float(results[3]),
                    f"map_all_classes": np.float(results[3]),
                    f"Server_{host_id}_test_box_loss": np.float(results[4]),
                    f"Server_{host_id}_test_obj_loss": np.float(results[5]),
                    f"Server_{host_id}_test_cls_loss": np.float(results[6]),
                    f"Server_{host_id}_round_idx": args.round_idx,
                    f"Round": args.round_idx,
                    
                }
            )
        
        for i,cls in enumerate(check_dataset(args.data_conf)['names']):
            MLOpsProfilerEvent.log_to_wandb(
                {
                    f"map_{cls}": maps[i],
                    f"Round_No": args.round_idx,
                }
            )
        logging.info(f"mAPs of all class in a list {maps}")

    def _test(self, test_data, device, args):
        
        #results, maps = self._val(test_data, device, args)
        
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        args = self.args

        test_metrics = {
            "test_correct": 0,
            "test_total": 0,
            "test_loss": 0,
        }

        if not test_data:
            logging.info("No test data for this trainer")
            return test_metrics

        model.eval()
        model.to(device)

        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = {
            k: v
            for k, v in enumerate(
                model.names if hasattr(model, "names") else model.module.names
            )
        }
        s = ("%20s" + "%12s" * 6) % (
            "Class",
            "Images",
            "Targets",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        p, r, f1, mp, mr, map50, map, t0, t1 = (
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
        loss = torch.zeros(3, device=device)
        compute_loss = ComputeLoss(model)
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

        for batch_i, (img, targets, paths, shapes) in enumerate(test_data):
            img = img.to(device, non_blocking=True)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img.float() / 256.0 - 0.5
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                inf_out, train_out = model(img)  # inference and training outputs

                # Loss
                loss += compute_loss([x.float() for x in train_out], targets)[1][
                    :3
                ]  # box, obj, cls

                # Run NMS
                output = non_max_suppression(
                    inf_out, conf_thres=args.conf_thres, iou_thres=args.iou_thres
                )

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                tcls,
                            )
                        )
                    continue

                # W&B logging
                # TODO

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device
                )
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (
                            (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        )  # prediction indices
                        pi = (
                            (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        )  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                                1
                            )  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if (
                                        len(detected) == nl
                                    ):  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            # TODO

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *stats, plot=False, save_dir=args.save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=args.nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # W&B logging
        # TODO
        # Print results
        pf = "%20s" + "%12.3g" * 6  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if args.yolo_verbose and args.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
            args.img_size,
            args.img_size,
            args.batch_size,
        )  # tuple

        # Return results
        model.float()  # for training
        maps = np.zeros(args.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        # all metrics
        # metrics = (mp, mr, map50, map, *(loss.cpu() / len(test_data)).tolist()), maps, t
        # logging.info(f"Test metrics: {metrics}")

        fedml.mlops.log(
            {
                f"round_idx": self.round_idx,
                f"test_mp": np.float(mp),
                f"test_mr": np.float(mr),
                f"test_map50": np.float(map50),
                f"test_map": np.float(map),
                f"test_loss": np.float(sum((loss.cpu() / len(test_data)).tolist())),
            }
        )
        
        MLOpsProfilerEvent.log_to_wandb(
            {
                f"round_idx": self.round_idx,
                f"test_mp": np.float(mp),
                f"test_mr": np.float(mr),
                f"test_map50": np.float(map50),
                f"test_map": np.float(map),
                f"test_loss": np.float(sum((loss.cpu() / len(test_data)).tolist())),
            }
        )

        test_metrics = {
            "test_correct": 0,
            "test_total": len(test_data),
            "test_loss": sum((loss.cpu() / len(test_data)).tolist()),
        }
        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        if args.round_idx is not None:
            self.round_idx = args.round_idx
            logging.info(f"round_idx: {self.round_idx}")
            if (self.round_idx + 1) % self.args.server_checkpoint_interval == 0:
                logging.info(f"Saving model at round {self.round_idx}")
                torch.save(
                    self.model,
                    self.args.save_dir / "weights" / f"aggr_model_{self.round_idx}.pt",
                )
        return True
