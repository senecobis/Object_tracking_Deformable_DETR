# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from email.policy import strict
import os
from pickle import NONE
import sys
import time
from os import path as osp
from unittest import result
from PIL import Image

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum,
                                          interpolate_tracks, plot_sequence)


""" Source the parent directory.
 for istance you can run 

 PYTHONPATH=$PYTHONPATH:/home/rpellerito/trackformer
 export PYTHONPATH
 
 """

mm.lap.default_solver = 'lap'


with open("/home/rpellerito/trackformer/cfgs/track.yaml", 'r') as stream:
    try:
        track_yam=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#ex = sacred.Experiment()
#ex.add_config('/home/rpellerito/trackformer/cfgs/track.yaml')
#ex.add_named_config('reid', '/home/rpellerito/trackformer/cfgs/track_reid.yaml')

#@ex.automain
def main(seed, dataset_name, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate, verbose, load_results_dir,
         data_root_dir, generate_attention_maps, frame_range,
         _config, _log, _run, obj_detector_model):

    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during
    # training. in that case we omit verbose outputs.
    #if obj_detector_model is None:
    #    sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        yaml.dump(
            _config,
            open(osp.join(output_dir, 'track.yaml'), 'w'),
            default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    if obj_detector_model is None:
        obj_detect_config_path = os.path.join(
            os.path.dirname(obj_detect_checkpoint_file),'config.yaml')
        
        obj_detect_args = nested_dict_to_namespace(yaml.unsafe_load(open(obj_detect_config_path)))
        img_transform = obj_detect_args.img_transform

        #obj_detect_args.dataset = "coco_panoptic"    # added by me 
        #obj_detect_args.dataset = "mot"

        obj_detector, _, obj_detector_post = build_model(obj_detect_args)
        # our model
        obj_detect_checkpoint = torch.load(
            obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)

        obj_detect_state_dict = obj_detect_checkpoint['model']

        new_obj_detect_state_dict = obj_detect_state_dict.copy()
        for field in new_obj_detect_state_dict:
            if field[0:4] == "detr":                
                obj_detect_state_dict[field[5:]] =  obj_detect_state_dict[field]
                del obj_detect_state_dict[field]
                new_field = field[5:]
        
        # load new layers
        track_att_checkpoint = torch.load(
            "models/mots20_train_masks/checkpoint.pth", map_location=lambda storage, loc: storage)

        track_att_state_dict = track_att_checkpoint['model']

        # add trackattention layers
        for keys in track_att_state_dict:
            if keys not in new_obj_detect_state_dict and keys[5:13] != "backbone":
                print("\n new key :", keys)
                obj_detect_state_dict[keys] = track_att_state_dict[keys]


        detr_model = False
        if detr_model:
            print("\n obj_detect_state_dict", obj_detect_state_dict["detr.class_embed.weight"])

            del obj_detect_state_dict["detr.class_embed.weight"]
            del obj_detect_state_dict["detr.class_embed.bias"]

        obj_detect_state_dict = {
            k.replace('detr.', ''): v
            for k, v in obj_detect_state_dict.items()
            if 'track_encoding' not in k} # this should delete detr. but it doesn't work


        obj_detector.load_state_dict(obj_detect_state_dict, strict=True) # Change strict
        # OBJECT DETECTOR IS THE MODEL 
        print("\n the obj_detector", obj_detector)

        if 'epoch' in obj_detect_checkpoint:
            print(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")


        obj_detector.cuda()
    else:
        obj_detector = obj_detector_model['model']
        obj_detector_post = obj_detector_model['post']
        img_transform = obj_detector_model['img_transform']

    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None
  
    tracker = Tracker(obj_detector, obj_detector_post, tracker_cfg,
                    generate_attention_maps, track_logger)

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = TrackDatasetFactory(
        dataset_name, root_dir=data_root_dir, img_transform=img_transform)
    print("\n dataset", dataset.__len__())
    for seq in dataset:
        tracker.reset()

        print(f"TRACK SEQ: {seq}")

        # frame 380 person is visible
        #start_frame = int(frame_range['start'] * len(seq))
        start_frame = int(frame_range['start']+380)
        print("\n",start_frame)

        #end_frame = int(frame_range['end'] * len(seq)) 
        end_frame = int(frame_range['start']+1380) 
        print("\n",end_frame)


        seq_loader = DataLoader(
            torch.utils.data.Subset(seq, range(start_frame, end_frame)))

        num_frames += len(seq_loader)

        results = seq.load_results(load_results_dir)

        if not results:
            start = time.time()

            for frame_id, frame_data in enumerate(tqdm.tqdm(seq_loader, file=sys.stdout)):
                #print("\n frame_data", frame_data)

                with torch.no_grad():
                    tracker.step(frame_data)

            results = tracker.get_results()
            #print("\n results", results)

            time_total += time.time() - start

           # _log.info(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
           # _log.info(f"RUNTIME: {time.time() - start :.2f} s")

            print(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
            print(f"RUNTIME: {time.time() - start :.2f} s")

            if interpolate:
                results = interpolate_tracks(results)

            if output_dir is not None:
                #_log.info(f"WRITE RESULTS")
                seq.write_results(results, output_dir)
        #else:
           # _log.info("LOAD RESULTS")

        if seq.no_gt:
            print(seq.no_gt)
            print("NO GT AVAILBLE")
        else:
            print(np.size(results))
            mot_accum = get_mot_accum(results, seq_loader)
            mot_accums.append(mot_accum)

            if verbose:
                mot_events = mot_accum.mot_events
                reid_events = mot_events[mot_events['Type'] == 'SWITCH']
                match_events = mot_events[mot_events['Type'] == 'MATCH']

                switch_gaps = []
                for index, event in reid_events.iterrows():
                    frame_id, _ = index
                    match_events_oid = match_events[match_events['OId'] == event['OId']]
                    match_events_oid_earlier = match_events_oid[
                        match_events_oid.index.get_level_values('FrameId') < frame_id]

                    if not match_events_oid_earlier.empty:
                        match_events_oid_earlier_frame_ids = \
                            match_events_oid_earlier.index.get_level_values('FrameId')
                        last_occurrence = match_events_oid_earlier_frame_ids.max()
                        switch_gap = frame_id - last_occurrence
                        switch_gaps.append(switch_gap)

                switch_gaps_hist = None
                if switch_gaps:
                    switch_gaps_hist, _ = np.histogram(
                        switch_gaps, bins=list(range(0, max(switch_gaps) + 10, 10)))
                    switch_gaps_hist = switch_gaps_hist.tolist()

                #_log.info(f'SWITCH_GAPS_HIST (bin_width=10): {switch_gaps_hist}')

        if output_dir is not None and write_images:
    
            plot_sequence(
                results, seq_loader, osp.join(output_dir, dataset_name, str(seq)),
                write_images, generate_attention_maps) 

    #if time_total:
    #    _log.info(f"RUNTIME ALL SEQS (w/o EVAL or IMG WRITE): "
    #              f"{time_total:.2f} s for {num_frames} frames "
    #              f"({num_frames / time_total:.2f} Hz)")

    if obj_detector_model is None:
       # _log.info(f"EVAL:")

        summary, str_summary = evaluate_mot_accums(
            mot_accums,
            [str(s) for s in dataset if not s.no_gt])
        print("\n summary", summary)

        return summary

    return mot_accums


if __name__ == "__main__":

    tracker_cfg = {
        # [False, 'center_distance', 'min_iou_0_5']
        "public_detections": False,
        # score threshold for detections
        "detection_obj_score_thresh": 0.4,
        # score threshold for keeping the track alive
        "track_obj_score_thresh": 0.4,
        # NMS threshold for detection
        "detection_nms_thresh": 0.9,
        # NMS theshold while tracking
        "track_nms_thresh": 0.9,
        # number of consective steps a score has to be below track_obj_score_thresh for a track to be terminated
        "steps_termination": 1,
        # distance of previous frame for multi-frame attention
        "prev_frame_dist": 1,
        # How many timesteps inactive tracks are kept and cosidered for reid
        "inactive_patience": -1,
        # How similar do image and old track need to be to be considered the same person
        "reid_sim_threshold": 0.0,
        "reid_sim_only": False,
        "reid_score_thresh": 0.4,
        "reid_greedy_matching": False}

    """ Run python src/new_track.py """
    
    """main(dataset_name="MOTS20-ALL", data_root_dir="data", \
        output_dir="data/outdir", write_images="pretty", seed=666, interpolate=False,\
        verbose=True, load_results_dir=None,  generate_attention_maps=False,\
        tracker_cfg=tracker_cfg, \
        obj_detect_checkpoint_file="models/mots20_train_masks/checkpoint.pth",
        frame_range={"start":0.0, "end":1.0}, _config="cfgs/track.yaml", _log=None, _run=None,
        obj_detector_model=None )"""
    
    """main(dataset_name="MOTS20-ALL", data_root_dir="data", \
        output_dir="data/outdir", write_images="pretty", seed=666, interpolate=False,\
        verbose=True, load_results_dir=None,  generate_attention_maps=False,\
        tracker_cfg=tracker_cfg, \
        obj_detect_checkpoint_file="/home/rpellerito/old_trackformer/models/Excav_detr_multi_frame/detr_panoptic_model.pth",
        frame_range={"start":0.0, "end":1.0}, _config="cfgs/track.yaml", _log=None, _run=None,
        obj_detector_model=None )"""

    main(dataset_name="EXCAV", data_root_dir="data/EXCAV/test", \
        output_dir="data/outdir/EXCAV_trackAtt", write_images="pretty", seed=666, interpolate=False,\
        verbose=True, load_results_dir=None,  generate_attention_maps=False,\
        tracker_cfg=tracker_cfg, \
        obj_detect_checkpoint_file="/home/rpellerito/old_trackformer/models/Excav_detr_multi_frame/detr_panoptic_model.pth",
        frame_range={"start":0.0, "end":1.0}, _config="cfgs/track.yaml", _log=None, _run=None,
        obj_detector_model=None )
 
    

