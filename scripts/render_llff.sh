#!/bin/bash

SCENE=flower
EXPERIMENT=llff/"$SCENE"
DATA_ROOT=data/nerf_llff_data
DATA_DIR="$DATA_ROOT"/"$SCENE"

rm exp/"$EXPERIMENT"/*
accelerate launch render.py \
  --gin_configs=configs/llff.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 120" \
  --gin_bindings="Config.render_video_fps = 30" \
  --gin_bindings="Config.factor = 4"