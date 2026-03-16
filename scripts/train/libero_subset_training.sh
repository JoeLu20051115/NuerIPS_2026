#!/bin/bash
# DreamZero LIBERO subset training launcher.
#
# This launcher is intended for a converted LIBERO subset stored in the same
# LeRobot-style layout used by the existing DreamZero training code.
#
# Recommended flow:
#   1. Build a runnable subset manifest with:
#      python scripts/data/build_libero_subset_manifest.py --suite libero_10 --num-tasks 10 --episodes-per-task 10 --verify-env
#   2. Convert that subset into LeRobot layout (video/state/action/language keys
#      matching the single-arm DreamZero schema).
#   3. Point LIBERO_DATA_ROOT to the converted subset and run this script.

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"./data/libero_subset_lerobot_100ep"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_libero_subset"}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-"./checkpoints/DreamZero-DROID"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"groot/vla/configs/deepspeed/zero2_offload.json"}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
REPORT_TO=${REPORT_TO:-tensorboard}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# ===========================================

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: Converted LIBERO subset not found at $LIBERO_DATA_ROOT"
    echo "Expected a LeRobot-style dataset root for DreamZero training."
    exit 1
fi

MODEL_OVERRIDES=()
if [ -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Using pretrained DreamZero checkpoint at $PRETRAINED_MODEL_PATH"
    MODEL_OVERRIDES+=(
        model._target_=groot.vla.model.dreamzero.base_vla.VLA.from_pretrained_for_tuning
        +model.pretrained_model_name_or_path=$PRETRAINED_MODEL_PATH
    )
else
    echo "Pretrained DreamZero checkpoint not found at $PRETRAINED_MODEL_PATH"
    echo "Falling back to Wan component initialization."
    MODEL_OVERRIDES+=(
        dit_version=$WAN_CKPT_DIR
        text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth
        image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
        vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth
        tokenizer_path=$TOKENIZER_DIR
    )
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=$REPORT_TO \
    data=dreamzero/libero_relative \
    wandb_project=dreamzero_libero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed=$DEEPSPEED_CONFIG \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    gradient_checkpointing=$GRADIENT_CHECKPOINTING \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    libero_data_root=$LIBERO_DATA_ROOT \
    "${MODEL_OVERRIDES[@]}"
