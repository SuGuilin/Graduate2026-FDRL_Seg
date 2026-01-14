python infer.py \
    --fusion_model_path ./FMTS_Fusion/Model_add_ca_color_loss_transform_Ours5/Infrared_Visible_Fusion/models/best_WMamba.pth \
    --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/ \
    --vi_path /home/sgl/W-Mamba/dataset/MSRS/vi/01000N.png \
    --ir_path /home/sgl/W-Mamba/dataset/MSRS/ir/01000N.png \
    --instruction "The one that kept going when the others stopped, fading into the dark where the path ends." \
    --tmp_fused_path ./Results/MSRS/out_fused_01000N.png \
    --think_output_path ./Results/MSRS/think_process_01000N.txt \
    --output_path ./Results/MSRS/out_fused_mask_01000N.png
