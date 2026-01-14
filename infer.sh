python infer.py \
    --fusion_model_path ./FMTS_Fusion/Model_add_ca_color_loss_transform_Ours5/Infrared_Visible_Fusion/models/best_WMamba.pth \
    --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/ \
    --vi_path /home/sgl/W-Mamba/dataset/MSRS/vi/00028N.png \
    --ir_path /home/sgl/W-Mamba/dataset/MSRS/ir/00028N.png \
    --instruction "bicycle" \
    --tmp_fused_path ./Results/out_fused.png \
    --think_output_path ./Results/think_process.txt \
    --output_path ./Results/out_fused_mask.png
