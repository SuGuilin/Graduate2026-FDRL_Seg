python infer.py \
    --fusion_model_path ./FMTS_Fusion/Model_add_ca_color_loss_transform_Ours6/Infrared_Visible_Fusion/models/best_WMamba.pth \
    --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/ \
    --vi_path /home/sgl/W-Mamba/dataset/M3FD_Fusion/vi/00196.png \
    --ir_path /home/sgl/W-Mamba/dataset/M3FD_Fusion/ir/00196.png \
    --instruction "Which individual requires the most alertness in this road situation?" \
    --tmp_fused_path ./Results/M3FD_Fusion/out_fused_00196.png \
    --think_output_path ./Results/M3FD_Fusion/think_process_00196.txt \
    --output_path ./Results/M3FD_Fusion/out_fused_mask_00196.png
