python infer.py \
    --fusion_model_path ./FMTS_Fusion/Model_add_ca_color_loss_transform_Ours6/Infrared_Visible_Fusion/models/best_WMamba.pth \
    --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-7B/ \
    --vi_path /home/sgl/W-Mamba/dataset/Roadscene/vi/FLIR_08874.png \
    --ir_path /home/sgl/W-Mamba/dataset/Roadscene/ir/FLIR_08874.png \
    --instruction "Find what's trying to get across between these moving vehicles." \
    --tmp_fused_path ./Results/Roadscene/out_fused_FLIR_08874.png \
    --think_output_path ./Results/Roadscene/think_process_FLIR_08874.txt \
    --output_path ./Results/Roadscene/out_fused_mask_FLIR_08874.png
