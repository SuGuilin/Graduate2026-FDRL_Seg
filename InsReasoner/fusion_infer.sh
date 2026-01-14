#python inference_scripts/infer_multi_object.py --image_path "./assets/rlfusion.png" --text "Who in this picture looks like they’ve just finished work and are heading home alone?" --reasoning_model_path ./pretrained_models/InsReasoner-3B-1k/
#python inference_scripts/infer_multi_object.py --image_path "./fusion_results/01000N.png" --text "The one that’s barely visible where the path runs out past the lights and the group, all on its own." --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/
#python inference_scripts/infer_multi_object.py --image_path "./fusion_results/01000N.png" --text "The one that kept going when the others stopped, fading into the dark where the path ends." --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/
#python inference_scripts/infer_multi_object.py --image_path "./fusion_results/00196.png" --text "Which individual requires the most alertness in this road situation?" --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-3B-1k/
#python inference_scripts/infer_multi_object.py --image_path "./fusion_results/FLIR_08874.png" --text "Find what's trying to get across between these moving vehicles." --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-7B/
python inference_scripts/infer_multi_object.py --image_path "/home/sgl/W-Mamba/ablation_fusion_results/full_model2.png" --text "bicycle" --reasoning_model_path /data3/suguilin/pretrained_models/InsReasoner-7B/ #InsReasoner-3B-1k/


