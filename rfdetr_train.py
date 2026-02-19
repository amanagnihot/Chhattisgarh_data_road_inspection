from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(
    # image_size=800,        # matches your tile size
    # max_image_size=800     # prevents unnecessary upscaling
)

model.train(
    dataset_dir=r"Road_inspection-8",
    epochs=100,                 # transformers benefit from longer training
    batch_size=1,               # change only if VRAM complains
    grad_accum_steps=8,         # effective batch = 8 (VERY good)
    lr=1e-4,
    use_ema=True,               # KEEP this — improves final mAP
    device=0,                   # GPU
    checkpoint_interval=10,     # save less often → faster training
    run="rfdetr_medium_road_defects",
    output_dir="./new_output/rfdetr_medium_100"
)
