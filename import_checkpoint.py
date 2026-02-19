import torch

CHECKPOINT_PATH = r"/media/user/New Volume1/Sakshi/chattishgarh/new_output/rfdetr_medium/checkpoint_best_total.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

print("=" * 60)
print("TOP-LEVEL KEYS IN CHECKPOINT:")
print("=" * 60)
for k in checkpoint.keys():
    print(f"  {k}")

print("\n" + "=" * 60)
print("ARGS / CONFIG (if saved):")
print("=" * 60)
for key in ["args", "config", "cfg", "opt", "hparams"]:
    if key in checkpoint:
        print(f"\n[{key}]:")
        print(checkpoint[key])

print("\n" + "=" * 60)
print("RELEVANT WEIGHT SHAPES FROM STATE DICT:")
print("=" * 60)
if "model" in checkpoint:
    state_dict = checkpoint["model"]
    keys_of_interest = [
        "refpoint_embed.weight",
        "query_feat.weight",
        "backbone.0.encoder.encoder.embeddings.position_embeddings",
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
    ]
    for k in keys_of_interest:
        if k in state_dict:
            print(f"  {k}: {state_dict[k].shape}")
        else:
            print(f"  {k}: NOT FOUND")

    print("\n--- All keys containing 'query' ---")
    for k, v in state_dict.items():
        if "query" in k.lower():
            print(f"  {k}: {v.shape}")

    print("\n--- All keys containing 'mask' ---")
    for k, v in state_dict.items():
        if "mask" in k.lower():
            print(f"  {k}: {v.shape}")
else:
    print("No 'model' key found. Printing all keys with shapes:")
    for k, v in checkpoint.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")

print("\n" + "=" * 60)
print("HOW TO INTERPRET:")
print("=" * 60)
print("refpoint_embed.weight shape[0]  => num_queries")
print("position_embeddings shape[1]-1  => num_patches")
print("patch_embeddings projection shape[2] => patch_size")
print("num_patches = (image_size / patch_size) ^ 2")
print("=> image_size = sqrt(num_patches) * patch_size")
print("=" * 60)