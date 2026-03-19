"""
Extract scene_encoder weights from a full Lightning checkpoint.

Usage:
    python -m ip.extract_scene_encoder --checkpoint ./checkpoints/model.pt --output ./checkpoints/scene_encoder.pt
"""
import argparse
import torch


def extract_scene_encoder(checkpoint_path, save_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    prefix = 'model.scene_encoder.'
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            encoder_state[k[len(prefix):]] = v

    if not encoder_state:
        available = sorted(set(k.split('.')[1] for k in state_dict.keys() if k.startswith('model.')))
        raise KeyError(f"No keys with prefix '{prefix}' found. "
                       f"Available sub-modules under 'model.': {available}")

    torch.save(encoder_state, save_path)
    print(f"Extracted {len(encoder_state)} parameters → {save_path}")

    # Verify it loads
    from ip.models.scene_encoder import SceneEncoder
    se = SceneEncoder(num_freqs=10, embd_dim=512)
    se.load_state_dict(encoder_state)
    print("Verification passed: weights load into SceneEncoder successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract scene_encoder from full checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to full Lightning checkpoint (model.pt / best.pt / last.pt)')
    parser.add_argument('--output', type=str, default='./checkpoints/scene_encoder.pt',
                        help='Where to save extracted scene_encoder weights')
    args = parser.parse_args()
    extract_scene_encoder(args.checkpoint, args.output)
