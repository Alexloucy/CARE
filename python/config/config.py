# Embedding type names used across the pipeline
# These must match between detection and ReID for caching to work

# Raw DINO features (from classification/detection)
RAW_EMBEDDING_TYPE = 'dinov3_raw'

# ReID embeddings (after adapter layer) - {species} replaced at runtime
# Format: f'dinov3_reid_{species}'
REID_EMBEDDING_PREFIX = 'dinov3_reid_'

# Unused - set to a non-existent name to disable adapter-only path
# The adapter-only approach produces slightly different results than full model
# Set this to RAW_EMBEDDING_TYPE to re-enable if backbones are updated
RAW_FOR_ADAPTER_TYPE = 'dinov3_raw_disabled'  # Non-existent, forces full model
