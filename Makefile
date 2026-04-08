# XL Fitness AI Overseer — Common Commands
# Run from the repo root on your Mac Mini.
# Usage: make <target>

REPO   = Matt-xlfitness/gym-ai-system
BRANCH = main

# ── Training ──────────────────────────────────────────────────────────────────

# Extract keypoint sequences from annotated videos (run before train)
extract:
	python3 train/extract_sequences.py \
		--annotations data/annotations/ \
		--videos data/raw/ \
		--output data/processed/

# Train the LSTM activity classifier (Mac Mini Apple Silicon)
train:
	python3 train/train_pytorch.py \
		--sequences data/processed/ \
		--review data/review/ \
		--output models/weights/activity_v1.onnx

# Train and include only annotations (no pre-extracted sequences)
train-from-raw:
	python3 train/train_pytorch.py \
		--annotations data/annotations/ \
		--videos data/raw/ \
		--output models/weights/activity_v1.onnx

# Auto-label a YouTube video using Claude Opus
autolabel:
	@echo "Usage: make autolabel URL=https://youtube.com/watch?v=xxx MACHINE=lat_pulldown"
	python3 train/auto_label_yt.py --url $(URL) --machine $(MACHINE)

# ── Deployment ────────────────────────────────────────────────────────────────

# Deploy latest ONNX model to a Pi (set PI= e.g. make deploy PI=pi@192.168.1.50)
deploy:
	@test -n "$(PI)" || (echo "Set PI=user@ip e.g. make deploy PI=pi@192.168.1.50" && exit 1)
	@test -f models/weights/activity_v1.onnx || (echo "No model found — run 'make train' first" && exit 1)
	ssh $(PI) "mkdir -p ~/xlf/models"
	scp models/weights/activity_v1.onnx $(PI):~/xlf/models/
	scp models/weights/activity_v1_meta.json $(PI):~/xlf/models/ 2>/dev/null || true
	ssh $(PI) "sudo systemctl restart xlf-overseer && echo 'Restarted OK'"
	@echo "Model deployed to $(PI)"

# View live logs from a Pi
logs:
	@test -n "$(PI)" || (echo "Set PI=user@ip" && exit 1)
	ssh $(PI) "sudo journalctl -u xlf-overseer -f"

# SSH into a Pi
ssh:
	@test -n "$(PI)" || (echo "Set PI=user@ip" && exit 1)
	ssh $(PI)

# ── Data management ───────────────────────────────────────────────────────────

# Count annotation segments per class across all JSON files
stats:
	python3 - <<'EOF'
import json, glob, collections
CLASS_NAMES = ["no_person","user_present","on_machine","good_rep","bad_rep","false_rep","resting","half_rep"]
counts = collections.Counter()
files = 0
for path in glob.glob("data/annotations/*.json"):
	with open(path) as f:
		data = json.load(f)
	for seg in data.get("annotations", []):
		counts[seg.get("class_id", -1)] += 1
	files += 1
print(f"Annotation files: {files}")
print(f"{'Class':<6} {'Label':<16} {'Count':>6}  {'Bar'}")
total = sum(counts.values())
for i, name in enumerate(CLASS_NAMES):
	n = counts[i]
	bar = "█" * min(n, 40)
	flag = " ⚠ need more" if n < 30 else ""
	print(f"  [{i}]  {name:<16} {n:>5}  {bar}{flag}")
print(f"Total: {total} segments  ({'READY' if min(counts[i] for i in range(8)) >= 30 else 'COLLECTING'})")
EOF

# List all unreviewed Pi clips
pending:
	python3 - <<'EOF'
import json, glob
pending = 0
for path in glob.glob("data/review/**/*_meta.json", recursive=True):
	with open(path) as f:
		meta = json.load(f)
	if meta.get("true_class") is None:
		pending += 1
		print(f"  PENDING  conf={meta['confidence']:.2f}  class={meta['class_name']:14}  {path}")
print(f"\nTotal pending review: {pending}")
EOF

# ── Development ───────────────────────────────────────────────────────────────

lint:
	ruff check pi/ face/ members/ train/ --ignore E501,E402

test:
	pytest train/test_synthetic.py -v

# Open annotation tool in browser
annotate:
	open pose/label.html

.PHONY: extract train train-from-raw autolabel deploy logs ssh stats pending lint test annotate
