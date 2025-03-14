from ultralytics import YOLO

model = YOLO("model.pt")
outputs = model.predict(source="0", return_outputs=True)  # treat predict as a Python generator
for output in outputs:
    # each output here is a dict.
    # for detection
    print(output["det"])  # np.ndarray, (N, 6), xyxy, score, cls
    # for segmentation
    print(output["det"])  # np.ndarray, (N, 6), xyxy, score, cls
    print(output["segment"])  # List[np.ndarray] * N, bounding coordinates of masks
    # for classify
    print(output["prob"])  # np.ndarray, (num_class, ), cls prob