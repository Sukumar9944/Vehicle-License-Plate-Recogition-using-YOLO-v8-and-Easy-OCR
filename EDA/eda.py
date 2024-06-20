from data_gradients.datasets.detection import YoloFormatDetectionDataset
from data_gradients.managers.detection_manager import DetectionAnalysisManager

train_loader = YoloFormatDetectionDataset(root_dir=r"G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR", images_dir=r"dataset\images\train", labels_dir=r"dataset\labels\train")
val_loader = YoloFormatDetectionDataset(root_dir=r"G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR", images_dir=r"dataset\images\val", labels_dir=r"dataset\labels\val")

analyzer = DetectionAnalysisManager(
    report_title="License Plate Object Detection and Recognition",
    train_data=train_loader,
    val_data=val_loader,
    class_names=['license_plate', 'vehicle'],
)

analyzer.run()