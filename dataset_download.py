# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="HBCxBn84cprnT2Ux1rw0")
project = rf.workspace("novametrics").project("road_inspection-he6og")
version = project.version(8)
dataset = version.download("coco-segmentation")
                