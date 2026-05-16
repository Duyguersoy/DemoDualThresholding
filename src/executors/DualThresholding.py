import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.DemoThresholdingg.src.utils.response import build_response
from components.DemoThresholdingg.src.models.PackageModel import PackageModel


class DualThresholding(Executor):
    def __init__(self,request: PackageModel,bootstrap: Component):
        super().__init__(self,request,bootstrap)
        self.requeset.model = PackageModel(**(self.request.data()))
        self.type = self.request.model.get_param("configType") or "GlobalThresholding"
        self.first_image = self.request.get_param("firstInputImage")
        self.second_image = self.request.get_param("secondInputImage")
        self.load_parameters()

    def load_parameters(self):
        if self.type == "GlobalThresholding":
            self.global_type = self.request.get_param("configGlobalType")
            if self.global_type in ["black white","black white inv","color like grey","blackening","blackening inv"]:
                self.th_value = self.request.get_param("thresholdvalue") or 127
            self.max_value = self.request.get_param("maxvalue") or 255

        elif self.type == "LocalThresholding":
            self.local_type = self.request.get_param("configLocalType")
            self.max_value = self.request.get_param("maxvalue")
            self.sub_block = self.request.get_param("subblock")
            self.off_set =  self.request.get_param("offset")

    
    @staticmethod
    def bootstrap(config:dict):
        return {}
    
    def execute(self,image):
        image = np.asarray(image).astype(np.uint8)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.type == "GlobalThresholding":
            if self.global_type == "black white":
                _, th_image = cv2.threshold(image, self.th_value, self.max_value, cv2.THRESH_BINARY)
            elif self.global_type == "black white inv":
                _, th_image = cv2.threshold(image, self.th_value, self.max_value, cv2.THRESH_BINARY_INV)
            elif self.global_type == "color like grey":
                _, th_image = cv2.threshold(image, self.th_value, self.max_value, cv2.THRESH_TRUNC)
            elif self.global_type == "blackening":
                _, th_image = cv2.threshold(image, self.th_value, self.max_value, cv2.THRESH_TOZERO)
            elif self.global_type == "blackening inv":
                _, th_image = cv2.threshold(image, self.th_value, self.max_value, cv2.THRESH_TOZERO_INV)
            elif self.global_type == "auto thresholding":
                _, th_image = cv2.threshold(image, 0, self.max_value, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        elif self.type == "LocalThresholding":
            if self.local_type == "mean":
                th_image = cv2.adaptiveThreshold(image, self.max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,self.sub_block, self.off_set)
            elif self.local_type == "gaussian":
                th_image = cv2.adaptiveThreshold(image, self.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,self.sub_block, self.off_set)

        return th_image

    def run(self):
        first_img = Image.get_frame(img=self.first_image,redis_db = self.redis_db)
        second_img = Image.get_frame(img=self.second_image,redis_db = self.redis_db)

        first_img.value = self.execute(first_img.value)
        second_img.value = self.execute(second_img.value)

        self.first_image = Image.set_frame(img=first_img.value, package_uID=self.uID, redis_db=self.redis_db)
        self.second_image = Image.set_frame(img=second_img.value, package_uID=self.uID, redis_db=self.redis_db)

        packageModel = build_response(context=self,is_dual_thresholding=True)
        return packageModel



if __name__ == "__main__":

    Executor(sys.argv[1]).run()
