
from sdks.novavision.src.helper.package import PackageHelper
from components.DemoThresholdingg.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    ThresholdingExecutor,
    ThresholdingResponse,
    ThresholdingOutputs,
    OutputImage,
    OutputImageSecond,
    
    DualThresholdingResponse,
    DualThresholdingRequest,
    DualThresholdingExecutor,
    DualThresholdingOutputs,
    OutputImageSecond,
)


def build_response(context,is_dual_thresholding:bool=False):
    if is_dual_thresholding:
        output_image = OutputImage(value=context.first_image)
        sec_output_image = OutputImageSecond(value=context.second_image)
        outputs = DualThresholdingOutputs(
            outputImage=output_image,
            secondOutputImage=sec_output_image
        )
        response = DualThresholdingResponse(outputs=outputs)
        selected_executor = DualThresholdingExecutor(value=response)
        return selected_executor
    
    outputImage = OutputImage(value=context.image)
    Outputs = ThresholdingOutputs(outputImage=outputImage)
    normalizationResponse = ThresholdingResponse(outputs=Outputs)
    normalizationExecutor = ThresholdingExecutor(value=normalizationResponse)
    executor = ConfigExecutor(value=normalizationExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
