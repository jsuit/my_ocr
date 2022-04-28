from torchvision import models
from . import resnet


def _get_backbone(model_name):
    return getattr(models, model_name)(pretrained=True)


def get_backbone(model_name, deforms=True):

    if "resnet" in model_name:
        return resnet_helper(model_name=model_name, deforms=deforms)
    else:
        raise ValueError(f"{model_name} is not supported right now")


def resnet_helper(model_name, deforms=True):

    pre_trained_model = _get_backbone(model_name=model_name)
    if deforms:
        model = getattr(resnet, f"deformable_{model_name}")()
    else:
        model = getattr(resnet, model_name)()

    model.load_state_dict(pre_trained_model.state_dict(), strict=False)
    return model
