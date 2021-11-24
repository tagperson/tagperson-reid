import torch

from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image.to(self.model.device)}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)

        # depends on prediction
        if not self.cfg.TEST.ATTRIBUTE.ENABLED:
            # return feature 2048 d
            return predictions.cpu()
        else:
            # return the attribute head and option head

            # TODO: attribute

            # option
            outputs = {}
            outputs['option_outputs'] = []
            for option_idx, _ in enumerate(predictions['option_outputs']):
                cur_option = predictions['option_outputs'][option_idx]['pred_class_logits'].cpu()
                cur_option = cur_option[:, 0]
                outputs['option_outputs'].append(cur_option)

            return outputs

