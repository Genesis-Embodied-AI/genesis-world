import torch

import genesis as gs
from genesis.utils.repr import brief


class Tensor(torch.Tensor):
    """
    This is the genesis customization of torch's Tensor datatype. This allows our customizations, a few safety checks and also more elegant end-to-end gradient flow.
    """

    @staticmethod
    def __new__(cls, *args, scene=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.scene = scene
        obj.uid = gs.UID()
        obj.parents = []
        return obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This overrides most of torch's operations. Here, we additionally let the returned tensor inherit parent tensors' scene, and check if the parent tensors being operated are derived from the same scene.
        """

        # FIXME: This is a temporary hack. Due to some unknown reason, the unbind operation is super slow for gs.Tensor
        # when requires_grad is True. This unbind operations seems to be the last internally called operation inside
        # pytorch when we perform tensor operations. However, magically, disabling it (returning None) doesn't seem to
        # affect anything, but helps bypass the time spent on it. Need to look into this further.
        if func.__name__ == "unbind":
            return

        if kwargs is None:
            kwargs = {}

        scene = None
        parents = []
        for arg in args:
            if isinstance(arg, cls):
                parents.append(arg.uid)
                if arg.scene is not None:
                    if scene is None:
                        scene = arg.scene
                    elif scene is not arg.scene:
                        gs.raise_exception(
                            f"Tensors not derived from the same Scene object: Scene {scene.uid} vs Scene {arg.scene.uid}. Consider calling `Tensor.sceneless()` to detach a tensor from its scene if gradient flow back to the scene is not needed."
                        )

        obj = super().__torch_function__(func, types, args, kwargs)

        try:  # obj is a tuple
            for element in obj:
                if isinstance(element, cls):
                    element.scene = scene
                    element.uid = gs.UID()
                    element.parents = parents
        except Exception:  # obj is a single tensor
            if isinstance(obj, cls):
                obj.scene = scene
                obj.uid = gs.UID()
                obj.parents = parents

        return obj

    def detach(self, *args, sceneless=True, **kwargs):
        obj = super().detach(*args, **kwargs)

        if sceneless:
            obj = obj.sceneless()

        return obj

    def backward(self, *args, **kwargs):
        super().backward(*args, **kwargs)

        # if it's a tensor derived from a genesis Scene, we let the gradient keep flowing
        if self.scene is not None:
            self.scene._backward()

    def zero_grad(self):
        """
        A handy method that resembles nn.Module.zero_grad().
        """
        if self.grad is not None:
            if self.grad.grad_fn is not None:
                self.grad.detach_()
            else:
                self.grad.requires_grad_(False)
            self.grad.zero_()

    def sceneless(self):
        """
        Return a tensor detached from a scene.
        """
        obj = self.clone()
        obj.scene = None
        return obj

    def _backward_from_qd(self, qd_kernel, *args):
        grad = gs.zeros_like(self, requires_grad=False)
        qd_kernel(*args, grad)
        self.backward(gradient=grad, retain_graph=True)

    def assert_contiguous(self):
        if not self.is_contiguous():
            gs.raise_exception("Tensor not contiguous.")

    def assert_sceneless(self):
        if self.scene is not None:
            gs.raise_exception(
                "Tensor not sceneless. If you are using a tensor derived from a scene to set scene configurations back, call Tensor.sceneless() to detach it from the scene."
            )

    def __repr__(self):
        return (
            super().__repr__().replace("Tensor", "gs.tensor")[:-1]
            + f", scene={brief(self.scene)}, requires_grad={self.requires_grad})"
        )
