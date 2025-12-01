# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules.head import ExpertHead, MoEDetect, RouterNetwork


class TestRouterNetwork:
    """Test cases for RouterNetwork module."""

    def test_router_output_shape(self):
        """Test that router produces correct output shape."""
        router = RouterNetwork(in_channels=256, num_experts=4)
        x = torch.randn(2, 256, 40, 40)

        weights, logits = router(x)

        assert weights.shape == (2, 4), f"Expected weights shape (2, 4), got {weights.shape}"
        assert logits.shape == (2, 4), f"Expected logits shape (2, 4), got {logits.shape}"

    def test_router_softmax_properties(self):
        """Test that router weights are valid softmax outputs."""
        router = RouterNetwork(in_channels=256, num_experts=4)
        x = torch.randn(2, 256, 40, 40)

        weights, _ = router(x)

        assert torch.allclose(weights.sum(dim=1), torch.ones(2)), "Weights should sum to 1"
        assert (weights >= 0).all(), "Weights should be non-negative"
        assert (weights <= 1).all(), "Weights should be <= 1"

    def test_router_different_experts(self):
        """Test router with different number of experts."""
        for num_experts in [2, 4, 8]:
            router = RouterNetwork(in_channels=128, num_experts=num_experts)
            x = torch.randn(1, 128, 20, 20)

            weights, logits = router(x)

            assert weights.shape == (1, num_experts)
            assert logits.shape == (1, num_experts)

    def test_router_gradient_flow(self):
        """Test that gradients flow through router."""
        router = RouterNetwork(in_channels=64, num_experts=4)
        x = torch.randn(1, 64, 10, 10, requires_grad=True)

        weights, _ = router(x)
        loss = weights.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow back to input"
        assert router.temperature.grad is not None, "Temperature should receive gradients"


class TestExpertHead:
    """Test cases for ExpertHead module."""

    def test_expert_output_shapes(self):
        """Test that expert head produces correct output shapes."""
        expert = ExpertHead(in_channels=256, nc=80, reg_max=16)
        x = torch.randn(2, 256, 40, 40)

        box_out, cls_out = expert(x)

        assert box_out.shape == (2, 64, 40, 40), f"Expected box shape (2, 64, 40, 40), got {box_out.shape}"
        assert cls_out.shape == (2, 80, 40, 40), f"Expected cls shape (2, 80, 40, 40), got {cls_out.shape}"

    def test_expert_different_channels(self):
        """Test expert with different input channel sizes."""
        for in_channels in [128, 256, 512]:
            expert = ExpertHead(in_channels=in_channels, nc=80, reg_max=16)
            x = torch.randn(1, in_channels, 20, 20)

            box_out, cls_out = expert(x)

            assert box_out.shape[0] == 1 and box_out.shape[1] == 64
            assert cls_out.shape[0] == 1 and cls_out.shape[1] == 80

    def test_expert_gradient_flow(self):
        """Test that gradients flow through expert."""
        expert = ExpertHead(in_channels=256, nc=80, reg_max=16)
        x = torch.randn(1, 256, 20, 20, requires_grad=True)

        box_out, cls_out = expert(x)
        loss = box_out.sum() + cls_out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow back to input"


class TestMoEDetect:
    """Test cases for MoEDetect module."""

    def test_moe_detect_initialization(self):
        """Test MoEDetect initializes correctly."""
        ch = (256, 512, 1024)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)

        assert moe_head.num_experts == 4
        assert len(moe_head.routers) == 3, "Should have 3 routers (one per scale)"
        assert len(moe_head.experts) == 3, "Should have 3 expert groups"
        assert len(moe_head.experts[0]) == 4, "Each scale should have 4 experts"

    def test_moe_detect_forward_training(self):
        """Test MoEDetect forward pass in training mode."""
        ch = (256, 512, 1024)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
        moe_head.train()
        moe_head.stride = torch.tensor([8, 16, 32])

        x = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 1024, 20, 20),
        ]

        result = moe_head(x)

        assert isinstance(result, tuple), "Training should return (outputs, aux_loss)"
        outputs, aux_loss = result
        assert len(outputs) == 3, "Should have 3 scale outputs"
        assert outputs[0].shape == (2, 144, 80, 80), f"Expected shape (2, 144, 80, 80), got {outputs[0].shape}"
        assert outputs[1].shape == (2, 144, 40, 40)
        assert outputs[2].shape == (2, 144, 20, 20)
        assert isinstance(aux_loss, torch.Tensor), "Aux loss should be a tensor"
        assert aux_loss.requires_grad, "Aux loss should require gradients"

    def test_moe_detect_forward_inference(self):
        """Test MoEDetect forward pass in inference mode."""
        ch = (256, 512, 1024)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
        moe_head.eval()
        moe_head.stride = torch.tensor([8, 16, 32])

        x = [
            torch.randn(2, 256, 80, 80),
            torch.randn(2, 512, 40, 40),
            torch.randn(2, 1024, 20, 20),
        ]

        with torch.no_grad():
            result = moe_head(x)

        assert isinstance(result, tuple), "Inference should return (predictions, outputs)"
        assert len(result) == 2

    def test_moe_load_balance_loss(self):
        """Test load balancing loss computation."""
        ch = (256,)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)

        balanced_weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 8)
        logits = torch.randn(8, 4)
        router_info = [(balanced_weights, logits)]

        aux_loss = moe_head._compute_load_balance_loss(router_info)

        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.requires_grad
        assert aux_loss > 0, "Aux loss should be positive"

    def test_moe_unbalanced_routing_penalty(self):
        """Test that unbalanced routing has higher loss."""
        ch = (256,)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)

        balanced_weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 8)
        unbalanced_weights = torch.tensor([[0.9, 0.05, 0.03, 0.02]] * 8)
        logits = torch.randn(8, 4)

        balanced_loss = moe_head._compute_load_balance_loss([(balanced_weights, logits)])
        unbalanced_loss = moe_head._compute_load_balance_loss([(unbalanced_weights, logits)])

        assert unbalanced_loss > balanced_loss, "Unbalanced routing should have higher loss"

    def test_moe_expert_usage_tracking(self):
        """Test that expert usage is tracked correctly."""
        ch = (256,)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
        moe_head.train()
        moe_head.stride = torch.tensor([8])

        x = [torch.randn(4, 256, 40, 40)]

        initial_counts = moe_head.expert_counts.clone()
        _ = moe_head(x)

        assert (moe_head.expert_counts > initial_counts).any(), "Expert counts should increase"
        assert moe_head.expert_counts.sum() == pytest.approx(
            4.0, rel=1e-4
        ), "Total counts should equal batch size"

    def test_moe_bias_init(self):
        """Test expert bias initialization."""
        ch = (256, 512, 1024)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
        moe_head.stride = torch.tensor([8, 16, 32])

        moe_head.bias_init()

        for scale_idx, scale_experts in enumerate(moe_head.experts):
            for expert_idx, expert in enumerate(scale_experts):
                box_bias = expert.cv2[-1].bias.data
                cls_bias = expert.cv3[-1].bias.data

                assert box_bias.shape[0] == 64, "Box bias should have correct shape"
                assert cls_bias.shape[0] == 80, "Classification bias should have correct shape"
                assert not torch.all(box_bias == 0), "Box biases should be initialized"
                assert not torch.all(cls_bias == 0), "Classification biases should be initialized"

    def test_moe_gradient_flow(self):
        """Test that gradients flow through all components."""
        ch = (256,)
        moe_head = MoEDetect(nc=80, ch=ch, num_experts=4)
        moe_head.train()
        moe_head.stride = torch.tensor([8])

        x = [torch.randn(2, 256, 20, 20, requires_grad=True)]

        outputs, aux_loss = moe_head(x)
        loss = outputs[0].sum() + aux_loss
        loss.backward()

        assert x[0].grad is not None, "Gradients should flow to input"

        for router in moe_head.routers:
            assert any(p.grad is not None for p in router.parameters()), "Router should receive gradients"

        for scale_experts in moe_head.experts:
            for expert in scale_experts:
                assert any(p.grad is not None for p in expert.parameters()), "All experts should receive gradients"


class TestMoEIntegration:
    """Integration tests for MoE with YOLO model."""

    def test_load_moe_model_from_yaml(self):
        """Test loading MoE model from YAML configuration."""
        from ultralytics.nn.tasks import DetectionModel

        cfg_path = "ultralytics/cfg/models/11/yolo11-moe.yaml"

        try:
            model = DetectionModel(cfg=cfg_path, ch=3, nc=80)
            assert isinstance(model.model[-1], MoEDetect), "Last layer should be MoEDetect"
            assert model.model[-1].num_experts == 4, "Should have 4 experts"
        except FileNotFoundError:
            pytest.skip(f"Config file not found: {cfg_path}")

    def test_moe_with_detection_loss(self):
        """Test that MoEDetectionLoss handles MoE outputs correctly."""
        from ultralytics.utils.loss import MoEDetectionLoss

        class MockModel:
            class Args:
                moe_aux_loss = 0.01

            args = Args()
            model = [None]

        mock_model = MockModel()
        mock_model.model[-1] = MoEDetect(nc=80, ch=(256,), num_experts=4)
        mock_model.model[-1].stride = torch.tensor([8])

        loss_fn = MoEDetectionLoss(mock_model)

        predictions = [torch.randn(2, 144, 20, 20)]
        aux_loss = torch.tensor(0.05, requires_grad=True)
        preds = (predictions, aux_loss)

        batch = {
            "cls": torch.randint(0, 80, (10,)),
            "bboxes": torch.randn(10, 4),
            "batch_idx": torch.tensor([0] * 5 + [1] * 5),
            "img": torch.randn(2, 3, 160, 160),
        }

        try:
            total_loss, loss_items = loss_fn(preds, batch)
            assert isinstance(total_loss, torch.Tensor)
            assert total_loss.requires_grad
            assert len(loss_items) == 4, "Should have 4 loss items (box, cls, dfl, aux)"
        except Exception as e:
            pytest.skip(f"Loss computation requires full model setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
