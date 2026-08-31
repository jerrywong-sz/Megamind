import torch

from src.models import build_model


def test_model_forward():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(
        pretrained=True
    ).to(device)

    images = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (4, 1)
    assert torch.isfinite(logits).all()


def test_convnext_forward():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(
        pretrained=False,
        architecture="convnext_tiny",
    ).to(device)

    images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_dinov2_forward(monkeypatch):
    class FakeDinoBackbone(torch.nn.Module):
        embed_dim = 8

        def forward(self, images):
            return torch.zeros(
                images.shape[0],
                self.embed_dim,
                device=images.device,
            )

    def fake_hub_load(
        repository,
        model_name,
        pretrained=True,
    ):
        assert repository == "facebookresearch/dinov2"
        assert model_name == "dinov2_vits14"
        assert pretrained is False

        return FakeDinoBackbone()

    monkeypatch.setattr(
        torch.hub,
        "load",
        fake_hub_load,
    )

    model = build_model(
        pretrained=False,
        architecture="dinov2_vits14",
    )

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_hybrid_effnet_dinov2_forward(monkeypatch):
    class FakeDinoBackbone(torch.nn.Module):
        embed_dim = 8

        def forward(self, images):
            return torch.zeros(
                images.shape[0],
                self.embed_dim,
                device=images.device,
            )

    def fake_hub_load(
        repository,
        model_name,
        pretrained=True,
    ):
        assert repository == "facebookresearch/dinov2"
        assert model_name == "dinov2_vits14"
        assert pretrained is False
        return FakeDinoBackbone()

    monkeypatch.setattr(
        torch.hub,
        "load",
        fake_hub_load,
    )

    model = build_model(
        pretrained=False,
        architecture="hybrid_effnet_dinov2",
    )
    images = torch.randn(2, 3, 224, 224)
    model.eval()

    with torch.no_grad():
        logits = model(images)
        ai_logits, domain_logits = model(
            images,
            alpha=1.0,
            return_domain=True,
        )

    assert logits.shape == (2, 1)
    assert ai_logits.shape == (2, 1)
    assert domain_logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(domain_logits).all()
