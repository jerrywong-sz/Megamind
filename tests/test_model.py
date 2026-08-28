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
